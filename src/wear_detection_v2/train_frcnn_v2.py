import os
import time

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from coco_dataset import WearDetectionDatasetV2, collate_fn
from model import get_frcnn_model_v2


def get_transform(train):
    transforms = [torchvision.transforms.ToTensor()]
    if train:
        transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))
    return torchvision.transforms.Compose(transforms)


def train_one_epoch(model, dataloader, optimizer, device, epoch=None):
    model.train()
    total_loss = 0
    n_batches = len(dataloader)

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if (batch_idx + 1) % 25 == 0:
            ep = f'[Epoch {epoch}] ' if epoch else ''
            print(f'{ep}Batch {batch_idx+1}/{n_batches} | Loss: {losses.item():.4f}')

    return total_loss / n_batches


@torch.no_grad()
def validate(model, dataloader, device):
    model.train()
    total_loss = 0
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

    return total_loss / len(dataloader)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    coco_path = 'datasets/annotated/annotations_wear_v2.coco.json'
    train_csv = 'datasets/splits/train.csv'
    val_csv = 'datasets/splits/val.csv'

    if not os.path.isfile(coco_path):
        print(f'ERROR: {coco_path} not found. Run convert_annotations_v2.py first.')
        return

    train_dataset = WearDetectionDatasetV2(
        coco_path, train_csv, transforms=get_transform(train=True)
    )
    val_dataset = WearDetectionDatasetV2(
        coco_path, val_csv, transforms=get_transform(train=False)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=0,
        collate_fn=collate_fn
    )

    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')

    model = get_frcnn_model_v2(num_classes=4, pretrained=True).to(device)

    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if 'backbone' in name:
                backbone_params.append(p)
            else:
                head_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-4},
    ], weight_decay=0.0001)

    checkpoint_dir = 'outputs/wear_detection_v2/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_epochs = 5
    patience = 2
    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 1

    if args.resume:
        last_ckpt = os.path.join(checkpoint_dir, 'last_frcnn_v2.pt')
        best_ckpt = os.path.join(checkpoint_dir, 'best_frcnn_v2.pt')
        if os.path.isfile(last_ckpt):
            ckpt = torch.load(last_ckpt, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            if os.path.isfile(best_ckpt):
                best_ckpt_data = torch.load(best_ckpt, map_location=device)
                best_val_loss = best_ckpt_data.get('val_loss', float('inf'))
            resumed_epoch = ckpt.get('epoch', '?')
            print(f'Resumed from epoch {resumed_epoch}, starting at epoch {start_epoch}')
        else:
            print(f'No checkpoint found at {last_ckpt}, starting from scratch')

    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    if args.resume and start_epoch > 1:
        for _ in range(start_epoch - 1):
            scheduler.step()

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch=epoch)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        print(
            f'Epoch {epoch:2d}/{num_epochs} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val Loss: {val_loss:.4f} | '
            f'LR: {current_lr:.2e} | '
            f'Time: {epoch_time:.0f}s'
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(checkpoint_dir, 'last_frcnn_v2.pt'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(checkpoint_dir, 'best_frcnn_v2.pt'))
            print(f'  -> New best model (val_loss={val_loss:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    print('Training complete!')


if __name__ == '__main__':
    main()
