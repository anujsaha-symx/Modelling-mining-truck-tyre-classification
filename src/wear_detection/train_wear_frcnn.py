import os
import time
import random
import torch
from torch.utils.data import DataLoader, Subset
from utils import (
    WearDetectionDataset, get_transform, get_model,
    collate_fn, save_checkpoint, NUM_CLASSES
)

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    coco_path = 'datasets/annotated/annotations_wear.coco.json'
    train_csv = 'datasets/splits/train.csv'
    val_csv = 'datasets/splits/val.csv'

    if not os.path.isfile(coco_path):
        print(f'ERROR: {coco_path} not found. Run convert_annotations.py first.')
        return
    full_train = WearDetectionDataset(
        coco_path, train_csv, transforms=get_transform(train=True)
    )
    val_dataset = WearDetectionDataset(
        coco_path, val_csv, transforms=get_transform(train=False)
    )
    train_labels = []
    for i in range(len(full_train)):
        _, t = full_train[i]
        has_non_tire = (t['labels'] == 3).sum().item() > 0
        train_labels.append(has_non_tire)
    non_tire_idx = [i for i, v in enumerate(train_labels) if v]
    tire_idx = [i for i, v in enumerate(train_labels) if not v]
    random.seed(42)
    n_tire = min(len(non_tire_idx), len(tire_idx))
    sampled_tire = random.sample(tire_idx, n_tire)
    train_indices = non_tire_idx + sampled_tire
    random.shuffle(train_indices)
    train_dataset = Subset(full_train, train_indices)
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=0,
        collate_fn=collate_fn
    )
    n_train = len(train_dataset)
    n_non_tire = len(non_tire_idx)
    print(f'Train samples: {n_train} ({n_non_tire} Non-Tire, {n_train - n_non_tire} Tire)')
    print(f'Val samples: {len(val_dataset)}')
    model = get_model(num_classes=NUM_CLASSES, freeze_backbone=True).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable params (frozen backbone): {trainable:,} / {total:,}')
    old_ckpt = 'outputs/detection/checkpoints/best_frcnn.pt'
    if os.path.isfile(old_ckpt):
        ckpt = torch.load(old_ckpt, map_location=device)
        old_keys = list(ckpt['model_state_dict'].keys())
        new_keys = list(model.state_dict().keys())
        old_weight = ckpt['model_state_dict']
        new_weight = model.state_dict()
        for key in old_weight:
            if key in new_weight and old_weight[key].shape == new_weight[key].shape:
                new_weight[key] = old_weight[key]
        model.load_state_dict(new_weight)
        print(f'Loaded checkpoint from {old_ckpt} with partial weight transfer')
    else:
        print('No checkpoint found, starting from pretrained COCO weights')
    for param in model.backbone.parameters():
        param.requires_grad = True
    print('Backbone UNFROZEN for fine-tuning')
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params (all unfrozen): {trainable:,} / {total:,}')
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    num_epochs = 15
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_dir = 'outputs/wear_detection/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch=epoch)
        val_loss = validate(model, val_loader, device)
        scheduler.step()
        epoch_time = time.time() - epoch_start
        print(
            f'Epoch {epoch:2d}/{num_epochs} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val Loss: {val_loss:.4f} | '
            f'Time: {epoch_time:.0f}s'
        )
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            os.path.join(checkpoint_dir, 'last_wear_frcnn.pt')
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(checkpoint_dir, 'best_wear_frcnn.pt')
            )
            print(f'  -> New best model (val_loss={val_loss:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
    print('Training complete!')

if __name__ == '__main__':
    main()