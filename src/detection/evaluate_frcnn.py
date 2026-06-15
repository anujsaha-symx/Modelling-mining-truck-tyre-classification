import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from coco_dataset import COCODetectionDataset, get_transform
from train_frcnn import get_model
from utils import collate_fn, load_checkpoint

@torch.no_grad()
def compute_image_level_metrics(model, dataloader, device, score_threshold=0.5):
    model.eval()
    tp = fp = fn = 0
    total = len(dataloader.dataset)

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for i, output in enumerate(outputs):
            gt_labels = targets[i]['labels']

            pred_labels = output['labels'].cpu()
            pred_scores = output['scores'].cpu()

            mask = pred_scores >= score_threshold
            pred_labels = pred_labels[mask]

            gt_tire = (gt_labels == 1).sum().item() > 0
            pred_tire = (pred_labels == 1).sum().item() > 0

            if gt_tire and pred_tire:
                tp += 1
            elif not gt_tire and pred_tire:
                fp += 1
            elif gt_tire and not pred_tire:
                fn += 1

    tn = total - (tp + fp + fn)
    return tp, fp, fn, tn

def plot_confusion_matrix(tp, fp, fn, tn, save_path):
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Non-Tire', 'Tire'],
        yticklabels=['Non-Tire', 'Tire'],
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Tire Gatekeeper')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    coco_path = 'datasets/annotated/annotations.coco.json'
    test_csv = 'datasets/splits/test.csv'
    checkpoint_path = 'outputs/detection/checkpoints/best_frcnn.pt'

    test_dataset = COCODetectionDataset(
        coco_path, test_csv, transforms=get_transform(train=False)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=0,
        collate_fn=collate_fn
    )
    print(f'Test samples: {len(test_dataset)}')

    model = get_model(num_classes=3, freeze_backbone=True).to(device)
    load_checkpoint(model, checkpoint_path, device=device)
    print('Loaded checkpoint from', checkpoint_path)

    print('\nComputing image-level gatekeeper metrics...')
    tp, fp, fn, tn = compute_image_level_metrics(model, test_loader, device)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall:    {recall:.4f}')
    print(f'  F1:        {f1:.4f}')
    print(f'  Accuracy:  {accuracy:.4f}')
    print(f'  TP={tp}  FP={fp}  FN={fn}  TN={tn}')

    os.makedirs('outputs/detection', exist_ok=True)

    metrics_data = {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Accuracy': accuracy,
        'TruePositives': tp,
        'FalsePositives': fp,
        'FalseNegatives': fn,
        'TrueNegatives': tn,
    }
    with open('outputs/detection/metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print('\nMetrics saved to outputs/detection/metrics.json')

    cm_path = 'outputs/detection/confusion_matrix.png'
    plot_confusion_matrix(tp, fp, fn, tn, cm_path)
    print(f'Confusion matrix saved to {cm_path}')

if __name__ == '__main__':
    main()