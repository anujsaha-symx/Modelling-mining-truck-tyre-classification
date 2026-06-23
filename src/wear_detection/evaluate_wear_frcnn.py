import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from utils import (
    WearDetectionDataset, get_transform, get_model,
    collate_fn, load_checkpoint, CLASS_NAMES, NUM_CLASSES
)

CLASS_LIST = ['Background', 'Good-Tire', 'Bad-Tire', 'Non-Tire']

@torch.no_grad()
def evaluate(model, dataloader, device, score_threshold=0.15, iou_threshold=0.5):
    model.eval()
    all_gt_labels = []
    all_pred_labels = []
    all_pred_scores = []
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for i, output in enumerate(outputs):
            gt_labels = targets[i]['labels'].cpu().numpy()
            pred_boxes = output['boxes'].cpu().numpy()
            pred_scores = output['scores'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            mask = pred_scores >= 0.05
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_labels = pred_labels[mask]
            gt_boxes = targets[i]['boxes'].cpu().numpy()
            matched_gt = set()
            matched_pred = set()
            for p_idx, (p_box, p_label, p_score) in enumerate(
                zip(pred_boxes, pred_labels, pred_scores)
            ):
                if p_score < score_threshold:
                    continue
                best_iou = 0
                best_gt = -1
                for g_idx, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if g_idx in matched_gt:
                        continue
                    iou = box_iou(
                        torch.from_numpy(p_box).unsqueeze(0),
                        torch.from_numpy(g_box).unsqueeze(0)
                    ).item()
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = g_idx
                if best_iou >= iou_threshold and best_gt >= 0:
                    matched_pred.add(p_idx)
                    matched_gt.add(best_gt)
                    all_gt_labels.append(int(gt_labels[best_gt]))
                    all_pred_labels.append(int(p_label))
                    all_pred_scores.append(float(p_score))
                else:
                    all_gt_labels.append(0)
                    all_pred_labels.append(int(p_label))
                    all_pred_scores.append(float(p_score))
            for g_idx, g_label in enumerate(gt_labels):
                if g_idx not in matched_gt:
                    all_gt_labels.append(int(g_label))
                    all_pred_labels.append(0)
                    all_pred_scores.append(0.0)
    return all_gt_labels, all_pred_labels, all_pred_scores

def compute_per_class_metrics(gt_labels, pred_labels, num_classes):
    metrics = {}
    for c in range(1, num_classes):
        tp = sum(1 for g, p in zip(gt_labels, pred_labels) if g == c and p == c)
        fp = sum(1 for g, p in zip(gt_labels, pred_labels) if g != c and p == c)
        fn = sum(1 for g, p in zip(gt_labels, pred_labels) if g == c and p != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[CLASS_LIST[c]] = {
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1': round(f1, 4),
            'TP': tp,
            'FP': fp,
            'FN': fn,
        }
    return metrics

def compute_confusion_matrix(gt_labels, pred_labels, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for g, p in zip(gt_labels, pred_labels):
        cm[g, p] += 1
    return cm

def compute_overall_accuracy(gt_labels, pred_labels):
    correct = sum(1 for g, p in zip(gt_labels, pred_labels) if g == p)
    return correct / len(gt_labels) if gt_labels else 0.0

@torch.no_grad()
def compute_map(model, dataloader, device):
    model.eval()
    all_gt_boxes = []
    all_gt_labels = []
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for i, output in enumerate(outputs):
            all_gt_boxes.append(targets[i]['boxes'].cpu())
            all_gt_labels.append(targets[i]['labels'].cpu())
            pred_boxes = output['boxes'].cpu()
            pred_scores = output['scores'].cpu()
            pred_labels = output['labels'].cpu()
            mask = pred_scores >= 0.05
            all_pred_boxes.append(pred_boxes[mask])
            all_pred_scores.append(pred_scores[mask])
            all_pred_labels.append(pred_labels[mask])
    iou_thresholds_50 = [0.5]
    iou_thresholds_95 = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    aps_50 = []
    aps_95 = []
    for cls in range(1, NUM_CLASSES):
        gt_per_img = []
        pred_per_img = []
        for i in range(len(all_gt_boxes)):
            gt_boxes = all_gt_boxes[i]
            gt_labels = all_gt_labels[i]
            gt_mask = gt_labels == cls
            gt_per_img.append(gt_boxes[gt_mask])
            pred_boxes = all_pred_boxes[i]
            pred_scores = all_pred_scores[i]
            pred_labels = all_pred_labels[i]
            pred_mask = pred_labels == cls
            pred_per_img.append({
                'boxes': pred_boxes[pred_mask],
                'scores': pred_scores[pred_mask],
                'labels': pred_labels[pred_mask],
            })
        ap50 = _compute_ap(gt_per_img, pred_per_img, iou_thresholds_50)
        ap95 = _compute_ap(gt_per_img, pred_per_img, iou_thresholds_95)
        aps_50.append(ap50)
        aps_95.append(ap95)
        class_name = CLASS_LIST[cls]
        print(f'  {class_name:15s}  AP50: {ap50:.4f}  AP50:95: {ap95:.4f}')
    map50 = np.mean(aps_50) if aps_50 else 0.0
    map5095 = np.mean(aps_95) if aps_95 else 0.0
    return map50, map5095

def _compute_ap(gt_per_img, pred_per_img, iou_thresholds):
    all_detections = []
    for img_idx, preds in enumerate(pred_per_img):
        for j in range(len(preds['boxes'])):
            all_detections.append({
                'img_idx': img_idx,
                'box': preds['boxes'][j],
                'score': preds['scores'][j].item(),
                'used': False,
            })
    all_detections.sort(key=lambda x: x['score'], reverse=True)
    num_gt = sum(len(g) for g in gt_per_img)
    tp = np.zeros(len(all_detections))
    fp = np.zeros(len(all_detections))
    for d_idx, det in enumerate(all_detections):
        img_idx = det['img_idx']
        gt_boxes = gt_per_img[img_idx]
        best_iou = 0
        best_gt = -1
        for g_idx in range(len(gt_boxes)):
            iou = box_iou(det['box'].unsqueeze(0), gt_boxes[g_idx].unsqueeze(0)).item()
            if iou > best_iou:
                best_iou = iou
                best_gt = g_idx
        max_iou_thresh = max(iou_thresholds)
        if best_iou >= max_iou_thresh and best_gt >= 0:
            if not hasattr(gt_per_img[img_idx], '_matched'):
                gt_per_img[img_idx]._matched = set()
            if best_gt not in gt_per_img[img_idx]._matched:
                tp[d_idx] = 1
                gt_per_img[img_idx]._matched.add(best_gt)
            else:
                fp[d_idx] = 1
        else:
            fp[d_idx] = 1
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / num_gt if num_gt > 0 else np.zeros_like(tp_cum)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-10)
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        p = np.max(prec[rec >= t]) if np.any(rec >= t) else 0.0
        ap += p / 11
    return ap

def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_LIST,
        yticklabels=CLASS_LIST,
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Wear Detection')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    coco_path = 'datasets/annotated/annotations_wear.coco.json'
    test_csv = 'datasets/splits/test.csv'
    checkpoint_path = 'outputs/wear_detection/checkpoints/best_wear_frcnn.pt'

    if not os.path.isfile(coco_path):
        print(f'ERROR: {coco_path} not found. Run convert_annotations.py first.')
        return
    if not os.path.isfile(checkpoint_path):
        print(f'ERROR: {checkpoint_path} not found. Run train_wear_frcnn.py first.')
        return

    test_dataset = WearDetectionDataset(
        coco_path, test_csv, transforms=get_transform(train=False)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=0,
        collate_fn=collate_fn
    )
    print(f'Test samples: {len(test_dataset)}')

    model = get_model(num_classes=NUM_CLASSES, freeze_backbone=True).to(device)
    load_checkpoint(model, checkpoint_path, device=device)
    print(f'Loaded checkpoint from {checkpoint_path}')

    print('\nComputing detection-level metrics...')
    gt_labels, pred_labels, pred_scores = evaluate(model, test_loader, device)

    per_class = compute_per_class_metrics(gt_labels, pred_labels, NUM_CLASSES)
    overall_acc = compute_overall_accuracy(gt_labels, pred_labels)
    cm = compute_confusion_matrix(gt_labels, pred_labels, NUM_CLASSES)

    print(f'\nOverall Accuracy: {overall_acc:.4f}')
    print(f'\nPer-Class Metrics:')
    for cls_name, m in per_class.items():
        print(f'  {cls_name:15s}  P={m["Precision"]:.4f}  R={m["Recall"]:.4f}  F1={m["F1"]:.4f}  '
              f'TP={m["TP"]}  FP={m["FP"]}  FN={m["FN"]}')

    print('\nComputing mAP...')
    map50, map5095 = compute_map(model, test_loader, device)
    print(f'\nmAP50:    {map50:.4f}')
    print(f'mAP50:95: {map5095:.4f}')

    output_dir = 'outputs/wear_detection'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/failure_cases', exist_ok=True)
    os.makedirs(f'{output_dir}/predictions', exist_ok=True)

    metrics_data = {
        'OverallAccuracy': overall_acc,
        'mAP50': map50,
        'mAP50:95': map5095,
        'PerClassMetrics': per_class,
        'ConfusionMatrix': cm.tolist(),
    }
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f'\nMetrics saved to {output_dir}/metrics.json')

    cm_path = f'{output_dir}/confusion_matrix.png'
    plot_confusion_matrix(cm, cm_path)
    print(f'Confusion matrix saved to {cm_path}')

if __name__ == '__main__':
    main()