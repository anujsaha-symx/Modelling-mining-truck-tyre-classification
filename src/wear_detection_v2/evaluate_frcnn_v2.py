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
from tqdm import tqdm

from coco_dataset import WearDetectionDatasetV2, collate_fn
from model import get_frcnn_model_v2, CLASS_NAMES, NUM_CLASSES

CLASS_LIST = ['Background', 'Tire', 'Cut', 'Non-Tire']
OUTPUT_DIR = 'outputs/wear_detection_v2'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def evaluate(model, dataloader, device, score_threshold=0.5, iou_threshold=0.5):
    model.eval()
    all_gt_labels = []
    all_pred_labels = []
    all_pred_scores = []
    for images, targets in tqdm(dataloader, desc='Evaluating'):
        images = [img.to(device) for img in images]
        outputs = model(images)
        for i, output in enumerate(outputs):
            gt_boxes = targets[i]['boxes'].cpu().numpy()
            gt_labels = targets[i]['labels'].cpu().numpy()
            pred_boxes = output['boxes'].cpu().numpy()
            pred_scores = output['scores'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            mask = pred_scores >= 0.05
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_labels = pred_labels[mask]
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

def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_LIST, yticklabels=CLASS_LIST)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Wear Detection V2')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'Confusion matrix saved to {save_path}')

@torch.no_grad()
def compute_map(model, dataloader, device):
    model.eval()
    all_gt_boxes = []
    all_gt_labels = []
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    for images, targets in tqdm(dataloader, desc='Computing mAP'):
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

    map50 = float(np.mean(aps_50)) if aps_50 else 0.0
    map5095 = float(np.mean(aps_95)) if aps_95 else 0.0
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

def generate_cut_recall_report(gt_labels, pred_labels, dataset, output_dir):
    gt_img_class = {}
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        labels = target['labels'].cpu().numpy()
        has_cut = 2 in labels
        has_tire = 1 in labels
        has_non_tire = 3 in labels
        if has_cut:
            gt_img_class[idx] = 'bad'
        elif has_tire:
            gt_img_class[idx] = 'good'
        elif has_non_tire:
            gt_img_class[idx] = 'non_tire'
        else:
            gt_img_class[idx] = 'unknown'
    bad_indices = [i for i, c in gt_img_class.items() if c == 'bad']
    good_indices = [i for i, c in gt_img_class.items() if c == 'good']
    non_tire_indices = [i for i, c in gt_img_class.items() if c == 'non_tire']
    bad_with_cut_detected = 0
    bad_cut_missed = 0
    false_cut_good = 0
    false_cut_non_tire = 0
    for idx in bad_indices:
        _, target = dataset[idx]
        detected_cut = False
        for g, p in zip(gt_labels, pred_labels):
            if g == 2 and p == 2:
                detected_cut = True
                break
        if detected_cut:
            bad_with_cut_detected += 1
        else:
            bad_cut_missed += 1
    for idx in good_indices:
        for g, p in zip(gt_labels, pred_labels):
            if p == 2:
                false_cut_good += 1
                break
    for idx in non_tire_indices:
        for g, p in zip(gt_labels, pred_labels):
            if p == 2:
                false_cut_non_tire += 1
                break
    total_bad = len(bad_indices)
    total_good = len(good_indices)
    total_non_tire = len(non_tire_indices)
    cut_tp = sum(1 for g, p in zip(gt_labels, pred_labels) if g == 2 and p == 2)
    cut_fp = sum(1 for g, p in zip(gt_labels, pred_labels) if g != 2 and p == 2)
    cut_fn = sum(1 for g, p in zip(gt_labels, pred_labels) if g == 2 and p != 2)
    cut_precision = cut_tp / (cut_tp + cut_fp) if (cut_tp + cut_fp) > 0 else 0.0
    cut_recall = cut_tp / (cut_tp + cut_fn) if (cut_tp + cut_fn) > 0 else 0.0
    cut_f1 = 2 * cut_precision * cut_recall / (cut_precision + cut_recall) if (cut_precision + cut_recall) > 0 else 0.0

    md_lines = [
        '# Cut Recall Report\n',
        '\n',
        '## Dataset Breakdown\n',
        '\n',
        f'- **Total bad tyres**: {total_bad}\n',
        f'- **Total good tyres**: {total_good}\n',
        f'- **Total non-tyres**: {total_non_tire}\n',
        '\n',
        '## Cut Detection Performance\n',
        '\n',
        f'- **Bad tyres with cut detected**: {bad_with_cut_detected}\n',
        f'- **Bad tyres where cut missed**: {bad_cut_missed}\n',
        f'- **False cut on good tyres**: {false_cut_good}\n',
        f'- **False cut on non-tyres**: {false_cut_non_tire}\n',
        '\n',
        '## Cut Metrics\n',
        '\n',
        f'- **Cut Precision**: {cut_precision:.4f}\n',
        f'- **Cut Recall**: {cut_recall:.4f}\n',
        f'- **Cut F1**: {cut_f1:.4f}\n',
        f'- **Cut TP**: {cut_tp}\n',
        f'- **Cut FP**: {cut_fp}\n',
        f'- **Cut FN**: {cut_fn}\n',
        '\n',
        '## Interpretation\n',
        '\n',
    ]

    if cut_recall >= 0.95:
        md_lines.append('- Cut Recall is excellent (>= 95%). The detector reliably finds damaged tyres.\n')
    elif cut_recall >= 0.85:
        md_lines.append('- Cut Recall is good (85-95%). Few damaged tyres are missed.\n')
    elif cut_recall >= 0.70:
        md_lines.append('- Cut Recall is moderate (70-85%). Some damaged tyres are missed.\n')
    else:
        md_lines.append('- Cut Recall is low (< 70%). Many damaged tyres are missed. Threshold adjustment or retraining may be needed.\n')

    md_lines.append('\n')
    md_lines.append('## Primary Business Metric\n')
    md_lines.append('\n')
    md_lines.append(
        'Cut Recall is the most important metric because missing a damaged tyre '
        '(false negative) is more costly than raising a false alarm (false positive).\n'
    )

    report = ''.join(md_lines)
    report_path = os.path.join(output_dir, 'cut_recall_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'Cut recall report saved to {report_path}')

    return {
        'total_bad': total_bad,
        'bad_with_cut_detected': bad_with_cut_detected,
        'bad_cut_missed': bad_cut_missed,
        'cut_precision': cut_precision,
        'cut_recall': cut_recall,
        'cut_f1': cut_f1,
        'cut_tp': cut_tp,
        'cut_fp': cut_fp,
        'cut_fn': cut_fn,
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'Using device: {DEVICE}')

    coco_path = 'datasets/annotated/annotations_wear_v2.coco.json'
    test_csv = 'datasets/splits/test.csv'
    checkpoint_path = 'outputs/wear_detection_v2/checkpoints/best_frcnn_v2.pt'

    if not os.path.isfile(checkpoint_path):
        print(f'ERROR: Checkpoint {checkpoint_path} not found. Train first.')
        return

    test_dataset = WearDetectionDatasetV2(
        coco_path, test_csv, transforms=None
    )
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=0,
        collate_fn=collate_fn
    )
    print(f'Test samples: {len(test_dataset)}')

    model = get_frcnn_model_v2(num_classes=4, pretrained=False).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get("epoch", "?")})')

    gt_labels, pred_labels, pred_scores = evaluate(
        model, test_loader, DEVICE, score_threshold=0.15, iou_threshold=0.5
    )

    num_classes = NUM_CLASSES
    per_class_metrics = compute_per_class_metrics(gt_labels, pred_labels, num_classes)

    cm = compute_confusion_matrix(gt_labels, pred_labels, num_classes)

    overall_acc = sum(1 for g, p in zip(gt_labels, pred_labels) if g == p) / len(gt_labels) if gt_labels else 0.0

    cut_report = generate_cut_recall_report(gt_labels, pred_labels, test_dataset, OUTPUT_DIR)

    metrics = {
        'OverallAccuracy': round(overall_acc, 4),
        'mAP50': 0.0,
        'mAP50:95': 0.0,
        'PerClassMetrics': per_class_metrics,
        'ConfusionMatrix': cm.tolist(),
    }

    metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Metrics saved to {metrics_path}')

    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(cm, cm_path)

    summary_path = os.path.join(OUTPUT_DIR, 'training_summary.md')

    print('\n--- Attempting mAP computation (may take a while on CPU) ---')
    try:
        map50, map5095 = compute_map(model, test_loader, DEVICE)
        metrics['mAP50'] = round(map50, 4)
        metrics['mAP50:95'] = round(map5095, 4)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'mAP50: {map50:.4f} | mAP50:95: {map5095:.4f}')
        summary_path = os.path.join(OUTPUT_DIR, 'training_summary.md')
    except Exception as e:
        print(f'mAP computation did not finish: {e}')
        print('All other results saved successfully.')
    for cls_name, m in per_class_metrics.items():
        print(f'  {cls_name:12s}  P={m["Precision"]:.4f}  R={m["Recall"]:.4f}  F1={m["F1"]:.4f}')
    print(f'  Cut Recall: {cut_report["cut_recall"]:.4f} (primary metric)')

if __name__ == '__main__':
    main()