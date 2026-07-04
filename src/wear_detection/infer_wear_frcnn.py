import argparse
import json
import os
import sys

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from utils import get_model, load_checkpoint, CLASS_NAMES, fix_path

DISPLAY_NAMES = {'Good-Tire': 'Tire', 'Bad-Tire': 'Cut', 'Non-Tire': 'Non-Tire'}
PRIORITY = {'Bad-Tire': 0, 'Good-Tire': 1, 'Non-Tire': 2}

GROUND_TRUTH_MAP = {'good': 'Good-Tire', 'bad': 'Bad-Tire', 'negative': 'Non-Tire'}


def decode_predictions(scores, labels, score_threshold):
    detections = []
    for score, label in zip(scores, labels):
        if score < score_threshold:
            continue
        internal_name = CLASS_NAMES.get(int(label), 'Non-Tire')
        display_name = DISPLAY_NAMES.get(internal_name, internal_name)
        detections.append({
            'label': display_name,
            'internal_label': internal_name,
            'confidence': round(float(score), 2),
        })
    return detections


def classify_output(boxes, scores, labels, score_threshold=0.15):
    detections = decode_predictions(scores, labels, score_threshold)

    has_cut = any(d['internal_label'] == 'Bad-Tire' for d in detections)
    has_tire = any(d['internal_label'] == 'Good-Tire' for d in detections)
    has_nontire = any(d['internal_label'] == 'Non-Tire' for d in detections)

    if has_cut:
        cut_dets = [d for d in detections if d['internal_label'] == 'Bad-Tire']
        result = {
            'final_class': 'Bad-Tire',
            'reason': 'Cut detected',
            'detections': cut_dets,
        }
        if has_tire:
            result['reason'] = 'Cut detected on tyre'
            tire_dets = [d for d in detections if d['internal_label'] == 'Good-Tire']
            result['detections'] = cut_dets + tire_dets
    elif has_tire:
        tire_dets = [d for d in detections if d['internal_label'] == 'Good-Tire']
        result = {
            'final_class': 'Good-Tire',
            'reason': 'Tyre detected, no cut detected',
            'detections': tire_dets,
        }
    elif has_nontire:
        nontire_dets = [d for d in detections if d['internal_label'] == 'Non-Tire']
        result = {
            'final_class': 'Non-Tire',
            'reason': 'No tyre detected',
            'detections': [{
                'label': 'Non-Tire',
                'confidence': round(nontire_dets[0]['confidence'], 2),
            }],
        }
    else:
        result = {
            'final_class': 'Non-Tire',
            'reason': 'No tyre detected',
            'detections': [{'label': 'Non-Tire', 'confidence': 1.00}],
        }

    return result


@torch.no_grad()
def predict(model, image_path, device, score_threshold=0.15):
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).to(device)

    model.eval()
    output = model([image_tensor])[0]

    boxes = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    labels = output['labels'].cpu().numpy()

    return boxes, scores, labels


@torch.no_grad()
def predict_image(model, image_path, device, score_threshold=0.15):
    boxes, scores, labels = predict(model, image_path, device, score_threshold)
    return classify_output(boxes, scores, labels, score_threshold)


def get_predicted_class(result):
    return result['final_class']


def get_confidence(result):
    if result['detections']:
        return max(d['confidence'] for d in result['detections'])
    return 0.0


def batch_inference(test_csv, checkpoint_path, output_dir, score_threshold=0.15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = get_model(num_classes=4).to(device)
    load_checkpoint(model, checkpoint_path, device=device)
    print(f'Loaded checkpoint from {checkpoint_path}')

    df = pd.read_csv(test_csv)
    results = []

    for idx, row in df.iterrows():
        filepath = fix_path(row['filepath'])
        gt_label_raw = row.get('class_name', '')
        ground_truth = GROUND_TRUTH_MAP.get(gt_label_raw, gt_label_raw)

        if not os.path.isfile(filepath):
            print(f'WARNING: {filepath} not found, skipping')
            continue

        result = predict_image(model, filepath, device, score_threshold)
        predicted_class = get_predicted_class(result)
        confidence = get_confidence(result)

        results.append({
            'filepath': filepath,
            'ground_truth': ground_truth,
            'predicted_class': predicted_class,
            'final_class': result['final_class'],
            'confidence': confidence,
            'reason': result['reason'],
        })

        if (idx + 1) % 50 == 0:
            print(f'Progress: {idx + 1}/{len(df)}')

    results_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'test_inference_results.csv')
    results_df.to_csv(out_path, index=False)
    print(f'Results saved to {out_path}')
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Infer wear detection on an image')
    parser.add_argument('image_path', nargs='?', default=None,
                        help='Path to input image')
    parser.add_argument('--score-threshold', type=float, default=0.15,
                        help='Detection confidence threshold (default: 0.15)')
    parser.add_argument('--batch', action='store_true',
                        help='Run batch inference on test set')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'outputs/wear_detection/checkpoints/best_wear_frcnn.pt'

    if not os.path.isfile(checkpoint_path):
        print(f'Error: checkpoint {checkpoint_path} not found. Train first.',
              file=sys.stderr)
        sys.exit(1)

    if args.batch:
        test_csv = 'datasets/splits/test.csv'
        if not os.path.isfile(test_csv):
            print(f'Error: {test_csv} not found.', file=sys.stderr)
            sys.exit(1)
        batch_inference(test_csv, checkpoint_path, 'outputs/wear_detection',
                        score_threshold=args.score_threshold)
        return

    if not args.image_path:
        parser.print_help()
        sys.exit(1)

    image_path = args.image_path
    if not os.path.isfile(image_path):
        print(f'Error: {image_path} not found', file=sys.stderr)
        sys.exit(1)

    model = get_model(num_classes=4).to(device)
    load_checkpoint(model, checkpoint_path, device=device)

    result = predict_image(model, image_path, device,
                           score_threshold=args.score_threshold)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
