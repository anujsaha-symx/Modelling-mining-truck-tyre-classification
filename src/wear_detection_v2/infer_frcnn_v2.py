import json
import os
import sys
import argparse

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from model import get_frcnn_model_v2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COLORS = {
    'Tire': (0, 255, 0),
    'Cut': (255, 0, 0),
    'Non-Tire': (0, 0, 255),
}
LABEL_MAP = {1: 'Tire', 2: 'Cut', 3: 'Non-Tire'}


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

    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    return image, boxes, scores, labels


def classify_output(boxes, scores, labels, score_threshold=0.15):
    detections = []
    has_cut = False
    has_tire = False
    has_non_tire = False
    cut_scores = []

    for box, score, label in zip(boxes, scores, labels):
        cls_name = LABEL_MAP.get(int(label), 'Unknown')
        detections.append({
            'class': cls_name,
            'confidence': round(float(score), 4),
            'bbox': [round(float(b), 2) for b in box],
        })
        if cls_name == 'Cut' and score >= score_threshold:
            has_cut = True
            cut_scores.append(score)
        elif cls_name == 'Tire' and score >= score_threshold:
            has_tire = True
        elif cls_name == 'Non-Tire' and score >= score_threshold:
            has_non_tire = True

    if has_cut:
        final_class = 'Bad-Tire'
        max_cut_score = max(cut_scores)
        reason = f'Cut detected (max score: {max_cut_score:.4f})'
    elif has_tire:
        final_class = 'Good-Tire'
        reason = 'Tire detected, no cut found'
    elif has_non_tire:
        final_class = 'Non-Tire'
        reason = 'Non-Tire detected'
    else:
        final_class = 'Unknown'
        reason = 'No objects detected above threshold'

    return {
        'final_class': final_class,
        'reason': reason,
        'detections': detections,
        'score_threshold': score_threshold,
    }


def draw_detections(image, output):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('arial.ttf', 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for det in output['detections']:
        x1, y1, x2, y2 = det['bbox']
        cls_name = det['class']
        color = COLORS.get(cls_name, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{cls_name} {det['confidence']:.2f}"
        draw.text((x1, max(0, y1 - 18)), label, fill=color, font=font)

    return image


def main():
    parser = argparse.ArgumentParser(description='Faster R-CNN V2 Wear Detection')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--score-threshold', type=float, default=0.15,
                        help='Detection confidence threshold (default: 0.15)')
    parser.add_argument('--checkpoint', type=str,
                        default='outputs/wear_detection_v2/checkpoints/best_frcnn_v2.pt',
                        help='Model checkpoint path')
    parser.add_argument('--output', type=str, default=None,
                        help='Save visualization to path')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Skip visualization output')

    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f'Error: {args.image_path} not found', file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.checkpoint):
        print(f'Error: checkpoint {args.checkpoint} not found', file=sys.stderr)
        sys.exit(1)

    model = get_frcnn_model_v2(num_classes=4, pretrained=False).to(DEVICE)
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    image, boxes, scores, labels = predict(
        model, args.image_path, DEVICE, score_threshold=args.score_threshold
    )
    output = classify_output(boxes, scores, labels, score_threshold=args.score_threshold)

    print(json.dumps(output, indent=2))

    if not args.no_visualize:
        vis_image = draw_detections(image.copy(), output)
        if args.output:
            vis_image.save(args.output)
            print(f'Visualization saved to {args.output}')
        else:
            out_dir = 'outputs/wear_detection_v2/predictions'
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.basename(args.image_path)
            out_path = os.path.join(out_dir, f'pred_{base}')
            vis_image.save(out_path)
            print(f'Visualization saved to {out_path}')


if __name__ == '__main__':
    main()
