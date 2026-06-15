import json
import os
import sys

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from train_frcnn import get_model
from utils import load_checkpoint

CLASS_NAMES = {1: 'Tire', 2: 'Non-Tire'}

@torch.no_grad()
def predict(model, image_path, device, score_threshold=0.5):
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
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        detections.append({
            'class': CLASS_NAMES.get(int(label), f'Class_{int(label)}'),
            'confidence': float(score),
            'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
        })
    return detections, image

def draw_boxes(image, detections):
    draw = ImageDraw.Draw(image)
    colors = {'Tire': 'lime', 'Non-Tire': 'red'}
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for det in detections:
        x1, y1, w, h = det['bbox']
        x2, y2 = x1 + w, y1 + h
        color = colors.get(det['class'], 'yellow')
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{det['class']} {det['confidence']:.2f}"
        if font:
            bbox = draw.textbbox((x1, y1 - 12), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1 - 12), label, fill='black', font=font)
        else:
            draw.text((x1, max(0, y1 - 10)), label, fill=color)
    return image

def gatekeeper_output(detections):
    tire_dets = [d for d in detections if d['class'] == 'Tire']
    if not tire_dets:
        return {'is_tire': False}
    best = max(tire_dets, key=lambda d: d['confidence'])
    return {'is_tire': True, 'confidence': best['confidence']}

def main():
    if len(sys.argv) < 2:
        print('Usage: python src/detection/infer_frcnn.py image.jpg', file=sys.stderr)
        sys.exit(1)
    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f'Error: {image_path} not found', file=sys.stderr)
        sys.exit(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'outputs/detection/checkpoints/best_frcnn.pt'
    model = get_model(num_classes=3).to(device)
    load_checkpoint(model, checkpoint_path, device=device)
    print(f'Loaded checkpoint from {checkpoint_path}', file=sys.stderr)
    detections, image = predict(model, image_path, device)
    output = {
        'detections': detections,
        'gatekeeper': gatekeeper_output(detections),
    }
    print(json.dumps(output, indent=2))
    vis_dir = 'outputs/detection/predictions'
    os.makedirs(vis_dir, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(image_path))[0] + '_pred.png'
    vis_path = os.path.join(vis_dir, out_name)
    vis_image = draw_boxes(image.copy(), detections)
    vis_image.save(vis_path)
    print(f'Visualization saved to {vis_path}', file=sys.stderr)

if __name__ == '__main__':
    main()