import json
import os
import sys
import torch
import torchvision.transforms as T
from PIL import Image
from utils import get_model, load_checkpoint, CLASS_NAMES

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
    return boxes, scores, labels

def classify_output(boxes, scores, labels):
    best_score = 0.0
    best_class = 'Non-Tire'
    for box, score, label in zip(boxes, scores, labels):
        if score > best_score:
            best_score = score
            best_class = CLASS_NAMES.get(int(label), 'Non-Tire')
    return {
        'class': best_class,
        'confidence': round(float(best_score), 2),
    }

def main():
    if len(sys.argv) < 2:
        print('Usage: python src/wear_detection/infer_wear_frcnn.py image.jpg',
              file=sys.stderr)
        sys.exit(1)
    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f'Error: {image_path} not found', file=sys.stderr)
        sys.exit(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'outputs/wear_detection/checkpoints/best_wear_frcnn.pt'
    if not os.path.isfile(checkpoint_path):
        print(f'Error: checkpoint {checkpoint_path} not found. Train first.',
              file=sys.stderr)
        sys.exit(1)
    model = get_model(num_classes=4).to(device)
    load_checkpoint(model, checkpoint_path, device=device)
    boxes, scores, labels = predict(model, image_path, device)
    output = classify_output(boxes, scores, labels)
    print(json.dumps(output, indent=2))

if __name__ == '__main__':
    main()