from PIL import ImageDraw, ImageFont

COLORS = {
    "Tire": (0, 255, 0),
    "Cut": (255, 0, 0),
    "Non-Tire": (0, 0, 255),
    "Good-Tire": (0, 255, 0),
    "Bad-Tire": (255, 0, 0),
}


def draw_boxes(image, detections):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, w, h = det["bbox"]
        x2, y2 = x1 + w, y1 + h
        cls_name = det["class"]
        color = COLORS.get(cls_name, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{cls_name} {det['confidence']:.2f}"
        label_bbox = draw.textbbox((x1, max(0, y1 - 20)), label, font=font)
        draw.rectangle(label_bbox, fill=color)
        text_color = (0, 0, 0) if cls_name in ("Tire",) else (255, 255, 255)
        draw.text((x1, max(0, y1 - 20)), label, fill=text_color, font=font)

    return image


def result_card_html(final_class, confidence, reason):
    if final_class == "Good-Tire":
        color = "#28a745"
        icon = "✓"
    elif final_class == "Bad-Tire":
        color = "#dc3545"
        icon = "✗"
    else:
        color = "#17a2b8"
        icon = "!"

    return f"""
    <div style="
        background: {color};
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    ">
        <div style="font-size: 48px; font-weight: bold;">{icon}</div>
        <div style="font-size: 28px; font-weight: bold;">{final_class}</div>
        <div style="font-size: 16px; opacity: 0.9;">Confidence: {confidence:.2%}</div>
        <div style="font-size: 14px; opacity: 0.75; margin-top: 5px;">{reason}</div>
    </div>
    """


def metric_card_html(label, value):
    return f"""
    <div style="
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    ">
        <div style="font-size: 12px; color: #6c757d; text-transform: uppercase;">{label}</div>
        <div style="font-size: 22px; font-weight: bold; color: #212529;">{value:.1%}</div>
    </div>
    """
