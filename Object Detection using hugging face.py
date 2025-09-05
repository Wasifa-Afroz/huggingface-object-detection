from transformers import pipeline
from PIL import Image, ImageDraw
import os

# Load pipeline
od_pipe = pipeline("object-detection", model="facebook/detr-resnet-50")

# List of images you want to process
image_files = ["img1.jpg", "img4.jpg", "img3.jpg"]

# Create an output folder if it doesn’t exist
os.makedirs("outputs", exist_ok=True)

for img_path in image_files:
    # Load image
    raw_image = Image.open(img_path)

    # Run detection
    pipeline_output = od_pipe(raw_image)

    # Draw results
    draw = ImageDraw.Draw(raw_image)
    for obj in pipeline_output:
        box = obj["box"]
        label = obj["label"]
        score = obj["score"]

        draw.rectangle([box["xmin"], box["ymin"], box["xmax"], box["ymax"]], outline="red", width=3)
        draw.text((box["xmin"], box["ymin"] - 10), f"{label} {score:.2f}", fill="red")

    # Save with new name in outputs folder
    output_path = os.path.join("outputs", f"detected_{os.path.basename(img_path)}")
    raw_image.save(output_path)
    print(f"✅ Saved: {output_path}")
