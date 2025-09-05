from transformers import pipeline
from PIL import Image, ImageDraw

# Load pipeline
od_pipe = pipeline("object-detection", model="facebook/detr-resnet-50")

# Load image
raw_image = Image.open("img1.jpg")

# Run detection
pipeline_output = od_pipe(raw_image)

# Draw results on image
draw = ImageDraw.Draw(raw_image)
for obj in pipeline_output:
    box = obj["box"]
    label = obj["label"]
    score = obj["score"]

    draw.rectangle([box["xmin"], box["ymin"], box["xmax"], box["ymax"]], outline="red", width=3)
    draw.text((box["xmin"], box["ymin"] - 10), f"{label} {score:.2f}", fill="red")

# Save or show image
raw_image.show()
raw_image.save("detected_img.jpg")
