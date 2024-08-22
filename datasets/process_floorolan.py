import os
from PIL import Image
import numpy as np

# floorplan_fuse_map
floorplan_fuse_map = {
    1: [0, 0, 0],            # background
    2: [192, 192, 224],      # closet
    3: [192, 255, 255],      # batchroom/washroom
    4: [224, 255, 192],      # livingroom/kitchen/dining room
    5: [255, 224, 128],      # bedroom
    6: [255, 160, 96],       # hall
    7: [255, 224, 224],      # balcony
    8: [255, 60, 128],       # extra label for opening (door&window)
    9: [255, 255, 255]       # extra label for wall line
}

# Invert the floorplan_fuse_map to create a mapping from color to label
color_to_label = {tuple(v): k for k, v in floorplan_fuse_map.items()}

# Path to the directory containing images
image_dir = "/data1/JM/code/Mask2Former/datasets/FloorPlan/annotations/validation_original"
output_dir = "/data1/JM/code/Mask2Former/datasets/FloorPlan/annotations/validation"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image in the directory
for img_file in os.listdir(image_dir):
    if img_file.endswith(".png"):  # Assuming images are in .png format
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        # Create an empty array to store the labels, initialized to 0 (background)
        label_img = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)

        # Map each pixel to its corresponding label
        for color, label in color_to_label.items():
            mask = np.all(img_np == color, axis=-1)
            label_img[mask] = label

        # Mark pixels that haven't been assigned a label as 0 (background)
        unmatched_pixels = np.sum(label_img == 0, axis=-1) == 3  # Find pixels that are still 0
        label_img[unmatched_pixels] = 0

        # Save the label image
        output_img = Image.fromarray(label_img)
        # output_img.save(os.path.join(output_dir, img_file))

        print(f"Processed {img_file}")

print("All images processed successfully.")
