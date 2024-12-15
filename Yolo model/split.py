import os
import shutil
import random

# Paths
input_images_folder = r"D:\JamilyaYOLO2\OpenLabeling\main\input"
input_labels_folder = r"D:\JamilyaYOLO2\OpenLabeling\main\output\YOLO_darknet"
output_base_folder = r"D:\JamilyaYOLO2\OpenLabeling\main\dataset"

# Create output directories
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(output_base_folder, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base_folder, split, "labels"), exist_ok=True)

# Parameters for splitting
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Get all image files
image_files = [f for f in os.listdir(input_images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(image_files)  # Shuffle for random split

# Calculate split sizes
total_images = len(image_files)
train_size = int(total_images * train_ratio)
val_size = int(total_images * val_ratio)
test_size = total_images - train_size - val_size

# Split images
train_files = image_files[:train_size]
val_files = image_files[train_size:train_size + val_size]
test_files = image_files[train_size + val_size:]

# Helper function to move files
def move_files(image_list, split_name):
    for image_file in image_list:
        # Image path
        image_src = os.path.join(input_images_folder, image_file)
        image_dest = os.path.join(output_base_folder, split_name, "images", image_file)

        # Label path
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_src = os.path.join(input_labels_folder, label_file)
        label_dest = os.path.join(output_base_folder, split_name, "labels", label_file)

        # Move image and corresponding label (if it exists)
        shutil.copy(image_src, image_dest)
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dest)

# Move files to respective directories
move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print("Dataset split completed!")
print(f"Train: {len(train_files)} images")
print(f"Val: {len(val_files)} images")
print(f"Test: {len(test_files)} images")
