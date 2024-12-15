import cv2
from ultralytics import YOLO
import os

# Set paths
input_folder = "D:\JamilyaYOLO2\OpenLabeling\main\input"  # Folder containing input images
output_folder = "output"  # Folder to save YOLO annotation files
os.makedirs(output_folder, exist_ok=True)

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8s.pt")  # Replace with the appropriate model variant

# Perform inference and generate labels
for image_name in os.listdir(input_folder):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(image_name)
        image_path = os.path.join(input_folder, image_name)

        # Check if image can be loaded
        image = cv2.imread(image_path)
        if image is None:
            print(f"WARNING ⚠️ Image Read Error: {image_path}")
            continue

        # Perform inference
        try:
            results = model(image_path)

            # Retrieve detection results
            detections = results[0].boxes.data  # Detected boxes (xyxy, confidence, class)

            # Generate YOLO format annotations
            image_h, image_w = image.shape[:2]
            annotations = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det.tolist()
                x_center = ((x1 + x2) / 2) / image_w
                y_center = ((y1 + y2) / 2) / image_h
                width = (x2 - x1) / image_w
                height = (y2 - y1) / image_h
                annotations.append(f"{int(cls)} {x_center} {y_center} {width} {height}\n")

                # Draw bounding boxes on the image
                label = f"Class {int(cls)}: {conf:.2f}"
                color = (0, 255, 0)  # Green for bounding boxes
                thickness = 2
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Save annotations to a .txt file
            label_file = os.path.join(output_folder, os.path.splitext(image_name)[0] + ".txt")
            with open(label_file, 'w') as f:
                f.writelines(annotations)

            print(f"Processed {image_name} -> {label_file}")

            # Display the image with bounding boxes
            cv2.imshow("Detections", image)
            if cv2.waitKey(0) & 0xFF == ord('q'):  # Press 'q' to quit early
                break

        except Exception as e:
            print(f"ERROR ⚠️ Failed to process {image_path}: {e}")

# Release OpenCV windows
cv2.destroyAllWindows()
print("Auto-labeling completed!")
