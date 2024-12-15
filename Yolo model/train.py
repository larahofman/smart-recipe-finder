from ultralytics import YOLO

# Load the YOLOv8 model
def main():
    model = YOLO("yolov8s.pt")  # Use a pre-trained YOLOv8n model. Replace with "yolov8s.pt", "yolov8m.pt", etc., as needed.

    # Train the model
    model.train(
        data="D:\JamilyaYOLO2\OpenLabeling\main\Jamilya.yaml",  # Path to your data.yaml file
        epochs=1000,                # Number of training epochs
        batch=2,                 # Batch size
        imgsz=640,                # Image size
        project="runs/train",     # Project directory for saving results
        name="yolov8_training",   # Name of the training run
        device=0                  # GPU ID (use "cpu" for CPU training)
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()