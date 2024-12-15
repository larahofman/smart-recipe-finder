import os

# Paths
label_folder = r"D:\JamilyaYOLO2\OpenLabeling\main\output\YOLO_darknet"
class_list_path = r"D:\JamilyaYOLO2\OpenLabeling\main\class_list.txt"

# Load class list
with open(class_list_path, 'r') as f:
    class_list = [line.strip() for line in f.readlines()]

# Map classes to their indices
class_to_index = {class_name: index for index, class_name in enumerate(class_list)}

# Iterate through label files
for label_file in os.listdir(label_folder):
    if label_file.endswith(".txt"):
        label_file_path = os.path.join(label_folder, label_file)

        # Find the matching class name in the file name
        matched_class = None
        for class_name in class_to_index:
            if class_name in label_file.lower():  # Case insensitive match
                matched_class = class_name
                break

        if matched_class is not None:
            # Update the file content
            with open(label_file_path, 'r') as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) > 0:
                    parts[0] = str(class_to_index[matched_class])  # Replace first number
                    updated_lines.append(" ".join(parts) + "\n")

            # Write the updated content back to the file
            with open(label_file_path, 'w') as f:
                f.writelines(updated_lines)

            print(f"Updated file: {label_file} (Class: {matched_class}, Index: {class_to_index[matched_class]})")
