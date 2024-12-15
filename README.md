# Image-Based Recipe Recommendation System Using YOLOv8 for Ingredient Recognition

This project provides a **Smart Recipe Finder** application using a YOLO model to detect ingredients from images and suggest matching recipes. The application consists of three key components:

1. **The Web Application**: Suggests recipes based on detected or manually entered ingredients.
2. **YOLO Model Training**: A section for training the YOLO model to recognize ingredients.
3. **Recipe Data Processing**: Processes and organizes the recipe dataset for the web app.

---

## Features
- Detect ingredients from uploaded photos or a camera feed using the YOLO model.
- Manually add ingredients for more accurate results.
- Adjust recipes for the number of servings and automatically update quantities.
- Get recipe suggestions, including additional recommendations if matches are partial.
- Display steps for recipes and photos for better visualization.

---

## Recognizable Ingredients
The application currently recognizes the following ingredients:
    ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 
    'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'lemon', 'lettuce', 
    'onion', 'orange', 'pear', 'peas', 'pineapple', 'potato', 'spinach', 'sweetpotato', 'tomato', 
    'butter', 'eggs', 'flour', 'milk', 'pasta', 'sugar']

---

## Project Structure
The repository is organized into three main folders:

1. **`application`**: Contains the Flask web app that detects ingredients and suggests recipes.
2. **`training`**: Includes the YOLOv8 model training scripts and dataset preparation.
3. **`processing`**: Handles recipe data processing and preparation for matching.

```
project-root/
├── application/
│   ├── app.py   # Flask backend
|   ├── filtered.csv    # Processed recipe dataset
|   ├── uploads/    # Uploaded images
|   ├── annotations/    # YOLO-annotated images          
│   └── templates/
│       └── index.html  # Frontend UI
├── training/
│   └── yolo_training.py   # YOLO model training scripts
├── processing/
│   ├── preprocess.ipynb    # Recipe data processing   
|   └── gz_recipe.csv.csv  #Original recipe dataset               
└── README.md
```

---

### Requirements
Before running the application, ensure you have the following installed:

- Python 3.8+
- Flask
- OpenCV
- Ultralytics (for YOLOv8)
- Pandas
- NumPy
- NLTK (Natural Language Toolkit for text processing)

Install dependencies using:
```bash
pip pip install flask opencv-python-headless ultralytics pandas numpy nltk
```

---

## Recipe Dataset Processing
The **processing** folder contains a Jupyter Notebook (**`preprocess.ipynb`**) used for cleaning, preprocessing, and preparing the recipe dataset. The following tasks were performed:
- Translation of recipes from Italian to English.
- Cleaning and formatting ingredient quantities and steps.
- Handling missing or inconsistent data.

**Note**: Some recipes were added manually after the processing phase to ensure variety and completeness.

---

## How to Run the Application
Follow these steps to run the **Smart Recipe Finder** locally:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your_username/recipe-suggestion-yolo.git
   cd recipe-suggestion-yolo/application
   ```

2. **Install Dependencies**
   Make sure you have Python installed. Install required libraries using:
   ```bash
   pip install flask opencv-python-headless ultralytics pandas numpy
   ```

3. **Run the Application**
   Start the Flask application by running:
   ```bash
   python app.py
   ```

4. **Open the Web Page**
   After starting the app, a link like this will appear:
   ```
   http://127.0.0.1:5000/
   ```
   - Copy and paste this link into your browser.
   - Upload an image of ingredients 
   - View suggested recipes and adjust serving sizes as needed.

5. **Using the Application**:
   - **Upload an Image**: Use the upload form to upload an image containing ingredients.
   - **Ingredient Detection**: View detected ingredients and add missing ones manually.
   - **Find Recipes**: Specify meal type and cooking time to get recipe suggestions.
   - **Adjust Quantities**: Change the number of servings dynamically, and ingredient quantities will update accordingly.

---


### Image Handling
- **Uploads**: Images uploaded by the user are stored in the `uploads/` folder.
- **Annotations**: YOLO-annotated images (showing detection results) are saved in the `annotations/` folder.

---

## License
This project is licensed under the MIT License. Feel free to use and modify it for your purposes.
