from ultralytics import YOLO
import cv2
import pandas as pd
import ast
from flask import Flask, request, render_template, jsonify, send_file
import numpy as np
import os
import re
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer


stemmer = PorterStemmer()

# Load YOLOv8 model
model = YOLO('best.pt')  

detected_ingredients = []

# Load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df['Parsed_Ingredients'] = df['Ingredients'].apply(lambda x: {item[0]: item[1] for item in ast.literal_eval(x)})
    return df

# Extract detected ingredients
def extract_ingredients(results):
    ingredients = []
    for r in results:
        for box in r.boxes:
            confidence = box.conf[0]  
            if confidence > 0.3:  
                ingredient = model.names[int(box.cls[0])]  # Get the class name
                ingredients.append(ingredient)
    return list(set(ingredients)) 


# Partial matching of detected ingredients with recipe ingredients
def is_partial_match(detected_ingredient, recipe_ingredient):
    detected_words = set(stemmer.stem(word.lower()) for word in detected_ingredient.split())
    recipe_words = set(stemmer.stem(word.lower()) for word in recipe_ingredient.split())
    return bool(detected_words & recipe_words)  # Returns True if any word match

# Recipe matching with additional recommendations
def match_recipes(detected_ingredients, time_limit, selected_category, recipes):
    matched = []
    time_limit = int(time_limit)
    additional_suggestions = []
    print("Detected Ingredients:", detected_ingredients)
    for _, row in recipes.iterrows():
        if row['Category'] == selected_category:
            ingredients = row['Parsed_Ingredients']
            match_count = 0
            for recipe_ingredient in ingredients.keys():
                for detected_ingredient in detected_ingredients:
                    if is_partial_match(detected_ingredient, recipe_ingredient):
                        match_count += 1
                        break  

            # Calculate match ratio
            match_ratio = match_count / len(ingredients)
            if match_ratio >= 0.7 and row['Estimated Total Cooking Time (minutes)'] <= time_limit:  
                matched.append({
                    'Name': row['Name'], 
                    'Ingredients': [{'Ingredient': k, 'Quantity': v} for k, v in ingredients.items()],
                    'Steps': row['Steps'].split('. '),
                    'Servings': row['Servings']
                })


    return matched

# Extract quantity from string
def extract_quantity(quantity_str):
    match = re.search(r"(\d+\.?\d*)", quantity_str)
    return int(match.group(1)) if match else None

# Adjust quantities based on servings
def adjust_quantities(ingredients, original_servings, new_servings):
    adjusted_ingredients = []
    for ingredient in ingredients:
        name = ingredient['Ingredient']
        quantity = ingredient['Quantity']
        
        # Replace 'q.b.' with a default value
        if 'q.b.' in quantity:
            quantity = '100g' 
        
        numeric = extract_quantity(quantity)
        unit = re.sub(r"(\d+\.?\d*)", "", quantity).strip()

        adjusted_quantity = (numeric * int(new_servings)) / int(original_servings) if numeric else 0
        adjusted_ingredients.append({
            'Ingredient': name,
            'Quantity': f"{adjusted_quantity:.2f} {unit}" if numeric else quantity
        })
    return adjusted_ingredients

# Web App with Flask
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
ANNOTATION_FOLDER = "annotations"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATION_FOLDER, exist_ok=True)
detected_ingredients = []

@app.route('/')
def home():
    global detected_ingredients
    detected_ingredients = []  
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    global detected_ingredients
    try:
        if 'image' in request.files:
            file = request.files['image']
            if not file:
                return jsonify({'error': 'No image provided'}), 400

            # Save uploaded image
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            img = cv2.imread(image_path)
            if img is None:
                return jsonify({'error': 'Invalid image format'}), 400

            results = model(img)
            new_ingredients = extract_ingredients(results)
            detected_ingredients.extend(new_ingredients)
            detected_ingredients = list(set(detected_ingredients))

            # Save YOLO annotated image
            annotated_path = os.path.join(ANNOTATION_FOLDER, f"annotated_{file.filename}")
            annotated_img = results[0].plot()
            cv2.imwrite(annotated_path, annotated_img)

            print("Detected ingredients from image:", detected_ingredients)

            return jsonify({'detected': detected_ingredients, 'annotated_image': f'/annotated/{os.path.basename(annotated_path)}'})

        else:
            return jsonify({'error': 'No input provided (image or camera IP).'}), 400

    except Exception as e:
        print("Error in /detect:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/annotated/<filename>')
def serve_annotated_image(filename):
    return send_file(os.path.join(ANNOTATION_FOLDER, filename), mimetype="image/png")


@app.route('/recipes', methods=['POST'])
def recipes():
    global detected_ingredients
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    manual_ingredients = data.get('manual_ingredients', [])
    detected_ingredients.extend(manual_ingredients)
    detected_ingredients = list(set(detected_ingredients))
    time_limit = int(data['time_limit'])
    selected_category = data['category']
    dataset = load_dataset('filtered.csv')

    recipes = match_recipes(detected_ingredients, time_limit, selected_category, dataset)
    
    for recipe in recipes:
        original_servings = recipe.get('Servings')
        recipe['AdjustedIngredients'] = adjust_quantities(recipe['Ingredients'], original_servings, original_servings)  # Initially use original servings
    
    if recipes:
        return jsonify({'recipes': recipes})
    else:
        return jsonify({'message': 'No matching recipes found!'})
    
@app.route('/adjust_quantities', methods=['POST'])
def adjust_quantities_endpoint():
    try:
        data = request.get_json()
        recipe_name = data.get('recipe_name')
        original_servings = data.get('original_servings', 1)
        new_servings = data.get('new_servings', 1)

        # Find the recipe in dataset
        dataset = load_dataset('filtered.csv')
        recipe = dataset[dataset['Name'] == recipe_name]
        if recipe.empty:
            return jsonify({'error': 'Recipe not found'}), 404

        # Adjust quantities
        parsed_ingredients = recipe.iloc[0]['Parsed_Ingredients']
        adjusted_ingredients = adjust_quantities(
            [{'Ingredient': k, 'Quantity': v} for k, v in parsed_ingredients.items()],
            original_servings,
            new_servings
        )

        return jsonify({'AdjustedIngredients': adjusted_ingredients})

    except Exception as e:
        print("Error in /adjust_quantities:", str(e))
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)