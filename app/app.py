import pickle
import numpy as np
import os
from flask import Flask, render_template, request, flash
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

# Load model, scaler, and columns with relative paths
try:
    model_path = os.path.join(project_dir, 'model', 'best_model_tuned.pkl')
    scaler_path = os.path.join(project_dir, 'model', 'scaler.pkl')
    columns_path = os.path.join(project_dir, 'model', 'model_columns.pkl')
    
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
    model_columns = pickle.load(open(columns_path, 'rb'))
    
    logger.info("All model files loaded successfully")
except Exception as e:
    logger.error(f"Error loading model files: {str(e)}")
    raise

# Create Flask app with proper static folder configuration
app = Flask(__name__, 
           static_folder=os.path.join(project_dir, 'static'),
           template_folder='templates')
app.secret_key = 'your-secret-key-here'  # Change this in production

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form
        input_data = {}

        # Numeric fields with validation
        numeric_fields = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
            'yr_built', 'yr_renovated'
        ]
        
        for field in numeric_fields:
            value = form.get(field, 0)
            try:
                input_data[field] = float(value)
            except ValueError:
                return render_template('index.html', 
                                    prediction_text=f"Error: Invalid value for {field}. Please enter a valid number.")

        # Validate ranges
        if not (0 <= input_data['bedrooms'] <= 20):
            return render_template('index.html', 
                                prediction_text="Error: Bedrooms must be between 0 and 20.")
        
        if not (0 <= input_data['bathrooms'] <= 10):
            return render_template('index.html', 
                                prediction_text="Error: Bathrooms must be between 0 and 10.")
        
        if not (0 <= input_data['view'] <= 4):
            return render_template('index.html', 
                                prediction_text="Error: View must be between 0 and 4.")
        
        if not (1 <= input_data['condition'] <= 5):
            return render_template('index.html', 
                                prediction_text="Error: Condition must be between 1 and 5.")

        # One-hot encoding for city and statezip
        city = form.get('city', '').strip()
        statezip = form.get('statezip', '').strip()
        
        if not city or not statezip:
            return render_template('index.html', 
                                prediction_text="Error: Please select both city and state zip.")
        
        for col in model_columns:
            if col.startswith('city_'):
                input_data[col] = 1.0 if col == f'city_{city}' else 0.0
            elif col.startswith('statezip_'):
                input_data[col] = 1.0 if col == f'statezip_{statezip}' else 0.0

        # Fill missing columns with 0
        for col in model_columns:
            if col not in input_data:
                input_data[col] = 0.0

        # Arrange features and scale
        final_features = np.array([input_data[col] for col in model_columns]).reshape(1, -1)
        final_features_scaled = scaler.transform(final_features)

        # Predict
        prediction = model.predict(final_features_scaled)[0]
        
        # Format prediction nicely
        if prediction < 0:
            prediction = 0  # Ensure non-negative price
        
        formatted_price = f"${prediction:,.2f}"
        return render_template('index.html', prediction_text=f"Estimated Price: {formatted_price}")

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html', 
                            prediction_text=f"Error: An unexpected error occurred. Please try again.")

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html', prediction_text="Error: Page not found."), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html', prediction_text="Error: Internal server error."), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)