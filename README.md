# House Price Prediction ML Project

A machine learning project that predicts house prices based on various features using a trained model deployed as a Flask web application.

## Project Structure

```
house-price-prediction-ml-project/
├── app/                    # Flask web application
│   ├── app.py             # Main Flask application
│   └── templates/         # HTML templates
│       └── index.html     # Main prediction form
├── data/                  # Dataset and processed data
│   ├── Housing.csv        # Original dataset
│   ├── Housing_cleaned.csv # Cleaned dataset
│   ├── X_train_scaled.npy # Scaled training features
│   ├── X_test_scaled.npy  # Scaled test features
│   ├── y_train.npy        # Training labels
│   └── y_test.npy         # Test labels
├── model/                 # Trained models and preprocessing
│   ├── best_model.pkl     # Base trained model
│   ├── best_model_tuned.pkl # Hyperparameter tuned model
│   ├── scaler.pkl         # Feature scaler
│   └── model_columns.pkl  # Feature column names
├── notebooks/             # Jupyter notebooks for analysis
│   ├── 01-data-exploration.ipynb
│   ├── 02-preprocessing.ipynb
│   ├── 03-train-test-split.ipynb
│   ├── 04-model-training.ipynb
│   ├── 05-model-evaluation.ipynb
│   └── 06-hyperparameter-tuning.ipynb
├── static/                # Static files
│   └── style.css          # CSS styling
├── requirements.txt        # Python dependencies
└── README.md             # Project documentation
```

## Features

The model predicts house prices based on the following features:
- **Bedrooms**: Number of bedrooms
- **Bathrooms**: Number of bathrooms
- **Sqft Living**: Living area square footage
- **Sqft Lot**: Lot size square footage
- **Floors**: Number of floors
- **Waterfront**: Whether the property is on waterfront (0/1)
- **View**: View rating (0-4)
- **Condition**: Property condition (1-5)
- **Sqft Above**: Above ground square footage
- **Sqft Basement**: Basement square footage
- **Year Built**: Construction year
- **Year Renovated**: Renovation year
- **City**: Property location city
- **State Zip**: Property zip code

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd house-price-prediction-ml-project
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app/app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Fill in the house features in the form and click "Predict" to get the estimated price

## Model Information

- **Algorithm**: Decision Tree Regressor with hyperparameter tuning
- **Performance**: Optimized for accuracy and interpretability
- **Preprocessing**: StandardScaler for feature normalization
- **Feature Engineering**: One-hot encoding for categorical variables

## Development

The project includes comprehensive Jupyter notebooks for:
- Data exploration and analysis
- Data preprocessing and cleaning
- Train-test splitting
- Model training and evaluation
- Hyperparameter tuning

## Technologies Used

- **Python**: Core programming language
- **Flask**: Web framework for deployment
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter**: Interactive development environment

## License

This project is for educational and demonstration purposes.

## Contributing

Feel free to submit issues and enhancement requests!
