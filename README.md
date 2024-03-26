# AI-driven-Personalized-Skincare
Create an AI-powered skincare app that analyzes users' skin and provides personalized product recommendations and routines.
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# Mock data simulating user inputs and corresponding skin types
# Skin types are classified into: 0 - Dry, 1 - Oily, 2 - Combination, 3 - Normal
data = {
    'Age': [25, 30, 22, 40, 35, 28],
    'Skin Sensitivity Level': [1, 3, 2, 5, 4, 2],  # Scale: 1 (Low) - 5 (High)
    'Hours of Sun Exposure Daily': [1, 0.5, 3, 2, 4, 1.5],
    'Skin Type': [0, 3, 1, 2, 0, 1]  # Target variable
}

df = pd.DataFrame(data)

# Splitting the dataset into features (X) and target variable (y)
X = df.drop('Skin Type', axis=1)
y = df['Skin Type']

# Training a simple Random Forest Classifier as our mock AI model
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X, y)

# Function to predict skin type based on user input
def predict_skin_type(age, sensitivity, sun_exposure):
    skin_types = ['Dry', 'Oily', 'Combination', 'Normal']
    prediction = clf.predict([[age, sensitivity, sun_exposure]])
    return skin_types[prediction[0]]

# Function to provide personalized skincare recommendations
def skincare_recommendations(skin_type):
    recommendations = {
        'Dry': 'Use hydrating cleansers and moisturizers. Avoid alcohol-based products.',
        'Oily': 'Use oil-free cleansers and moisturizers. Consider salicylic acid treatments.',
        'Combination': 'Use gentle cleansers and moisturize adequately. Spot treat as necessary.',
        'Normal': 'Maintain with a balanced cleanser and moisturizer. SPF is key for all skin types.'
    }
    return recommendations[skin_type]

# Example of using the app
user_age = 28
user_sensitivity = 2
user_sun_exposure = 1.5

predicted_skin_type = predict_skin_type(user_age, user_sensitivity, user_sun_exposure)
print(f"Predicted Skin Type: {predicted_skin_type}")
print("Personalized Skincare Recommendations:")
print(skincare_recommendations(predicted_skin_type))
