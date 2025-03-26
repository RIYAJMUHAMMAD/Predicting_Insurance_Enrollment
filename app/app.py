import pickle, json
from flask import Flask, request, jsonify
import numpy as np
import xgboost as xgb  # Import the xgboost library

# Load the model (assuming it's saved using xgboost's save_model)
try:
    model = xgb.Booster()
    model.load_model('model/best_xgboost_model.bin')
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

# Define the mappings

with open("data/label_transformation.json", "r") as fp:
    mappings = json.load(fp)


# Define the order of features expected by the model
feature_order = ['age', 'gender', 'marital_status', 'salary', 'employment_type', 'region', 'has_dependents', 'tenure_years']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()

        # Encode categorical features and prepare numerical input
        processed_data = []
        for feature in feature_order:
            if feature not in data:
                return jsonify({'error': f'Missing feature in input: {feature}'}), 400

            value = data[feature]
            if feature in mappings:
                if isinstance(value, str):
                    if value in mappings[feature]:
                        processed_data.append(mappings[feature][value])
                    else:
                        return jsonify({'error': f'Invalid value for feature {feature}: {value}'}), 400
                elif isinstance(value, (int, np.integer)):
                    # Assuming if it's an integer, it's already encoded
                    # You might want to add a check if the integer is within the valid encoded range
                    processed_data.append(value)
                else:
                    return jsonify({'error': f'Invalid data type for feature {feature}: {value}'}), 400
            else:
                # For numerical features (age, salary, tenure_years, enrolled), directly append
                if isinstance(value, (int, float)):
                    processed_data.append(value)
                else:
                    return jsonify({'error': f'Invalid data type for feature {feature}: {value}'}), 400

        input_array = np.array([processed_data])

        dmatrix = xgb.DMatrix(input_array, feature_names=feature_order)

        probability = model.predict(dmatrix)[0]  # Assuming single instance prediction

        threshold = 0.5  # You can adjust this threshold if needed
        class_label = 1 if probability > threshold else 0

        # Return the prediction as JSON (now a class label)
        return jsonify({'prediction': int(class_label)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)