from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and dataset
model = joblib.load('model/model.pkl')
data = pd.read_csv('/Users/anishsmac/Desktop/DiamondPrice/diamonds.csv')

# Get unique values for dropdowns
cut_options = data['cut'].unique()
color_options = data['color'].unique()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', cut_options=cut_options, color_options=color_options)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        carat = float(data['carat'])
        cut = data['cut']
        color = data['color']
        depth = float(data.get('depth', 60))
        table = float(data.get('table', 60))
        x = float(data.get('x', 5.0))
        y = float(data.get('y', 5.0))
        z = float(data.get('z', 3.0))

        # Create a DataFrame for the input data
        input_data = pd.DataFrame({
            'carat': [carat],
            'cut': [cut],
            'color': [color],
            'depth': [depth],
            'table': [table],
            'x': [x],
            'y': [y],
            'z': [z]
        })

        # One-hot encode categorical variables
        input_data_encoded = pd.get_dummies(input_data, columns=['cut', 'color'], drop_first=True)

        # Align with the model's features
        missing_cols = set(model.feature_names_in_) - set(input_data_encoded.columns)
        for col in missing_cols:
            input_data_encoded[col] = 0

        input_data_encoded = input_data_encoded[model.feature_names_in_]

        # Make prediction
        prediction = model.predict(input_data_encoded)[0]

        # Return the prediction result as JSON
        return jsonify({'predicted_price': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
