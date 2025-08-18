# app.py
from flask import Flask, request, jsonify
import pandas as pd

# load pickle model
import joblib
import os

# Use absolute path for the model
model_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "best_xgboost_model.pkl",
)
loaded_model = joblib.load(model_path)

print(f"Model loaded from: {model_path}")

app = Flask(__name__)

# Define the feature order used during training
FEATURES = [
    "edge_noise",
    "text_density",
    "grayscale_variance",
    "alpha_channel_density",
    "unique_font_colors",
    "reported_income",
]


def predict_sklearn(data):
    # Convert input data to DataFrame with correct feature order
    df = pd.DataFrame([data], columns=FEATURES)
    return loaded_model.predict(df)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["data"]
        prediction = predict_sklearn(data)

        return jsonify(
            {
                "prediction": int(prediction[0]),
                "prediction_label": "EDITED" if prediction[0] == 1 else "CLEAN",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)


# trained features
# features = [
#     "edge_noise",
#     "text_density",
#     "grayscale_variance",
#     "alpha_channel_density",
#     "unique_font_colors",
#     "reported_income",
# ]
# {
#     "data": {
#         "edge_noise": 0.5,
#         "text_density": 0.3,
#         "grayscale_variance": 0.2,
#         "alpha_channel_density": 0.4,
#         "unique_font_colors": 5,
#         "reported_income": 60000,
#     }

# }
