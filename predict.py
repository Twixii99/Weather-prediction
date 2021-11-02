import pickle
import xgboost as xgb
from flask import Flask, request, jsonify

model_file = 'model_xgb_model.bin'
with open(model_file, 'rb') as m_in:
    dv, model = pickle.load(m_in)

app = Flask('weather_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    weather_case = request.get_json()

    X = dv.transform([weather_case])
    y_pred = model.predict(xgb.DMatrix(X, feature_names=dv.get_feature_names()))
    rain = y_pred >= 0.5

    result = {
        'Raining probability': float(y_pred),
        'Will it rain ?': bool(rain)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)