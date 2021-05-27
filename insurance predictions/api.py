from flask import Flask,request,jsonify
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/predict',methods=["POST"])
def predict_values():
    json_ = request.json
    data = pd.DataFrame(json_)
    trf_data = data_transformer.transform(data)
    predictions = model.predict(trf_data)

    return jsonify({"prediction":str(predictions)})

if __name__=='__main__':

    # loading model
    model = joblib.load('model.pkl')

    # loading data transformer
    data_transformer = joblib.load('data_transformer.pkl')

    app.run(port=6000,debug=True)