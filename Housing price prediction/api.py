from flask import Flask, json,request,jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
    if model:
        try:
            # getting json data
            json_ = request.json
            
            # converting to a dataframe
            query = pd.DataFrame(json_)

            # preprocessing data and then predicting values
            prediction = list(model.predict(pipeline.transform(query)))
            
            # return predictions
            return jsonify({'prediction':str(prediction)})
        except:
            return jsonify({'trace':traceback.format_exc()})
    else:
        print('Train the model first')
        return ('no model here to use')


if __name__ == '__main__':
    # loading model
    model = joblib.load("model.pkl")
    print("model loaded")

    # Our custom transformer
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6
    class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
        def __init__(self,add_bedroom_per_room=True):
            self.add_bedroom_per_room = add_bedroom_per_room
        
        def fit(self,X,y=None):
            return self
        
        def transform(self,X,y=None):
            rooms_per_household = X[:,rooms_ix] / X[:,households_ix]
            population_per_household = X[:,population_ix] / X[:,households_ix]
            
            if self.add_bedroom_per_room:
                bedrooms_per_room = X[:,bedrooms_ix] / X[:,rooms_ix]
                return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
            
            else:
                return np.c_[X,rooms_per_household,population_per_household]
    
    # loading pipeline
    pipeline = joblib.load("DataPreprocessing_pipeline.pkl") 
    
    app.run(port=6000,debug=True)