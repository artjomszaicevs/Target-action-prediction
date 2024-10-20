import dill

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


with open('model/model.pkl', 'rb') as file:
    model = dill.load(file)
    # df_csv = pd.read_csv('model/data/df_ready.csv').drop('price_category', axis=1)
    #
    # types = {
    #     'int64': 'int',
    #     'float64': 'float'
    # }
    # for k, v in df_csv.dtypes.items():
    #     print(f'{k}: {types.get(str(v), "str")}')


    class Form(BaseModel):
        utm_source: str
        utm_medium: str
        utm_campaign: str
        utm_adcontent: str
        utm_keyword: str
        device_category: str
        device_os: str
        device_brand: str
        device_model: str
        device_screen_resolution: str
        device_browser: str
        geo_country: str
        geo_city: str


    class Prediction(BaseModel):
        prediction: object


    @app.get('/status')
    def status():
        return "I'm OK"


    @app.get('/version')
    def version():
        return model['metadata']


    @app.post('/predict', response_model=Prediction)
    def predict(form: Form):
        df = pd.DataFrame.from_dict([form.dict()])
        y = model['model'].predict(df)

        return {
            'prediction': str(y[0])
        }