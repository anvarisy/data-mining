import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from kmeans import Kmeans
from preprocess import Preprocess
from fastapi.openapi.utils import get_openapi

app = FastAPI()


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Data Mining Documentation",
        version="1.0.0",
        description="This is for study purpose",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/")
async def home():
    return {"message": "love koala panda"}


class DataModel(BaseModel):
    segment: int
    country: int
    product: int
    discount_band: int
    unit_sold: float
    manufacturing_price: float
    sale_price: float
    gross_sales: float
    discounts: float
    sales: float
    cogs: float
    profit: float
    date: int
    month: int
    year: int

    def to_dict(self):
        return {
            "segment": self.segment,
            "country": self.country,
            "product": self.product,
            "discount_band": self.discount_band,
            "unit_sold": self.unit_sold,
            "manufacturing_price": self.manufacturing_price,
            "sale_price": self.sale_price,
            "gross_sales": self.gross_sales,
            "discounts": self.discounts,
            "sales": self.sales,
            "cogs": self.cogs,
            "profit": self.profit,
            "date": self.date,
            "month": self.month,
            "year": self.year,
        }


class Scoring(BaseModel):
    silhoutte_score: float
    calinski_score: float
    davies_score: float

    def to_dict(self):
        return {
            "silhoutte_score": self.silhoutte_score,
            "calinski_score": self.calinski_score,
            "davies_score": self.davies_score,
        }


model = joblib.load('./models/model.pkl')


@app.post('/predict')
async def predict(data: DataModel):
    print("predicting")
    # data is already an instance of DataModel and is valid
    data_dict = data.to_dict()  # convert the data to a dictionary
    data_df = pd.DataFrame([data_dict])  # convert the dictionary to a DataFrame
    # Make the prediction
    prediction = model.predict(data_df)  # use the DataFrame as input to the predict method

    return {"prediction": prediction.tolist()}


@app.post('/training')
async def training():
    if os.path.exists('./data/Financials.csv'):
        data = pd.read_csv('./data/Financials.csv')

        kmeans = Kmeans()
        kmeans.run(data, 4)
    else:
        preprocess = Preprocess()
        preprocess.run()
    return {"message": "Training completed"}  # kembalikan hasil sebagai response JSON


@app.get('/score', response_model=Scoring)
async def score():
    data = pd.read_csv('./data/Financials.csv')
    predicted_clusters = model.predict(data)

    # Calculate Silhouette score
    sil_score = silhouette_score(data, predicted_clusters)
    # Kalkulasi Calinski-Harabasz score
    ch_score = calinski_harabasz_score(data, predicted_clusters)
    # kalkulasi davis boulder scrore
    db_score = davies_bouldin_score(data, predicted_clusters)
    scoring = Scoring(silhoutte_score=sil_score, calinski_score=ch_score, davies_score=db_score)
    return scoring


if __name__ == '__main__':
    print("Running")
    # app.run(debug=True)
    # data = pd.read_csv('./data/Financials.csv')
    # kmeans = Kmeans()
    # kmeans.run(data, 4)
