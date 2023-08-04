import os
import pandas as pd
from fastapi import FastAPI

from kmeans import Kmeans
from preprocess import Preprocess
import joblib
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def getHome():
    return {"message": "koala panda"}


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


if __name__ == '__main__':
    print("Running")
    # app.run(debug=True)
    # data = pd.read_csv('./data/Financials.csv')
    # kmeans = Kmeans()
    # kmeans.run(data, 4)
