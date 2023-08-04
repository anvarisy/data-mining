import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import calendar


class Preprocess:
    def __init__(self):
        pass

    def run(self):
        print("processing data.......")
        # Import Data
        df = pd.read_csv('Financials.csv')
        # Read exist column
        print(df.columns)
        df.columns = df.columns.str.strip()
        # Langkah 2: Membersihkan dan memproses data
        # Menghapus tanda dollar dan mengonversi kolom yang relevan menjadi numerik
        for col in ['Units Sold', 'Manufacturing Price', 'Sale Price','Discounts', 'Gross Sales', 'Sales', 'COGS',
                    'Profit']:
            # for i, val in enumerate(df[col]):
            #     try:
            #         float(val)
            #     except ValueError:
            #         print(f"Error on line {i}: {val}")
            df[col] = df[col].replace('[\$,]', '', regex=True)\
                      .replace(' -   ', np.nan)\
                      .replace('\((.*?)\)', '-\g<1>', regex=True).astype(float)
        df = df.fillna(0)
        #
        # Mengonversi tanggal menjadi format datetime jika diperlukan
        # df['Date'] = pd.to_datetime(df['Date'])
        df = df.drop(columns='Date')
        # konversi string into integer
        encoder_segment = LabelEncoder()
        encoder_country = LabelEncoder()
        encoder_product = LabelEncoder()
        discount_band_encoder = LabelEncoder()

        df['Segment'] = encoder_segment.fit_transform(df['Segment'])
        df['Country'] = encoder_country.fit_transform(df['Country'])
        df['Product'] = encoder_product.fit_transform(df['Product'])
        df['Discount Band'] = discount_band_encoder.fit_transform(df['Discount Band'])
        df['Month Name'] = df['Month Name'].str.strip().apply(lambda x: list(calendar.month_name).index(x))
        df.to_csv('data/Financials.csv', index=False)
        print("preprocess complete")
        return df
