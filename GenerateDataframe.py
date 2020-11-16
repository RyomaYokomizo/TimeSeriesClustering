import pandas as pd
import numpy as np

class GenerateTimeSeriesDF():
    def __init__(self, file_name, date_valiable_name, product_valiable_name, values_valiable_name):
        self.file_name = file_name
        self.date_valiable_name = date_valiable_name
        self.product_valiable_name = product_valiable_name
        self.values_valiable_name = values_valiable_name

    def read_file(self):
        if '.csv' in self.file_name:
            dataframe = pd.read_csv(self.file_name)
        elif '.txt' in self.file_name:
            dataframe = pd.read_table(self.file_name, seq=',')
        return dataframe
    
    def generate_ts_df(self):
        df = self.read_file()
        time_series_df = df.groupby(
            index=self.date_valiable_name,
            columns=self.product_valiable_name,
            vlues=self.values_valiable_name,
            aggfung='sum',
            fill_values=0)
        return time_series_df