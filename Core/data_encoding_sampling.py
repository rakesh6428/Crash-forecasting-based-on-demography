import Core.data_visualisation as visual
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataEncoding:

    def __init__(self, path):
        self.main_dataset = pd.read_pickle(path)

    def get_dataset(self):
        return self.main_dataset

    def generate_temporary_dataset(self):
        temp_dataset = self.main_dataset
        temp_dataset.set_index('Accident_Index', inplace=True)
        return temp_dataset
        # print(temp_dataset.head())

    def seperate_non_int_columns(self,dataset):
        int_columns = dataset.select_dtypes(include=['float','int','int64'])
        non_int_columns = dataset.select_dtypes(exclude=['float','int','int64'])
        return int_columns,non_int_columns

    def encoding_dataset(self,int_columns,non_int_columns):
        non_int_encoded_data = non_int_columns.apply(LabelEncoder().fit_transform)
        final_encoded_dataset = pd.concat([non_int_encoded_data,int_columns],axis=1,sort=False)
        return final_encoded_dataset


data_encoding_object = DataEncoding(
    "/Volumes/SJSU/CS271 Topics in Machine Learning/Final project/Dataset/main_frame.pkl")
temp_dataset = data_encoding_object.generate_temporary_dataset()
int_columns,non_int_columns = data_encoding_object.seperate_non_int_columns(temp_dataset)
encoded_dataset = data_encoding_object.encoding_dataset(int_columns,non_int_columns)

print("Nothing")




#temp_dataset = data_encoding_object.main_dataset.select_dtypes(include = ['object']).copy()
#print(temp_dataset[temp_dataset.isnull().any(axis=1)])
#main_dataset = data_encoding_object.remove_nan()
#
# def remove_nan(self):
#     self.main_dataset['Driver_Home_Area_Type'] = self.main_dataset['Driver_Home_Area_Type'].replace(
#         ['Data missing or out of range'], 'Urban area')
#     self.main_dataset = self.main_dataset.fillna({"model": "UNKNOWN"})
#     self.main_dataset = self.main_dataset.fillna({"Propulsion_Code": "Petrol"})
#     self.main_dataset = self.main_dataset.fillna({"2nd_Road_Class": "A"})
#     self.main_dataset = self.main_dataset.fillna({"LSOA_of_Accident_Location": "UNKNOWN"})
#     self.main_dataset = self.main_dataset.fillna({"Time": "17:00"})
#     self.main_dataset = self.main_dataset.fillna({"make": "Ford"})
#     self.main_dataset = self.main_dataset.fillna({"InScotland": "No"})
#     return self.main_dataset



