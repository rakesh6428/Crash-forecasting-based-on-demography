import Core.data_visualisation as visual
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


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

    def visualise_data(self,encoded_dataset):
        plt.figure(figsize=(10, 5))
        ax = sns.countplot(x="Serious_Class",data = encoded_dataset)
        plt.title("Severity of Accidents", fontsize = 20)
        plt.xlabel("Severity",fontsize=15)
        plt.ylabel("Accidents Count", fontsize=15)
        plt.savefig('Views/accident_severity_classes.png')
        plt.show()

    def undersample_dataset(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=27)
        train_data = pd.concat([X_train,y_train],axis=1)
        severe_data = train_data[train_data.Serious_Class == 1]
        less_severe_data = train_data[train_data.Serious_Class == 0]
        undersample_less_severe = resample(less_severe_data,n_samples=len(severe_data), random_state=0)
        undersampled_data=pd.concat([severe_data,undersample_less_severe])
        return undersampled_data
        #print(undersampled_data.Serious_Class.value_counts())

    def fetch_encoded_sampled_Data(self):
        temp_dataset = self.generate_temporary_dataset()
        int_columns,non_int_columns = self.seperate_non_int_columns(temp_dataset)
        encoded_dataset = self.encoding_dataset(int_columns,non_int_columns)

        # Split the data
        X = encoded_dataset.drop(['Serious_Class'],axis=1)
        y = encoded_dataset['Serious_Class']
        #print(X[X.isnull().any(axis=1)])
        # Visualise the amount of Data in each classes ( Severe and Less Severe )
        self.visualise_data(temp_dataset)

        # Visualised data shows the samples are not equally balanced
        # Perform the Under Sampling
        sampled_dataset = self.undersample_dataset(X,y)
        sampled_X = sampled_dataset.drop(['Serious_Class'],axis=1)
        sampled_y = sampled_dataset['Serious_Class']
        return sampled_dataset,sampled_X,sampled_y



# print(encoded_dataset.Serious_Class.value_counts())
# print("Nothing")
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



