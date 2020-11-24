import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class DataVisualisation:
    def __init__(self,path):
        self.main_dataset = pd.read_pickle(path)

    def get_dataset(self):
        return self.main_dataset

    def accident_year_view(self,accidents_year):
        plt.figure(figsize=(10, 5))
        sns.barplot(accidents_year.index, accidents_year.values, color="b")
        plt.title("Accidents Per Year", fontsize=20, fontweight="bold")
        plt.xlabel("\nYear", fontsize=15, fontweight="bold")
        plt.ylabel("\nNumber of Accidents", fontsize=15, fontweight="bold")
        plt.savefig('Views/accidents.png')
        plt.show()

    def speed_serious_view(self,severe_class,less_severe_class):
        interval = [20, 30, 40, 50, 60, 70]
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 12))

        ax1 = sns.countplot("Serious_Class", hue="Speed_limit", hue_order=interval,
                            palette="plasma", data=less_severe_class, ax=ax[0])
        ax2 = sns.countplot("Serious_Class", hue="Speed_limit", hue_order=interval,
                            palette="plasma", data=severe_class, ax=ax[1])
        fig.suptitle("Accident-SpeedLimit Correlation", fontsize=15, fontweight="bold")
        ax1.set_xlabel('Less Severe Accidents', fontsize=12)
        ax2.set_xlabel('Severe Accidents', fontsize=12)
        ax1.set_ylabel('Number of Accidents', fontsize=12)
        ax2.set_ylabel('Number of Accidents', fontsize=12)
        plt.savefig('Views/speed.png')
        fig.show()

    def junction_accidents_view(self,sever_class,less_severe_class):
        interval = ['Give way or uncontrolled', 'Auto traffic signal', 'Authorised person',
                    'Stop sign', 'Not at junction or within 20 metres', 'Data missing or out of range']
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 12))

        ax1 = sns.countplot("Serious_Class", hue="Junction_Control", hue_order=interval,
                            palette="plasma", data=less_severe_class, ax=ax[0])
        ax2 = sns.countplot("Serious_Class", hue="Junction_Control", hue_order=interval,
                            palette="plasma", data=severe_class, ax=ax[1])
        fig.suptitle("Accident-Junction_Control Correlation", fontsize=15, fontweight="bold")
        ax1.set_xlabel('Less Severe Accidents', fontsize=12)
        ax2.set_xlabel('Severe Accidents', fontsize=12)
        ax1.set_ylabel('Number of Accidents', fontsize=12)
        ax2.set_ylabel('Number of Accidents', fontsize=12)
        plt.savefig('Views/junction.png')
        fig.show()

    def driver_gender_accidents_view(self, severe_class, less_severe_class):
        interval = ["Male", "Female", "Missing Data"]
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 12))

        ax1 = sns.countplot("Serious_Class", hue="Sex_of_Driver", hue_order=interval,
                            palette="plasma", data=less_severe_class, ax=ax[0])
        ax2 = sns.countplot("Serious_Class", hue="Sex_of_Driver", hue_order=interval,
                            palette="plasma", data=severe_class, ax=ax[1])
        fig.suptitle("Accident-Sex of Driver Correlation", fontsize=15, fontweight="bold")
        ax1.set_xlabel('Less Severe Accidents', fontsize=12)
        ax2.set_xlabel('Severe Accidents', fontsize=12)
        ax1.set_ylabel('Number of Accidents', fontsize=12)
        ax2.set_ylabel('Number of Accidents', fontsize=12)
        plt.savefig('Views/gender.png')
        fig.show()

    def driver_age_accidents_view(self, severe_class, less_severe_class):
        interval = ['0 - 5', '16 - 20', '21 - 25','26 - 35','36 - 45','46 - 55','56 - 65','66 - 75','Data missing or out of range','Over 75']
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 12))

        ax1 = sns.countplot("Serious_Class", hue="Age_Band_of_Driver", hue_order=interval,
                            palette="plasma", data=less_severe_class, ax=ax[0])
        ax2 = sns.countplot("Serious_Class", hue="Age_Band_of_Driver", hue_order=interval,
                            palette="plasma", data=severe_class, ax=ax[1])
        fig.suptitle("Accident- Age_Band_of_Driver Correlation", fontsize=15, fontweight="bold")
        ax1.set_xlabel('Less Severe Accidents', fontsize=12)
        ax2.set_xlabel('Severe Accidents', fontsize=12)
        ax1.set_ylabel('Number of Accidents', fontsize=12)
        ax2.set_ylabel('Number of Accidents', fontsize=12)
        plt.savefig('Views/age.png')
        fig.show()

data_visualisation_object = DataVisualisation("/Volumes/SJSU/CS271 Topics in Machine Learning/Final project/Dataset/main_frame.pkl")
main_dataset = data_visualisation_object.get_dataset()

accidents_year = main_dataset.groupby(['Year'])['Accident_Index'].count()

# Visualise the accidents per year basis
data_visualisation_object.accident_year_view(accidents_year)

# Split the data set into 2 groups - Severe and Less Severe Accidents
severe_class = main_dataset[(main_dataset['Serious_Class'] == "Severe")]
less_severe_class = main_dataset[(main_dataset['Serious_Class'] == "Less Severe")]

# Visualise the speed and severity correlation
data_visualisation_object.speed_serious_view(severe_class,less_severe_class)

# Visualise effect of the Junction control
data_visualisation_object.junction_accidents_view(severe_class,less_severe_class)

# Visualise demographics of the driver
data_visualisation_object.driver_gender_accidents_view(severe_class,less_severe_class)
data_visualisation_object.driver_age_accidents_view(severe_class,less_severe_class)





