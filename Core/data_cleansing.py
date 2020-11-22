import numpy as np
import pandas as pd


class DataProcessor:
    accident_frame = []
    vehicle_frame = []
    main_frame = []

    def __init__(self, accident_ds_path, vehicle_ds_path):
        self.accident_dataset = pd.read_csv(accident_ds_path, low_memory=False, chunksize=30000)
        print(self.accident_dataset.shape())
        self.vehicle_dataset = pd.read_csv(vehicle_ds_path, low_memory=False, chunksize=30000)

    def remove_junk_data(self, dataset_name):
        out_of_range = "Data missing or out of range"
        if dataset_name == "accident_dataset":
            dataset = self.accident_dataset
        else:
            dataset = self.vehicle_dataset

        for chunk_data in dataset:
            if dataset_name == "accident_dataset":
                usefuldata = chunk_data[(chunk_data.Year.astype(int) >= 2012)]
                '''usefuldata = chunk_data[(chunk_data['Junction_Control'] != out_of_range)
                                        & (chunk_data['Carriageway_Hazards'] != out_of_range)
                                        & (chunk_data.Year.astype(int) >= 2012)
                                        & (chunk_data['Road_Type'] != "Unknown")
                                        & (chunk_data['Road_Surface_Conditions'] != out_of_range)
                                        & (chunk_data['Weather_Conditions'] != out_of_range)
                                        & (chunk_data.Year.astype(int) <= 2017)]'''
                self.accident_frame.append(usefuldata)
                inter_frame_accident = pd.concat(self.accident_frame)
                print(inter_frame_accident.head())
                return inter_frame_accident
            elif dataset_name == "vehicle_dataset":
                usefuldata = chunk_data[(chunk_data.Year.astype(int) >= 2012)]
                '''usefuldata = chunk_data[(chunk_data['Vehicle_Type'] != out_of_range)
                                        & (chunk_data['Skidding_and_Overturning'] != out_of_range)
                                        & (chunk_data['Age_Band_of_Driver'] != out_of_range)
                                        & (chunk_data['Was_Vehicle_Left_Hand_Drive'] != out_of_range)
                                        & (chunk_data.Year.astype(int) >= 2012)
                                        & (chunk_data['Hit_Object_in_Carriageway'] != out_of_range)
                                        & (chunk_data['Sex_of_Driver'] != out_of_range)
                                        & (chunk_data.Year.astype(int) <= 2017)
                                        ]'''
                self.vehicle_frame.append(usefuldata)
                inter_frame_vehicle = pd.concat(self.vehicle_frame)
                print(inter_frame_vehicle.head())
                return inter_frame_vehicle

    def merge_frames(self, accident_frame_set, vehicle_frame_set):
        self.main_frame = pd.merge(accident_frame_set, vehicle_frame_set)
        return self.main_frame


data_processor_object = DataProcessor(
    "/Volumes/SJSU/CS271 Topics in Machine Learning/Final project/Dataset/Accident_Information.csv",
    "/Volumes/SJSU/CS271 Topics in Machine Learning/Final project/Dataset/Vehicle_Information.csv")

accident_frame = data_processor_object.remove_junk_data("accident_dataset")
vehicle_frame = data_processor_object.remove_junk_data("vehicle_dataset")
main_frame = data_processor_object.merge_frames(accident_frame, vehicle_frame)
print(main_frame.columns)
print(main_frame.shape)

