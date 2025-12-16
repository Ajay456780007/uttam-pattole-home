# import pandas as pd
#
#
# def classification_head():
#     data1 = pd.read_csv("../Dataset/Rainfall_Dataset/archive/rainfall in india 1901-2015.csv").iloc[:4100]
#     data2 = pd.read_csv("../Dataset/Temperature_Dataset/archive/train_dataset.csv").iloc[:4100]
#     data3 = pd.read_csv("../Dataset/Wind_Dataset/archive/wind_dataset.csv").iloc[:4100]
#
#     rain_p33 = 907.91
#     rain_p66 = 1393.65
#     rain_extreme = 2314.86
#
#     temp_p33 = 29.10
#     temp_p66 = 31.90
#     temp_extreme = 33.41
#
#     wind_p33 = 7.08
#     wind_p66 = 11.34
#     wind_extreme = 14.78
#
#     def get_rain(pred_rain):
#
#         if (pred_rain > rain_p66) or (pred_rain >= rain_extreme):
#             return 1
#         return 0
#
#     def get_temp(pred_temp):
#
#         if (pred_temp > temp_p66) or (pred_temp >= temp_extreme):
#             return 1
#         return 0
#
#     def get_wind(pred_wind):
#
#         if (pred_wind > wind_p66) or (pred_wind >= wind_extreme):
#             return 1
#         return 0
#
#     rain_series = data1["ANNUAL"]
#     temp_series = data2["Next_Tmax"]
#     wind_series = data3["WIND"]
#
#     combined = pd.DataFrame({
#         "rain": rain_series.values,
#         "temp": temp_series.values,
#         "wind": wind_series.values,
#     })
#
#     def compute_8class_label(row):
#         rain_flag = get_rain(row["rain"])
#         wind_flag = get_wind(row["wind"])
#         temp_flag = get_temp(row["temp"])
#
#         label = rain_flag * 4 + wind_flag * 2 + temp_flag * 1
#         return label
#
#     combined["classification_head"] = combined.apply(compute_8class_label, axis=1)
#
#     combined[["classification_head"]].to_csv(
#         "classification_head.csv",
#         index=False
#     )
#
#
# # classification_head()

import numpy as np
import pandas as pd


def generate_class(y, data):
    rain_p33 = 907.91
    rain_p66 = 1393.65
    rain_extreme = 2314.86

    temp_p33 = 29.10
    temp_p66 = 31.90
    temp_extreme = 33.41

    wind_p33 = 7.08
    wind_p66 = 11.34
    wind_extreme = 14.78

    if data == "rain":
        def get_rain(pred_rain):

            if (pred_rain > rain_p66) or (pred_rain >= rain_extreme):
                return 1
            return 0

        y = np.array(y)
        out = [get_rain(i) for i in y]

        return pd.Series(out, name='rain_head')

    if data == "wind":
        def get_wind(pred_wind):
            if (pred_wind > wind_p66) or (pred_wind >= wind_extreme):
                return 1
            return 0

        y = np.array(y)
        out = [get_wind(i) for i in y]

        return pd.Series(out, name='wind_head')

    if data == "temp":
        def get_temp(pred_temp):
            if (pred_temp > temp_p66) or (pred_temp >= temp_extreme):
                return 1
            return 0

        y = np.array(y)
        out = [get_temp(i) for i in y]
        return pd.Series(out, name='temp_head')
