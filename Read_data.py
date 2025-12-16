import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.impute import KNNImputer
from termcolor import cprint
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
# from Sub_Functions.class_head import classification_head
from imblearn.over_sampling import SMOTE
import smogn
from Sub_Functions.class_head import generate_class


def Read_data(DB):
    if DB == "Rainfall_Dataset":
        # Reading the dataset
        data = pd.read_csv("Dataset/Rainfall_Dataset/archive/rainfall in india 1901-2015.csv")[:500]
        # dropping duplicate columns and rows from the data
        data = data.drop_duplicates()
        # dropping the unwanted columns
        data.drop(columns=["Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec"], inplace=True)
        # removing the ANNUAL column as we want to predict it
        X = data.drop(columns=["ANNUAL"])
        # adding the ANNUAL column in the y variable as it need to be predicted
        y = data["ANNUAL"]

        out = generate_class(y, data="rain")

        # X = pd.concat([X, out], axis=1)
        # Creating instances for the label encoder as the KNN imputation accepts numeric columns
        le = LabelEncoder()
        # Applying the encoder to the SUBDIVISION column as it was TEXT
        X["SUBDIVISION"] = le.fit_transform(X["SUBDIVISION"])
        # creating instance for KNN IMPUTATION with neighbors 5
        Imp = KNNImputer(n_neighbors=3)

        cprint(f"[💀] Performing KNN Imputation to Fill the nan values for {DB}...", color="white", on_color="on_cyan")
        # Performing imputation
        After_imputation = Imp.fit_transform(X)
        # Applying KNN imputation in the label values
        y = Imp.fit_transform(np.array(y).reshape(-1, 1)).ravel()

        cprint(f"[✅] Completed KNN Imputation for {DB}...", color="red", on_color="on_white")

        # Creating instance for the standard scalar
        std_scale = StandardScaler()
        # performing normalization using the standard scalar
        std_out = std_scale.fit_transform(After_imputation)
        # Converting the array to numpy array
        std_out_df = pd.DataFrame(std_out, columns=X.columns)
        # converting the labels into series for concating
        y_series = pd.Series(y, name='y')
        # Concating the features and labels for applying the smote for regression
        fina_df = pd.concat([std_out_df,out, y_series], axis=1)
        cprint("[📈] Applying Smote..", color="yellow", on_color="on_magenta")
        # Instance for smote created and the smote is applied
        smote = smogn.smoter(fina_df, 'y', rel_thres=0.5, k=5, samp_method="extreme")
        # get the resampled x and resampled y from the SMOTE
        X_resampled = smote.drop(columns=['y']).values
        y_resampled = smote['y'].values
        # CREATING DIRECTORY TO SAVE THE FEATURES AND THE LABELS
        os.makedirs(f"data_loader/{DB}/", exist_ok=True)
        # saving the features
        np.save(f"data_loader/{DB}/Features.npy", X_resampled)
        np.save(f"data_loader/{DB}/Labels.npy", y_resampled)
        print("=========================================")
        cprint(F"Data saved for {DB}..👍", color="blue", on_color="on_white")
        print("=========================================")

    if DB == "Temperature_Dataset":
        # Reading the dataset
        data = pd.read_csv("Dataset/Temperature_Dataset/archive/train_dataset.csv")[:1000]
        # dropping duplicate columns and rows from the data
        data = data.drop_duplicates()
        # removing the Next_Tmax column as we want to predict it
        X = data.drop(columns=["Next_Tmax"])
        # adding the Next_Tmax column in the y variable as it need to be predicted
        y = data["Next_Tmax"]

        out = generate_class(y, data="temp")
        # X = pd.concat([X, out], axis=1)
        # creating instance for KNN IMPUTATION with neighbors 5
        Imp = KNNImputer(n_neighbors=3)

        cprint(f"[💀] Performing KNN Imputation to Fill the nan values for {DB}...", color="white", on_color="on_cyan")
        # Performing imputation
        After_imputation = Imp.fit_transform(X)
        # Applying KNN imputation in the label values
        y = Imp.fit_transform(np.array(y).reshape(-1, 1)).ravel()

        cprint(f"[✅] Completed KNN Imputation for {DB}...", color="red", on_color="on_white")
        # Creating instance for the standard scalar
        std_scale = StandardScaler()
        # performing normalization using the standard scalar
        std_out = std_scale.fit_transform(After_imputation)
        # Converting the array to numpy array
        std_out_df = pd.DataFrame(std_out, columns=X.columns)
        # converting the labels into series for concating
        y_series = pd.Series(y, name='y')
        # Concating the features and labels for applying the smote for regression
        fina_df = pd.concat([std_out_df,out, y_series], axis=1)
        cprint("[📈] Applying Smote..", color="yellow", on_color="on_magenta")
        # Instance for smote created and the smote is applied
        smote = smogn.smoter(fina_df, 'y', rel_thres=0.5, k=5, samp_method="extreme")
        # get the resampled x and resampled y from the SMOTE
        X_resampled = smote.drop(columns=['y']).values
        y_resampled = smote['y'].values
        # CREATING DIRECTORY TO SAVE THE FEATURES AND THE LABELS
        os.makedirs(f"data_loader/{DB}/", exist_ok=True)
        # saving the features
        np.save(f"data_loader/{DB}/Features.npy", X_resampled)
        np.save(f"data_loader/{DB}/Labels.npy", y_resampled)
        print("=========================================")
        cprint(F"Data saved for {DB}..👍", color="blue", on_color="on_white")
        print("=========================================")

    if DB == "Wind_Dataset":
        # Reading the dataset
        data = pd.read_csv("Dataset/Wind_Dataset/archive/wind_dataset.csv")[:500]
        # dropping duplicate columns and rows from the data
        data = data.drop_duplicates()

        data.drop(columns=["DATE"], inplace=True)
        # removing the WIND column as we want to predict it
        X = data.drop(columns=["WIND"])
        # adding the WIND column in the y variable as it need to be predicted
        y = data["WIND"]

        out = generate_class(y, data="temp")
        X = pd.concat([X, out], axis=1)
        # creating instance for KNN IMPUTATION with neighbors 5
        Imp = KNNImputer(n_neighbors=5)

        cprint(f"[💀] Performing KNN Imputation to Fill the nan values for {DB}...", color="white", on_color="on_cyan")
        # Performing imputation
        After_imputation = Imp.fit_transform(X)
        # Applying KNN imputation in the label values
        y = Imp.fit_transform(np.array(y).reshape(-1, 1)).ravel()

        cprint(f"[✅] Completed KNN Imputation for {DB}...", color="red", on_color="on_white")
        # Creating instance for the standard scalar
        std_scale = StandardScaler()
        # performing normalization using the standard scalar
        std_out = std_scale.fit_transform(After_imputation)
        # Converting the array to numpy array
        std_out_df = pd.DataFrame(std_out, columns=X.columns)
        # converting the labels into series for concating
        y_series = pd.Series(y, name='y')
        # Concatenating the features and labels for applying the smote for regression
        fina_df = pd.concat([std_out_df,out, y_series], axis=1)
        cprint("[📈] Applying Smote..", color="yellow", on_color="on_magenta")
        # Instance for smote created and the smote is applied
        smote = smogn.smoter(fina_df, 'y', rel_thres=0.5, k=5, samp_method="extreme")
        # get the resampled x and resampled y from the SMOTE
        X_resampled = smote.drop(columns=['y']).values
        y_resampled = smote['y'].values
        # CREATING DIRECTORY TO SAVE THE FEATURES AND THE LABELS
        os.makedirs(f"data_loader/{DB}/", exist_ok=True)
        # saving the features
        XY_resampled = np.hstack([X_resampled, y_resampled.reshape(-1, 1)])
        np.save(f"data_loader/{DB}/Features.npy", X_resampled)
        np.save(f"data_loader/{DB}/Labels.npy",y_resampled)
        print("=========================================")
        cprint(F"Data saved for {DB}..👍", color="blue", on_color="on_white")
        print("=========================================")


def build_classification_head(base_dir="data_loader"):
    rain_feat_path = os.path.join(base_dir, "Rainfall_Dataset", "Features.npy")
    temp_feat_path = os.path.join(base_dir, "Temperature_Dataset", "Features.npy")
    wind_feat_path = os.path.join(base_dir, "Wind_Dataset", "Features.npy")

    # Load
    rain_feat = np.load(rain_feat_path)[:500]  # shape: (N, d_r+1)
    temp_feat = np.load(temp_feat_path)[:500]  # shape: (N, d_t+1)
    wind_feat = np.load(wind_feat_path)[:500]  # shape: (N, d_w+1)

    # Extract heads (last column) and features (all but last column)
    rain_head = rain_feat[:, -1]
    temp_head = temp_feat[:, -1]
    wind_head = wind_feat[:, -1]

    rain_feat_only = rain_feat[:, :-1]
    temp_feat_only = temp_feat[:, :-1]
    wind_feat_only = wind_feat[:, :-1]

    if not (len(rain_head) == len(temp_head) == len(wind_head)):
        raise ValueError("Rain, Temp, Wind feature files have different sample counts.")

    class_head = (rain_head.astype(int) * 4 +
                  wind_head.astype(int) * 2 +
                  temp_head.astype(int) * 1)

    cls_dir = os.path.join(base_dir, "Classification_head")
    os.makedirs(cls_dir, exist_ok=True)
    cls_path = os.path.join(cls_dir, "class_head.npy")
    np.save(cls_path, class_head)

    # Overwrite Features.npy without last column
    np.save(rain_feat_path, rain_feat_only)
    np.save(temp_feat_path, temp_feat_only)
    np.save(wind_feat_path, wind_feat_only)


