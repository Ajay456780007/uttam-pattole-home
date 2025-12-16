import os

import numpy as np
from termcolor import cprint, colored

from Proposed_model.proposed_model2 import Proposed_model2
from Sub_Functions.Load_data import models_return_metrics, Load_data2, Load_data, train_test_splitter, \
    train_test_splitter1, Regression_splitter1
from Proposed_model.proposed_model import Proposed_model


# from proposed_model.proposed_model import proposed_model


class Analysis:
    def __init__(self, Data):
        self.lab = None
        self.feat = None
        self.DB = Data
        self.E = [20, 40, 60, 80, 100]
        self.save = False

    def Data_loading(self):
        self.feat = np.load(f"data_loader/{self.DB}/Features.npy")
        # loading the labels
        self.lab = np.load(f"data_loader/{self.DB}/Labels.npy")

    def COMP_Analysis(self):
        self.Data_loading()
        tr = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        models_return_metrics(self.DB, ok=True, epochs=5)

        perf_names = ["ACC", "SEN", "SPE", "F1score", "REC", "PRE", "TPR", "FPR", "PC1", "RE1"]
        perf_names_2 = ["MAE", "RMSE", "MAE", "R2", "COR"]
        os.makedirs(f"Analysis1/Comparative_Analysis/{self.DB}/", exist_ok=True)
        files_name = [f"Analysis1/Comparative_Analysis/{self.DB}/{name}_1.npy" for name in perf_names]

        A = np.load(f"Temp/{self.DB}/Comp/Model1/all_metrics.npy", allow_pickle=True).tolist()
        B = np.load(f"Temp/{self.DB}/Comp/Model2/all_metrics.npy", allow_pickle=True).tolist()
        C = np.load(f"Temp/{self.DB}/Comp/Model3/all_metrics.npy", allow_pickle=True).tolist()
        D = np.load(f"Temp/{self.DB}/Comp/Model4/all_metrics.npy", allow_pickle=True).tolist()
        E = np.load(f"Temp/{self.DB}/Comp/Model5/all_metrics.npy", allow_pickle=True).tolist()
        F = np.load(f"Temp/{self.DB}/Comp/Model6/all_metrics.npy", allow_pickle=True).tolist()
        G = np.load(f"Temp/{self.DB}/Comp/Model7/all_metrics.npy", allow_pickle=True).tolist()

        all_models = [A, B, C, D, E, F, G]

        if self.save:
            for j in range(len(perf_names)):
                new = []
                for model_metrics in all_models:
                    x = [row[j] for row in model_metrics]
                    new.append(x)
                # if self.save:
                np.save(files_name[j], np.array(new))

    def PERF_Analysis(self):
        epoch = [1, 2, 3, 4, 5]
        Performance_Results = []
        Training_Percentage = 40
        cprint("The performance Analysis starts....", color="blue", on_color="on_white")

        for i in range(6):
            cprint(f"[⚠️] Performance Analysis Count  Is {i + 1} Out Of 6", 'cyan', on_color='on_grey')
            output = []
            for ep in epoch:
                x_train1, x_test1, y_train1, y_test1, x_train2, x_test2, y_train2, y_test2, x_train3, x_test3, y_train3, y_test3, C_train, C_test = Regression_splitter1(
                    self.DB, percent=Training_Percentage / 100)
                result = Proposed_model(x_train1, x_test1, y_train1, y_test1, x_train2, x_test2, y_train2, y_test2,
                                        x_train3, x_test3, y_train3, y_test3, C_train, C_test, ep)
                output.append(result)
            Performance_Results.append(output)

            Training_Percentage += 10

        cprint("The results are saved successfully for Performance Analysis...", color="yellow", on_color="on_white")
        cprint("[✅] Execution of Performance Analysis Completed", 'green', color="yellow", on_color="on_white")
