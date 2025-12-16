from sklearn.metrics import mean_squared_error, multilabel_confusion_matrix, accuracy_score, precision_score, \
    recall_score, f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import numpy as np

import math
import warnings

# from tensorflow.python.ops.metrics_impl import recall

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def calc_precision_recall(y_true, y_pred):
    import pandas as pd

    # Convert predictions to Series with index matching y_true
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred, index=y_true.index)

    TP = FP = FN = 0

    for i in y_true.index:
        if y_true[i] == y_pred[i] == 1:
            TP += 1
        elif y_pred[i] == 1 and y_true[i] != y_pred[i]:
            FP += 1
        elif y_pred[i] == 0 and y_true[i] != y_pred[i]:
            FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 1
    recall = TP / (TP + FN) if (TP + FN) > 0 else 1

    return precision, recall

def tt(value):
    power = math.ceil(math.log10(value) - 1)
    A1 = 100 ** (math.log10(value) - power)
    return A1


def main_est_parameters(y_true, pred):
    """
    :param y_true: true labels
    :param pred: predicted labels
    :return: performance metrics in list dtype
    """
    precision, recall = calc_precision_recall(y_true, pred)
    cm = multilabel_confusion_matrix(y_true, pred)
    cm = sum(cm)
    TP = cm[0, 0]  # True Positive
    FP = cm[0, 1]  # False Positive
    FN = cm[1, 0]  # False Negative
    TN = cm[1, 1]  # True Negative
    Acc = (TP + TN) / (TP + TN + FP + FN)
    Sen = TP / (TP + FN)
    Spe = TN / (TN + FP)
    Pre = TP / (TP + FP)
    Rec = TP / (TP + FN)
    F1score = 2 * (Pre * Rec) / (Pre + Rec)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return [Acc, Sen, Spe, F1score, Rec, Pre, TPR, FPR,precision, recall]


def main_est_parameters_mul(y_true, pred):
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, average='macro', zero_division=0)
    rec = recall_score(y_true, pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    with np.errstate(divide='ignore', invalid='ignore'):
        specificity = TN / (TN + FP)
        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)

    spe = np.nanmean(specificity)
    fpr_macro = np.nanmean(fpr)
    tpr_macro = np.nanmean(tpr)

    return [acc, tpr_macro, spe, f1, rec, prec, tpr_macro, fpr_macro]


def Evaluation_Metrics1(y, y_pred):
    mse = tt(mean_squared_error(y, y_pred))
    rmse = np.sqrt(mse)
    mae = tt(mean_absolute_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    cor = np.sqrt(r2)
    return [mse, rmse, mae, r2, cor]


def Evaluation_Metrics(y, y_pred):
    y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred_clean = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    mse = mean_squared_error(y_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_clean, y_pred_clean)
    r2 = r2_score(y_clean, y_pred_clean)
    cor = np.sqrt(max(0, r2))

    return [mse, rmse, mae, r2, cor]

import numpy as np
from sklearn.metrics import confusion_matrix


def compute_metrics(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)

    num_classes = cm.shape[0]
    sensitivity_list = []
    specificity_list = []
    F1_score_list = []
    Pre_list = []
    Rec_list = []
    FPR_list = []

    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        Pre = TP / (TP + FP) if (TP + FP) > 0 else 0
        Rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1_score = 2 * (Pre * Rec) / (Pre + Rec) if (Pre + Rec) > 0 else 0
        FPR = FP / FP + TN if (FP + TN) > 0 else 0

        F1_score_list.append(F1_score)
        Pre_list.append(Pre)
        Rec_list.append(Rec)
        sensitivity_list.append(sensitivity)
        FPR_list.append(FPR)
        specificity_list.append(specificity)

    # Accuracy: total correct predictions / total predictions
    accuracy = np.trace(cm) / np.sum(cm)

    avg_precision = np.mean(Pre_list)
    avg_recall = np.mean(Rec_list)
    avg_f1_score = np.mean(F1_score_list)
    avg_sensitivity = np.mean(sensitivity_list)
    avg_specificity = np.mean(specificity_list)
    TPR = avg_recall
    avg_FPR = np.mean(FPR_list)
    return [accuracy, avg_sensitivity, avg_specificity, avg_f1_score, avg_recall, avg_precision, TPR, avg_FPR]


def compute_metrics2(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)

    num_classes = cm.shape[0]
    sensitivity_list = []
    specificity_list = []
    F1_score_list = []
    Pre_list = []
    Rec_list = []
    FPR_list = []

    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        Pre = TP / (TP + FP) if (TP + FP) > 0 else 0
        Rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1_score = 2 * (Pre * Rec) / (Pre + Rec) if (Pre + Rec) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

        F1_score_list.append(F1_score)
        Pre_list.append(Pre)
        Rec_list.append(Rec)
        sensitivity_list.append(sensitivity)
        FPR_list.append(FPR)
        specificity_list.append(specificity)

    # Accuracy: total correct predictions / total predictions
    accuracy = np.trace(cm) / np.sum(cm)
    accuracy = accuracy

    avg_precision = np.mean(Pre_list)
    avg_precision = avg_precision
    avg_recall = np.mean(Rec_list)
    avg_recall = avg_recall
    avg_f1_score = np.mean(F1_score_list)
    avg_f1_score = avg_f1_score
    avg_sensitivity = np.mean(sensitivity_list)
    avg_sensitivity = avg_sensitivity
    avg_specificity = np.mean(specificity_list)
    avg_specificity = avg_specificity
    TPR = avg_recall
    avg_FPR = np.mean(FPR_list)

    return [accuracy, avg_sensitivity, avg_specificity, avg_f1_score, avg_recall, avg_precision, TPR, avg_FPR]


import customtkinter as ctk
import tkinter as tk

# Configure appearance and theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# Main app window
app = ctk.CTk()
# Global variable to store result
result_var = tk.StringVar()


# Callback for YES
def on_yes():
    result_var.set('Yes')  # Set result to 'y'
    popup.destroy()


# Callback for NO
def on_no():
    result_var.set('No')  # Set result to 'n'
    popup.destroy()


# Function to open popup and wait for response
def open_popup(text, title="Confirm Action"):
    global popup
    popup = ctk.CTkToplevel(app)

    # Define popup size
    popup_width = 300
    popup_height = 150

    # Get screen width and height
    screen_width = popup.winfo_screenwidth()
    screen_height = popup.winfo_screenheight()

    # Calculate position x, y to center the popup
    x = int((screen_width / 2) - (popup_width / 2))
    y = int((screen_height / 2) - (popup_height / 2))

    popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")
    popup.title(title)

    label = ctk.CTkLabel(popup, text=text, font=("Arial", 16))
    label.pack(pady=20)

    # Frame to hold Yes/No buttons
    button_frame = ctk.CTkFrame(popup, fg_color="transparent")
    button_frame.pack(pady=10)

    yes_button = ctk.CTkButton(button_frame, text="Yes", command=on_yes, width=100)
    yes_button.pack(side="left", padx=10)

    no_button = ctk.CTkButton(button_frame, text="No", command=on_no, width=100)
    no_button.pack(side="left", padx=10)

    # Wait for the popup to close before continuing
    popup.grab_set()
    app.wait_window(popup)

    # After popup closes, get result
    response = result_var.get()
    # print(f"User selected: {response}")  # 'y' or 'n'
    return response


