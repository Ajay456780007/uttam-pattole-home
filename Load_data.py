import numpy as np
import pandas as pd
from termcolor import cprint

from Comparative_models.Model1 import CNN_1


from Comparative_models.Model2 import CNN_BiLSTM_1
from Comparative_models.Model3 import RNN_1
from Comparative_models.Model4 import CNN_LSTM_1
from Comparative_models.Model5 import BiLSTM_1
from Comparative_models.Model6 import LSTM_1
from Comparative_models.Model7 import ANN_1


def Load_data(DB):
    Feat1 = np.load(f"data_loader/Rainfall_Dataset/Features.npy")
    Lab1 = np.load(f"data_loader/Rainfall_Dataset/Labels.npy")

    Feat2 = np.load("data_loader/Temperature_Dataset/Features.npy")
    Lab2 = np.load("data_loader/Temperature_Dataset/Labels.npy")

    Feat3 = np.load("data_loader/Wind_Dataset/Features.npy")
    Lab3 = np.load("data_loader/Wind_Dataset/Labels.npy")

    classification_head = np.load("data_loader/Classification_head/class_head.npy")

    return Feat1, Feat2, Feat3, Lab1, Lab2, Lab3, classification_head


def Regression_splitter1(data, percent, num=10):
    feat1, feat2, feat3, label1, label2, label3, CH = Load_data(data)

    # data_index = np.random.uniform(0, len(np.array(label)))

    training_sequence1 = feat1[:10]
    training_labels1 = label1[:10]
    testing_sequence1 = feat1[10:20]
    testing_labels1 = label1[10:20]

    training_sequence2 = feat2[:10]
    training_labels2 = label2[:10]
    testing_sequence2 = feat2[10:20]
    testing_labels2 = label2[10:20]

    training_sequence3 = feat3[:10]
    training_labels3 = label3[:10]
    testing_sequence3 = feat3[10:20]
    testing_labels3 = label3[10:20]

    C_train = CH[:10]
    C_test = CH[10:20]

    return [training_sequence1, testing_sequence1, training_labels1, testing_labels1, training_sequence2,
            testing_sequence2, training_labels2, testing_labels2
        , training_sequence3, testing_sequence3, training_labels3, testing_labels3, C_train, C_test]


def Load_data2(DB):
    Feat1 = np.load(f"../data_loader/Rainfall_Dataset/Features.npy")
    Lab1 = np.load(f"../data_loader/Rainfall_Dataset/Labels.npy")

    Feat2 = np.load("../data_loader/Temperature_Dataset/Features.npy")
    Lab2 = np.load("../data_loader/Temperature_Dataset/Labels.npy")

    Feat3 = np.load("../data_loader/Wind_Dataset/Features.npy")
    Lab3 = np.load("../data_loader/Wind_Dataset/Labels.npy")

    return [Feat1, Feat2, Feat3, Lab1, Lab2, Lab3]


def train_test_splitter(data, percent, num=500):
    feat, label = Load_data2(data)  # load your features and labels
    unique_classes = np.unique(label)

    selected_indices_per_class = []

    for cls in unique_classes:
        class_indices = np.where(label == cls)[0]
        # Randomly choose `num` samples per class without replacement
        selected_class_indices = np.random.choice(class_indices, num, replace=True)
        selected_indices_per_class.append(selected_class_indices)

    # Concatenate all selected indices from each class
    selected_indices = np.concatenate(selected_indices_per_class)

    # Shuffle the combined indices
    np.random.shuffle(selected_indices)

    # Get balanced features and labels
    balanced_feat = feat[selected_indices]
    balanced_label = label[selected_indices]

    data_size = balanced_feat.shape[0]
    split_point = int(data_size * percent)  # training data size

    # Split into train/test sets
    training_sequence = balanced_feat[:split_point]
    training_labels = balanced_label[:split_point]
    testing_sequence = balanced_feat[split_point:]
    testing_labels = balanced_label[split_point:]

    return training_sequence, testing_sequence, training_labels, testing_labels


def train_test_splitter1(data, percent, num=500):
    feat, label = Load_data(data)  # load your features and labels
    unique_classes = np.unique(label)

    selected_indices_per_class = []

    for cls in unique_classes:
        class_indices = np.where(label == cls)[0]
        # Randomly choose `num` samples per class without replacement
        selected_class_indices = np.random.choice(class_indices, num, replace=True)
        selected_indices_per_class.append(selected_class_indices)

    # Concatenate all selected indices from each class
    selected_indices = np.concatenate(selected_indices_per_class)

    # Shuffle the combined indices
    np.random.shuffle(selected_indices)

    # Get balanced features and labels
    balanced_feat = feat[selected_indices]
    balanced_label = label[selected_indices]

    data_size = balanced_feat.shape[0]
    split_point = int(data_size * percent)  # training data size

    # Split into train/test sets
    training_sequence = balanced_feat[:split_point]
    training_labels = balanced_label[:split_point]
    testing_sequence = balanced_feat[split_point:]
    testing_labels = balanced_label[split_point:]

    return training_sequence, testing_sequence, training_labels, testing_labels


def models_return_metrics(data, epochs, ok=True, percents=None, force_retrain=False):
    import os

    training_percentages = percents if percents is not None else [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # from Comparative_models.Model2 import CNN_BiLSTM_1
    # from Comparative_models.Model3 import RNN_1
    # from Comparative_models.Model4 import CNN_LSTM_1
    # from Comparative_models.Model5 import BiLSTM_1
    # from Comparative_models.Model6 import LSTM_1
    # from Comparative_models.Model7 import ANN_1
    model_registry = {
        "Model1": CNN_1,
        "Model2": CNN_BiLSTM_1,
        "Model3": RNN_1,
        "Model4": CNN_LSTM_1,
        "Model5": BiLSTM_1,
        "Model6": LSTM_1,
        "Model7": ANN_1
    }

    if ok:
        for model_name, model_fn in model_registry.items():
            cprint(f"\n==== Training model: {model_name} ====",color="white",on_color="on_blue")
            all_metrics = []

            for percent in training_percentages:
                cprint(f"  → Training {model_name} with {int(percent * 100)}% training data...",color="white",on_color="on_green")

                x_train1, x_test1, y_train1, y_test1, x_train2, x_test2, y_train2, y_test2, x_train3, x_test3, y_train3, y_test3, C_train, C_test = \
                    Regression_splitter1(data, percent=percent)

                metrics = model_fn(x_train1, x_test1, y_train1, y_test1, x_train2, x_test2, y_train2, y_test2, x_train3,
                                   x_test3, y_train3, y_test3, C_train, C_test, epochs)
                # else:
                #     metrics = model_fn(x_train1, x_test1, y_train1, y_test1, x_train2, x_test2, y_train2, y_test2,
                #                        x_train3, x_test3, y_train3, y_test3, C_train, C_test, epochs)

                all_metrics.append(metrics)

            # Save after all percentages
            save_path = f"Temp/data/Comp/{model_name}/all_metrics.npy"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, np.array(all_metrics, dtype=object))

            print(f"✔ Saved all metrics for {model_name} to {save_path}")
