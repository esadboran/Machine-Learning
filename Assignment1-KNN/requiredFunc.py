import random
import numpy as np
import pandas as pd
from KNN_model import KNN
from Calculations import *


# shuff function takes a list and changes its all elements' indexes.
def shuff(lis):
    shuff_data = list()
    x = [x for x in range(len(lis))]
    for i in range(len(lis)):
        num = random.choice(x)
        x.remove(num)
        shuff_data.append(lis[num])

    return np.array(shuff_data)


# normalize function takes a list and makes its all elements' values between 0 and 1 according to the Min-Max Scaling rules.
def normalize(X, norm='MinMax'):
    X_normalized = np.empty(np.shape(X))
    for col in range(X.shape[1]):
        X_normalized[:, col] = (X[:, col] - X[:, col].min()) / (X[:, col].max() - X[:, col].min())
    return X_normalized


# train_test_split_kfold function applies the k-fold Cross Validation rules. It takes a list and then seperates it to test and train lists according to the k value.
def train_test_split_kfold(X, y, k_fold_iterations, k_fold=5):
    idx = int(len(X) / k_fold)
    folds_x = [X[:idx], X[idx:idx * 2], X[idx * 2:idx * 3], X[idx * 3:idx * 4], X[idx * 4:idx * 5]]
    folds_y = [y[:idx], y[idx:idx * 2], y[idx * 2:idx * 3], y[idx * 3:idx * 4], y[idx * 4:idx * 5]]
    X_test = folds_x[k_fold_iterations - 1]
    y_test = folds_y[k_fold_iterations - 1]
    X_train = np.delete(folds_x, [k_fold_iterations - 1], 0)
    y_train = np.delete(folds_y, [k_fold_iterations - 1], 0)
    X_train = X_train.reshape(idx * (k_fold - 1), np.shape(X)[1])  # 60 == special features nums
    y_train = y_train.reshape(idx * (k_fold - 1))

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def cross_val(X, y):
    cross_val = list()
    for i in range(5):
        cross_val.append(train_test_split_kfold(X, y, i, k_fold=5))
    cross_val = np.array(cross_val, dtype=object)

    # cross_val = [X1,X2,X3,X4,X5]   X = [X_train, X_test , y_train, y_test]
    return np.array(cross_val)


# this method use 5-fold cross validation to try k-nn models with different k value.
# then it show them the dataframe.
def results_cv(X, y, model_type, weights, normalized, calculations):
    array_np_tmp = np.empty((5, 5))
    array_np_tmp_1 = np.empty((5, 5))
    array_np_tmp_2 = np.empty((5, 5))
    array_np_tmp_3 = np.empty((5, 5))
    array_df = list()
    if normalized == True:
        cross_valid = cross_val(normalize(X), y)
    else:
        cross_valid = cross_val(X, y)
    for j in range(5):
        X_train, X_test, y_train, y_test = cross_valid[j]
        knnmodel = KNN(X_train, X_test, y_train, n_neighbors=5, model_type=model_type, weights=weights)
        dist_train_set = knnmodel.fit_transform()
        for i in [1, 3, 5, 7, 9]:
            knnmodel.set_n_neighbors(i)
            y_pred = knnmodel.predict(X_test)
            if (calculations == "metric"):
                array_np_tmp_1[j][int((i - 1) / 2)] = calculate_accuracy(y_pred, y_test)
                array_np_tmp_2[j][int((i - 1) / 2)] = precision(y_pred, y_test, pd.unique(y))
                array_np_tmp_3[j][int((i - 1) / 2)] = recall(y_pred, y_test, pd.unique(y))
            elif (calculations == "mae"):
                array_np_tmp[j][int((i - 1) / 2)] = mean_absolute_error(y_pred, y_test)
                df = pd.DataFrame(array_np_tmp, index=['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5'],
                                  columns=["mae_1", "mae_3", "mae_5", "mae_7", "mae_9"])
                array_df.append(df)
    if (calculations == "metric"):
        df_1 = pd.DataFrame(array_np_tmp_1, index=['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5'],
                            columns=[1, 3, 5, 7, 9])
        array_df.append(df_1)
        df_2 = pd.DataFrame(array_np_tmp_2, index=['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5'],
                            columns=[1, 3, 5, 7, 9])
        array_df.append(df_2)
        df_3 = pd.DataFrame(array_np_tmp_3, index=['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5'],
                            columns=[1, 3, 5, 7, 9])
        array_df.append(df_3)
        return array_df
    elif (calculations == "mae"):
        return df
