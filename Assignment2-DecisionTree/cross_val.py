import pandas as pd

from DecisionTreeClass import *
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# A function that divides the data by 5 and makes 5 different models and returns us the data performance measurements.
def cross_validation(model, X, y, label, pruning, k_split=5):
    idx = X.index.values.copy()
    np.random.shuffle(idx) # We shuffled values
    
    folds_idx = np.split(idx, k_split) #Divide 5 
    #We create a 4 list for performance matrix
    accuracy = list()
    f1 = list()
    precision = list()
    recall = list()
    
    for i in range(k_split): 
        tmp = folds_idx.copy()
        test_ind = tmp.pop(i)
        train_ind = [x for sub in tmp for x in sub]
        X_train, X_test, y_train, y_test = X.iloc[train_ind], X.iloc[test_ind], y[train_ind], y[test_ind]
        model.fit(X_train, y_train, label, pruning)  # fit model (model = DecisionTreeClassfier())
        y_pred = model.predict(X_test)  # model predict
        
        #Model performance values
        f1.append(f1_score(y_test, y_pred, average="binary", pos_label="No")) 
        precision.append(precision_score(y_test, y_pred, average="binary", pos_label="No"))
        recall.append(recall_score(y_test, y_pred, average="binary", pos_label="No"))
        accuracy.append(accuracy_score(y_test, y_pred))

    df = pd.DataFrame(list(zip(accuracy, recall, precision, f1)),
                      columns=["Accuracy", "Recall", "Precision", "F1_score"])
    df = df.T
    return df
