import numpy as np


class KNN:
    n_neighbors = 5  # for k = {1,3,5,7,9}
    model_type = "Classification"  # Classification and Regression
    X_train = None
    y_train = None
    X_test = None
    y_pred = None
    dist_train_test = None
    weights = False  # Weighted or Unweighted

    def __init__(self, X_train, X_test, y_train, n_neighbors=5, model_type="Classification", weights=False):
        self.n_neighbors = n_neighbors
        self.model_type = model_type
        self.weights = weights
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train

    def set_n_neighbors(self, n_neighbors):
        self.n_neighbors = n_neighbors

    # KNN fit transform method calculates the distance between test data and train data by using distance method which is           defined below.
    def fit_transform(self):
        self.dist_train_test = np.empty((len(self.X_test), len(self.X_train)))
        for i in range(len(self.X_test)):
            for j in range(len(self.X_train)):
                self.dist_train_test[i][j] = distance(self.X_test[i], self.X_train[j])
        return self.dist_train_test

    # predict method generates y_pred list for each KNN model which are 'Unweighted Classification Model',                           'Weighted Classification Model', 'Unweighted Regression Model' and 'Weighted Regression Model'.
    def predict(self, X_test):
        self.y_pred = list()

        if not self.weights:
            # This block is for Unweighted Classification Model.

            if self.model_type == 'Classification':
                for rowidx in range(len(self.dist_train_test)):
                    row = self.dist_train_test[rowidx]
                    idx = np.argsort(row)[:self.n_neighbors]
                    vals, counts = np.unique(self.y_train[idx], return_counts=True)
                    index = np.argmax(counts)
                    target_idx = vals[index]
                    self.y_pred.append(target_idx)


            # This block is for Unweighted Regression Model.

            elif self.model_type == 'Regression':
                for rowidx in range(len(self.dist_train_test)):
                    total = 0
                    row = self.dist_train_test[rowidx]
                    idx = np.argsort(row)[:self.n_neighbors]
                    for j in idx:
                        total += self.y_train[j]
                    self.y_pred.append(total / len(idx))


        elif self.weights:
            # This block is for Weighted Classification Model.

            if self.model_type == 'Classification':
                for rowidx in range(len(self.dist_train_test)):
                    row = self.dist_train_test[rowidx]
                    idx = np.argsort(row)[:self.n_neighbors]
                    from collections import defaultdict
                    dflt_dict = defaultdict(float)
                    for i in range(self.n_neighbors):
                        if row[idx][i] == 0:
                            target_value = self.y_train[idx][i]
                            break
                        else:
                            dflt_dict[self.y_train[idx][i]] += 1 / row[idx][i]

                        target_value = max(dflt_dict.items(), key=lambda a: a[1])[0]
                    self.y_pred.append(target_value)


            # This block is for Weighted Regression Model.

            elif self.model_type == 'Regression':
                for rowidx in range(len(self.dist_train_test)):
                    weight_count = 0
                    total = 0
                    row = self.dist_train_test[rowidx]
                    idx = np.argsort(row)[:self.n_neighbors]
                    for j in idx:
                        total += self.y_train[j] * (1 / row[j])
                        weight_count += (1 / row[j])
                    self.y_pred.append(total / weight_count)

        return np.array(self.y_pred)


# distance function calcultes Euclidean Distance for KNN algorithm.
def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

