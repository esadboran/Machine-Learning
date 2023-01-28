# calculate_accuracy function compares two different arrays which are y_pred and y_test. This function returns the similarity between these array's elements.
def calculate_accuracy(y_pred, y_test):
    count = 0
    for i in range(len(y_test)):
        if y_pred[i] != y_test[i]:
            count += 1
    return round((1 - count / len(y_test)), 5)


# precision function returns precision ratio of model according to the macro-averaging method. It compares y_test's and y_pred's elements for each element in label list and returns the similarity.
def precision(y_test, y_pred, label_list):
    prec = 0
    for i in label_list:
        correct = 0
        wrong = 0
        for j in range(len(y_pred)):
            if (y_pred[j] == i):
                if (y_test[j] == i):
                    correct += 1
                else:
                    wrong += 1
        precision_for_i = correct / (correct + wrong)
        prec += precision_for_i
    return prec / (len(label_list))


# recall function returns recall ratio of model according to macro-averaging method. It compares t_test's and y_pred's elements for each element in label list and return the similarity.
def recall(y_test, y_pred, label_list):
    rec = 0
    for i in label_list:
        correct = 0
        wrong = 0
        for j in range(len(y_test)):
            if (y_test[j] == i):
                if (y_pred[j] == i):
                    correct += 1
                else:
                    wrong += 1
        recall_for_i = correct / (correct + wrong)
        rec += recall_for_i
    return rec / len(label_list)


# mean_absolute_error function calculates the difference between y_pred's and y_test's all elements. Then it returns mean of the differences.
def mean_absolute_error(y_pred, y_test):
    count = 0
    for i in range(len(y_test)):
        count += abs(y_pred[i] - y_test[i])
    return count * (1 / len(y_test))













