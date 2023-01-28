import copy

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from Tree import *


class DecisionTreeClassifier:

    
    def fit(self, X_train, y_train, label, pruning=False):
        self.label = label
        self.tree = Tree()
        self.prune_list = list()
        #İf pruining is true , we create validation set
        if pruning:
            #İf pruining is true , we create validation set (20/80 = 0.25)
            X_train_prune, X_valid, y_train_prune, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
            
            self.data = pd.concat([X_train_prune, y_train_prune], axis=1) # We concat X_train and Y_train
            Tree.create_tree(self.tree, None, self.data, self.label, list(X_train.columns)) #Create a Decision Tree
            self.pruning_decision_tree(self.tree, self.data, self.label, X_valid, y_valid)  # Pruining Process
        else:
            self.data = pd.concat([X_train,y_train],axis=1)  #We concat X_train and Y_train
            
            Tree.create_tree(self.tree, None, self.data, self.label, list(X_train.columns)) #Create a Decision Tree

    def predict(self, X_test):
        y_pred = list()
        for _,e in X_test.iterrows():
                pre = self.fun(e) #Predict for each value [Yes,No]
                y_pred.append(pre)
        return y_pred

    def fun(self,e):
        try:
            att = e[self.tree.nodes]
            node = self.tree.edges[att]
            while isinstance(node, Tree):
                att = e[node.nodes]
                node = node.edges[att]
            if node == "?": # If we have not encountered such a data set, we write 'no' instead.
                return 'No'
            else:
                return node 
        except:
            return 'No'
    
    
    """– Step 1: Catalog all twigs in the tree
        – Step 2: Find the twig with the least Information Gain
        – Step 3: Remove all child nodes of the twig
        – Step 4: Relabel twig as a leaf (Set the majority of ”Positive” or ”Negative” as leaf value)
        – Step 5: Measure the accuracy value of your decision tree model with removed twig on the validation set (”Current Accuracy”)
        – If ”Current Accuracy ≥ Last Accuracy” : Jump to ”Step1”
        Else : Revert the last changes done in Step 3,4 and then terminate"""
    
    
    def pruning_decision_tree(self, node, data, label, X_valid, y_valid):
        prune = copy.deepcopy(self)
        twig = dict()
        self.minGain(node, data, label, twig)  # Find the twig with the least Information Gain
        attribute = min(twig, key=twig.get)  # Find the twig with the least Information Gain
        self.downsizing(prune.tree, attribute, data, label)

        curr_accuracy = prune.calculate_accuracy(X_valid, y_valid)
        last_accuracy = self.calculate_accuracy(X_valid, y_valid)
        if curr_accuracy >= last_accuracy: #If ”Current Accuracy ≥ Last Accuracy” : Jump to ”Step1”
            self.tree = prune.tree
            self.pruning_decision_tree(self.tree, data, label, X_valid, y_valid)

    def calculate_accuracy(self, X_valid, y_valid): 

        return accuracy_score(self.predict(X_valid), y_valid)

    def downsizing(self, node, attribute, data, label): #Contiune pruning work
        for k, v in node.edges.items():
            if isinstance(v, str):
                continue
            if v.nodes == attribute:
                new_label = data[data[node.nodes] == k][label].value_counts().index[0]
                node.edges[k] = new_label
            else:
                next_node = v
                feature_value_data = data[data[node.nodes] == k]
                self.downsizing(next_node, attribute, feature_value_data, label)

    def minGain(self, root_node, data, label, twig):  # Find the twig with the least Information Gain
        liste = list(map(lambda x: isinstance(x, str), root_node.edges.values()))

        if False not in liste:
            twig[root_node.nodes] = Tree.calc_info_gain(data, root_node.nodes, label)
        else:
            for k, v in root_node.edges.items():
                next_node = v
                if not isinstance(v, str):
                    feature_value_data = data[data[root_node.nodes] == k]
                    self.minGain(next_node, feature_value_data, label, twig)
