
import numpy as np


class Tree:

    nodes = None
    edges = None

    def __init__(self, attribute=None):
        self.nodes = attribute
        self.edges = dict()

    def add(self, attribute, subtree):
        self.edges[attribute] = subtree

    @staticmethod
    def max_gain(data, column, label):#Find gain
        arr = dict()
        for x in column:
            arr[x] = Tree.calc_info_gain(data, x, label)
        return max(arr, key=arr.get)

    @staticmethod
    def main_entropy(data, label): #Find S entropy
        sumValues = 0
        for x in data[label].value_counts():
            sumValues += - (x / len(data)) * np.log2(x / (len(data)))
        return sumValues

    @staticmethod
    def entropy(data, attribute, typ, label):  #Find all entropy
        num = data[data[attribute] == typ][label].value_counts() / len(data[data[attribute] == typ][label])
        entropy = np.sum(-num * np.log2(num))
        return entropy

    @staticmethod
    def calc_info_gain(data, attribute, label): #Find all gain
        gain = Tree.main_entropy(data, label)
        count = data[attribute].value_counts() / len(data[attribute])

        for typ in data[attribute].unique():

            entropy= Tree.entropy(data, attribute, typ, label)
            gain -= count[typ] * entropy
        return gain

  
    @staticmethod
    def create_tree(root, prev_feature_value, train_data, label, columns):
        if len(train_data) != 0 : # if we encounter a special data, we have this data set and you may get length of data 0.

            subtree = False
            
            while not subtree:
                if len(columns) <= 0:
                    return
                
                max_info_feature = Tree.max_gain(train_data, columns, label)  # We find the Attribute and Value to add to our tree
                columns.remove(max_info_feature)
                subtree, train_data = Tree.generate_sub_tree(max_info_feature, train_data, label)


            if prev_feature_value != None:  # we add subtree to our tree
                new_node = subtree
                root.add(prev_feature_value, new_node)
                next_root = new_node
                
            else:  
                root.nodes = subtree.nodes
                root.edges = subtree.edges
                next_root = subtree

            for branch, node in list(next_root.edges.items()): 
                if node == "?":
                    feature_value_data = train_data[train_data[max_info_feature] == branch]
                    Tree.create_tree(next_root, branch, feature_value_data, label, columns)

    @staticmethod
    def generate_sub_tree(feature_name, train_data, label):
        subtree = Tree(feature_name)  # create a subtree with column with max gain

        feature_value_count = train_data[feature_name].unique() #Ex:  Age: Small or Medium or Large

        if len(feature_value_count) < 2:  # If our node has only one child  we return false and remove it
            return False, train_data

        for feature_value in feature_value_count:
            feature_value_data = train_data[train_data[feature_name] == feature_value]  # We replace the data with the data we customized

            leaf_node_control = False

            for c in train_data[label].unique():  # c = yes or no 
                if len(feature_value_data[label].unique()) == 1: # if the  data label in the remaining data is the same we add a decision tree
                    subtree.add(feature_value, c)
                    train_data = train_data[train_data[feature_name] != feature_value]
                    leaf_node_control = True
            
            #For example, for the data that we may encounter in the train set 
            #that is not in our train set, ? We put them in and process them later.
            if not leaf_node_control: 
                subtree.add(feature_value, "?")

        return subtree, train_data


