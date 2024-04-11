from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def makeDecisionTree(data):
    scaler = MinMaxScaler()
    def entropy(target_col):
        elements, counts = np.unique(target_col, return_counts=True)
        entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
        return entropy

    # Calculate the information gain of a dataset
    def InfoGain(data, split_attribute_name, target_name="class"):
        # Calculate the entropy of the total dataset
        total_entropy = entropy(data[target_name])

        # Calculate the values and the corresponding counts for the split attribute
        vals, counts = np.unique(data[split_attribute_name], return_counts=True)

        # Calculate the weighted entropy
        Weighted_Entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])

        # Calculate the information gain
        Information_Gain = total_entropy - Weighted_Entropy
        return Information_Gain

    # ID3 Algorithm
    def ID3(data, originaldata, features, target_attribute_name="class", parent_node_class = None):
        # If all target_values have the same value, return this value
        if len(np.unique(data[target_attribute_name])) <= 1:
            return np.unique(data[target_attribute_name])[0]

        # If the dataset is empty, return the mode target feature value in the original dataset
        elif len(data) == 0:
            return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]

        # If the feature space is empty, return the mode target feature value of the direct parent node
        elif len(features) == 0:
            return parent_node_class

        # If none of the above conditions holds true, grow the tree!
        else:
            # Set the default value for this node --> The mode target feature value of the current node
            parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

            # Select the feature which best splits the dataset
            item_values = [InfoGain(data, feature, target_attribute_name) for feature in features] # Return the information gain values for the features in the dataset
            best_feature_index = np.argmax(item_values)
            best_feature = features[best_feature_index]

            # Create the tree structure. The root gets the name of the feature with the maximum information gain
            tree = {best_feature:{}}

            # Remove the feature with the best information gain from the feature space
            features = [i for i in features if i != best_feature]

            # Grow a branch under the root node for each possible value of the root node feature
            for value in np.unique(data[best_feature]):
                value = value
                # Split the dataset along the value of the feature with the largest information gain and create sub_datasets
                sub_data = data.where(data[best_feature] == value).dropna()

                # Call the ID3 algorithm for each of those sub_datasets with the new parameters
                subtree = ID3(sub_data, originaldata, features, target_attribute_name, parent_node_class)

                # Add the sub tree, grown from the sub_dataset to the tree under the root node
                tree[best_feature][value] = subtree

            return tree
    X = data.drop(columns=['Heart_Disease'])
    y = data['Heart_Disease']
    x = X.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled=min_max_scaler.fit_transform(x)


    model = KMeans(n_clusters=2)
    model = model.fit(x_scaled)
    all = data.columns
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=422)
    decision_tree = ID3(data, data,all[:-1], all[-1])
    print(decision_tree)

def predict(data, tree):
    # Print the instance data for debugging
    print("Data:", data)
    
    # If the current node is a leaf node, return its class label
    if not isinstance(tree, dict):
        # Print the predicted class label
        print("Leaf Node - Predicted Class:", tree)
        return tree

    # Otherwise, find the attribute to test by getting the key of the tree
    attribute = list(tree.keys())[0]
    print("Current Attribute:", attribute)
    
    # Get the value of the current attribute from the instance data
    instance_value = data[attribute]
    print("Instance Value for Current Attribute:", instance_value)
    
    # Get the branches for the current attribute
    branches = tree[attribute].keys()
    print("Branches:", branches)
    
    # Traverse the next branch based on the instance's attribute value
    if instance_value in branches:
        print("Traversing Branch:", instance_value)
        return predict(data, tree[attribute][instance_value])
    else:
        # If the attribute value is not in the tree, we cannot make a prediction
        print("Attribute Value not found in Tree. Cannot Predict.")
        return 'Outlier'