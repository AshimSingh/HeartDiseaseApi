import pandas as pd
import numpy as np
import sys
sys.setrecursionlimit(1500)

# Small value to avoid division by zero
EPSILON = np.finfo(float).eps

# Function to calculate entropy of the dataset
def find_entropy(df):
    # Assuming the last column is the class label
    Class = df.keys()[-1] #returns heart disease aailye laai
    entropy = 0
    values = df[Class].unique()
    # print(values)  #0 1
    for value in values:
        fraction = df[Class].value_counts()[value] / len(df[Class])
        #print(df[Class].value_counts()[value],len(df[Class])) #120 yes 150 No
        # Entropy formula: -p*log2(p)
        entropy += -fraction * np.log2(fraction + EPSILON) # 0.9910760598382216 entropy of whole dataset
    return entropy

# Function to calculate entropy of a specific attribute
def find_entropy_attribute(df, attribute):
    Class = df.keys()[-1]
    # print(Class)
    target_variables = df[Class].unique()
    variables = df[attribute].unique()
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute] == variable][df[Class] == target_variable])
            den = len(df[attribute][df[attribute] == variable])
            fraction = num / (den + EPSILON)
            # Entropy formula: -p*log2(p)
            entropy += -fraction * np.log2(fraction + EPSILON)
        fraction2 = den / len(df)
        entropy2 += -fraction2 * entropy
    return abs(entropy2)

# Function to find the attribute with the highest information gain
def find_winner(df):
    IG = []
    for key in df.keys()[:-1]:
        # Calculate information gain for each attribute
        IG.append(find_entropy(df) - find_entropy_attribute(df, key))
    # Return the attribute with the highest information gain
    return df.keys()[:-1][np.argmax(IG)]

# Function to get a subtable of the dataset based on a specific attribute value
def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)

# Function to build the decision tree recursively
def build_decision_tree(df, tree=None):
    # print(df)
    # find_entropy(df)
    Class = df.keys()[-1]
    # Get the attribute with the highest information gain
    node = find_winner(df)
    attValue = np.unique(df[node])
    if tree is None:
        tree = {}
        tree[node] = {}
    for value in attValue:
        
        # Get subset of data based on attribute value
        subtable = get_subtable(df, node, value)
        clValue, counts = np.unique(subtable['Heart_Disease'], return_counts=True)
        if len(counts) == 1:
            # If subset is pure, assign the class label
            tree[node][value] = clValue[0]
        else:
            # Recursively build the tree
            tree[node][value] = build_decision_tree(subtable)
    return tree


def classify_instance(test, tree, default=2):
    attribute = next(iter(tree))
    if test[attribute] in tree[attribute].keys():
        result = tree[attribute][test[attribute]]
        if isinstance(result, dict):
            return classify_instance(test, result)
        else:
            return result
    else:
        return default