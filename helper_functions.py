import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

discrete_features = ['lip', 'chg']
class_mapping = {'cp': 0, 'im': 1, 'pp': 2, 'imU': 3, 'om': 4, 'omL': 5, 'imL': 6, 'imS': 7}
reverse_class_mapping = {value: key for key, value in class_mapping.items()}

def get_data():
    column_names = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
    data_df = pd.read_csv("ecoli.data", header=None, names=column_names, sep='\s+')

    # Mapping classes to integers
    data_df['class_label'] = data_df['class'].map(class_mapping)

    # Convert DataFrame to a list of dictionaries
    data_list = data_df.to_dict(orient='records')

    return data_list

def sigmoid(x, a, b):
    return 1 / (1 + np.exp(a * x + b))

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    # Initialize parameters a and b
    a, b = 1.0, 0.0

    # Gradient descent
    for _ in range(epochs):
        # Calculate the predictions using the current parameters
        predictions = sigmoid(X, a, b)
        
        # Calculate the error
        errors = predictions - y
        
        # Calculate the gradients
        gradient_a = np.sum(errors * predictions * (1 - predictions) * X)
        gradient_b = np.sum(errors * predictions * (1 - predictions))
        
        # Update the parameters
        a += learning_rate * gradient_a
        b += learning_rate * gradient_b 

    return a, b

def fit_sigmoid_params(node, data):
    # Extract relevant feature and class label
    feature_values = np.array([item[node.feature] for item in data])
    
    # Create binary labels: 1 if the data point belongs to the right subtree classes, 0 otherwise
    class_labels = np.array([1 if item['class_label'] in node.get_class_labels(node.right) else 0 for item in data])

    # Learn the sigmoid parameters using gradient descent
    a, b = gradient_descent(feature_values, class_labels)

    # Update node's sigmoid parameters
    node.sigmoid_params = {'a': a, 'b': b}