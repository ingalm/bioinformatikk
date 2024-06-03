from tree import create_tree
from helper_functions import get_data, sigmoid, discrete_features, reverse_class_mapping
import numpy as np
import csv
from sklearn.model_selection import KFold

def create_and_train_tree(data):
    root = create_tree()
    populate_probabilities(root, data)
    return root

def populate_probabilities(node, data):
    if node is not None:
        node.calculate_conditional_probabilities(data)
        populate_probabilities(node.left, data)
        populate_probabilities(node.right, data)

def classify(tree, features):
    current_node = tree
    while current_node.class_label is None:
        feature_value = features.get(current_node.feature)

        if current_node.feature in discrete_features:
            # Check if the feature value is present in the probability dictionary
            if feature_value in current_node.probability:
                prob_left = current_node.probability[feature_value]['left']
                prob_right = current_node.probability[feature_value]['right']
            else:
                # If the feature value is not present, use the probabilities of the one value that is in the dictionary.
                # This is used in cases where the tree is not trained on samples of one kind, therefore not being able to identify it
                only_value = next(iter(current_node.probability))
                prob_left = current_node.probability[only_value]['left']
                prob_right = current_node.probability[only_value]['right']
                
            current_node = current_node.left if prob_left > prob_right else current_node.right
        
        else:
            # Use sigmoid function for continuous features
            a = current_node.sigmoid_params['a']
            b = current_node.sigmoid_params['b']
            prob = sigmoid(feature_value, a, b)
            current_node = current_node.right if prob > 0.5 else current_node.left

    return current_node.class_label

def validate_tree(tree, validation_data):
    class_statistics = {i: {'correct': 0, 'total': 0} for i in range(8)}
    predictions = []

    # Initialize dictionary to track correct predictions and total predictions per class
    for data_point in validation_data:
        class_label = data_point['class_label']
        if class_label not in class_statistics:
            class_statistics[class_label] = {'correct': 0, 'total': 0}
    
    # Classify each instance, update the statistics, and store predictions
    for data_point in validation_data:
        predicted_label = classify(tree, data_point)
        actual_label = data_point['class_label']
        
        predictions.append((data_point['Sequence Name'], reverse_class_mapping[predicted_label], reverse_class_mapping[actual_label]))
        
        class_statistics[actual_label]['total'] += 1
        if actual_label in class_statistics:
            if predicted_label == actual_label:
                class_statistics[actual_label]['correct'] += 1

    # Calculate and print accuracy for each class
    for class_label in class_statistics:
        correct = class_statistics[class_label]['correct']
        total = class_statistics[class_label]['total']
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"Class {reverse_class_mapping[class_label]} Accuracy: {accuracy:.2f}% (Correct: {correct}, Total: {total})")

    # Optionally, you can also calculate overall accuracy if needed
    overall_correct = sum(stats['correct'] for stats in class_statistics.values())
    overall_total = sum(stats['total'] for stats in class_statistics.values())
    overall_accuracy = (overall_correct / overall_total) * 100 if overall_total > 0 else 0
    print(f"Overall Validation Accuracy: {overall_accuracy:.2f}%")

    # Save predictions to a CSV file
    with open("predictions.csv", "w", newline='') as csvfile:
        fieldnames = ['Sequence Name', 'Predicted Label', 'Actual Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for sequence, predicted_label, actual_label in predictions:
            writer.writerow({'Sequence Name': sequence, 'Predicted Label': predicted_label, 'Actual Label': actual_label})

    return overall_accuracy


def print_tree(node, level=0):
    indent = "    " * level  # Creates indentation based on the level of the node
    if node is not None:
        if node.class_label is not None:
            print(f"{indent}{node.name} - Class Label: {node.class_label}")
        else:
            print(f"{indent}{node.name} - Feature: {node.feature}")
            if node.feature in discrete_features:
                print(f"{indent}  Probability Table: {node.probability}")
            else:
                print(f"{indent}  Sigmoid Params: {node.sigmoid_params}")

        print_tree(node.left, level + 1)
        print_tree(node.right, level + 1)
    else:
        print(f"{indent}None")


data = get_data()
root = create_and_train_tree(data)
print_tree(root)
validate_tree(root, data)

def cross_validate(data, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=30)
    accuracies = []

    for train_index, test_index in kf.split(data):
        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]

        root = create_and_train_tree(train_data)
        accuracy = validate_tree(root, test_data)
        accuracies.append(accuracy)

    overall_accuracy = np.mean(accuracies)
    print(f'Average accuracy over {k} folds: {overall_accuracy:.2f}%')

#cross_validate(data, k=10)

import matplotlib.pyplot as plt

def plot_sigmoids():
    def plot_sigmoid(feature_values, a, b, title):
        x = np.linspace(min(feature_values), max(feature_values), 100)
        y = sigmoid(x, a, b)
        plt.plot(x, y, label=f'sigmoid(a={a:.2f}, b={b:.2f})')
        plt.scatter(feature_values, sigmoid(feature_values, a, b), color='red', s=5)
        plt.xlabel('Feature Value')
        plt.ylabel('Sigmoid Output')
        plt.title(title)
        plt.legend()
        plt.show()

    feature_values_root = np.array([item['mcg'] for item in data])
    plot_sigmoid(feature_values_root, root.sigmoid_params.get("a"), root.sigmoid_params.get("b"), 'Sigmoid for root (mcg)')

    # Plot sigmoid for other nodes if needed
    # gvh
    feature_values_gvh = np.array([item['gvh'] for item in data])
    plot_sigmoid(feature_values_gvh, root.left.right.sigmoid_params.get("a"), root.left.right.sigmoid_params.get("b"), 'Sigmoid for gvh')

    # alm2
    feature_values_alm2 = np.array([item['alm2'] for item in data])
    plot_sigmoid(feature_values_alm2, root.left.right.right.sigmoid_params.get("a"), root.left.right.right.sigmoid_params.get("b"), 'Sigmoid for alm2')

    # alm1
    feature_values_alm1 = np.array([item['alm1'] for item in data])
    plot_sigmoid(feature_values_alm1, root.right.sigmoid_params.get("a"), root.right.sigmoid_params.get("b"), 'Sigmoid for alm1')

    # aac
    feature_values_aac = np.array([item['aac'] for item in data])
    plot_sigmoid(feature_values_aac, root.left.right.right.left.sigmoid_params.get("a"), root.left.right.right.left.sigmoid_params.get("b"), 'Sigmoid for aac')

# Plot the learned sigmoid curves
plot_sigmoids()