import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# # Define the column names based on the dataset description
column_names = [
    'sequence_name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class'
]

# Load the dataset
file_path = 'ecoli.data'
ecoli_data = pd.read_csv(file_path, sep='\s+', names=column_names)

# Group the data by class and calculate the mean for each feature
feature_means = ecoli_data.drop(columns='sequence_name').groupby('class').mean()

# Plotting the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(feature_means, annot=True, cmap='viridis')
plt.title('Heatmap of Feature Means by Class')
plt.show()
