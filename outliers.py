import pandas as pd
import numpy as np
from scipy.spatial import distance

# Load your heart_rate data
df = pd.read_csv('combined_measurements.csv')

# Assuming df is your DataFrame and it has columns 'id', 'activity', 'heart_rate'
def detect_outliers_distance(df, threshold):
    outliers = []
    for _, sub_df in df.groupby(['id', 'activity']):
        center = sub_df['heart_rate'].mean()  # Calculate the center point (e.g., mean) for each group
        distances = distance.euclidean(sub_df['heart_rate'].values.flatten(), np.full_like(sub_df['heart_rate'], center))  # Compute distances from center point
        sub_df['outlier'] = distances > threshold  # Mark outlier data
        outliers.append(sub_df[sub_df['outlier']])  # Append outliers to the list
    outliers_df = pd.concat(outliers)  # Concatenate outliers from all groups
    return outliers_df

threshold = 20  # Adjust the distance threshold as per your needs
outliers_df = detect_outliers_distance(df, threshold)

print(outliers_df)



