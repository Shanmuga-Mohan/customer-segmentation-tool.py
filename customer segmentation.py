import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
data = {
    'CustomerID': np.arange(1, 51),
    'Age': np.random.randint(18, 70, size=50),
    'Annual Income (k$)': np.random.randint(15, 150, size=50),
    'Spending Score (1-100)': np.random.randint(1, 101, size=50)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

# Save the dataset to a CSV file
df.to_csv('customer_data_sample.csv', index=False)
import pandas as pd

# Load the dataset
df = pd.read_csv('customer_data_sample.csv')

# Select relevant features for clustering
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Display the first few rows to confirm it's loaded correctly
print(df.head())
from sklearn.cluster import KMeans

# Initialize KMeans with a chosen number of clusters, e.g., 3
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model to the data
kmeans.fit(X)

# Predict the clusters for each customer
df['Cluster'] = kmeans.predict(X)

# Display the dataset with the cluster labels
print(df.head())

# Optionally, save the clustered dataset to a CSV file
df.to_csv('customer_data_clustered.csv', index=False)
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv('customer_data_sample.csv')
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Train K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Streamlit UI
st.title("Customer Segmentation Tool")

# Input fields for new customer data
age = st.number_input('Age', min_value=18, max_value=70, value=25)
income = st.number_input('Annual Income (k$)', min_value=15, max_value=150, value=50)
spending_score = st.number_input('Spending Score (1-100)', min_value=1, max_value=100, value=50)

# Predict cluster
new_data = pd.DataFrame({'Age': [age], 'Annual Income (k$)': [income], 'Spending Score (1-100)': [spending_score]})
cluster = kmeans.predict(new_data)[0]

st.write(f"The customer belongs to Cluster: {cluster}")

# Display clustered data
st.write("Clustered Customer Data")
st.dataframe(df)
