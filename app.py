import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the trained models
with open('model.pkl', 'rb') as f:
    linear_regression_model = pickle.load(f)

with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

# Function to predict waiting time
def predict_waiting_time(patient_type, financial_class, weekday, hours):
    # Prepare input features for prediction
    input_data = {'patient_type': [patient_type],
                  'financial_class': [financial_class],
                  'weekday': [weekday],
                  'hours': [hours]}
    
    # Convert input features to DataFrame
    input_df = pd.DataFrame(input_data)
    
    # One-hot encode categorical features
    input_df = pd.get_dummies(input_df, columns=['patient_type', 'financial_class', 'weekday'])
    
    # Ensure that all columns are present, and in the same order, as during training
    expected_columns = ['hours', 'financial_class_HMO', 'financial_class_INSURANCE',
       'financial_class_MEDICARE', 'financial_class_PRIVATE', 'weekday_Monday',
       'weekday_Saturday', 'weekday_Sunday', 'weekday_Thursday',
       'weekday_Tuesday', 'weekday_Wednesday']
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match the order during training
    input_df = input_df[expected_columns]
    # Make prediction using the model
    predicted_waiting_time = linear_regression_model.predict(input_df)
    
    return predicted_waiting_time[0]

# Function to visualize clustering
def visualize_clustering(df):
    # Add cluster labels to the original dataframe
    df['cluster'] = kmeans_model.labels_

    # Visualize the clusters
    plt.figure(figsize=(10, 6))
    for cluster in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]
        plt.hist(cluster_data['waiting_per_minutes'], bins=20, alpha=0.5, label=f'Cluster {cluster}')
    plt.xlabel('Waiting Time (minutes)')
    plt.ylabel('Frequency')
    plt.title('Waiting Time Clusters')
    plt.legend()
    st.pyplot()

# Streamlit App
def main():
    st.title('Patient Waiting Time Prediction & Clustering Dashboard')

    df = pd.read_csv('hospitalwaiting.csv')
    
    # Sidebar for input parameters
    st.sidebar.title('Input Parameters')
    patient_type = st.sidebar.selectbox('Patient Type', ['OUTPATIENT'])
    financial_class = st.sidebar.selectbox('Financial Class', ['HMO', 'INSURANCE', 'MEDICARE', 'PRIVATE'])
    weekday = st.sidebar.selectbox('Weekday', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    hours = st.sidebar.slider('Entry Time (Hours)', 0, 23, 12)
    
    # Predict waiting time
    predicted_time = predict_waiting_time(patient_type, financial_class, weekday, hours)
    
    st.subheader('Predicted Waiting Time:')
    st.write(f'The predicted waiting time is: {predicted_time:.2f} minutes')
    
    # Visualize clustering
    st.subheader('Waiting Time Clustering:')
    visualize_clustering(df)

if __name__ == "__main__":
    main()