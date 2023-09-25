# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 23:07:29 2023

@author: Prakhar Agrawal
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data for Ayurvedic medicine, hospitals, and colleges
medicine_data = pd.read_csv(r"C:/Users/Prakhar Agrawal/Downloads/Final_data.csv" , encoding='utf-8')
medicine_data = medicine_data.dropna()

hospital_data = pd.read_csv(r"C:/Users/Prakhar Agrawal/Downloads/hospitals.csv" , encoding='utf-8')
hospital_data = hospital_data.dropna()

college_data = pd.read_csv(r"C:/Users/Prakhar Agrawal/Downloads/ayurvedic-colleges_2- (1).csv" , encoding='utf-8')
college_data = college_data.drop_duplicates(subset='College ID')


def recommend_medicines(user_disease):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(medicine_data['Diseases Cured'])
    user_input_vector = tfidf_vectorizer.transform([user_disease])
    cosine_similarities = cosine_similarity(user_input_vector, tfidf_matrix)
    similar_indices = [i for i, score in enumerate(cosine_similarities[0]) if score > 0.5]

    recommendations = []
    for idx in similar_indices:
        recommendation = {
            "Ayurvedic Medicine": medicine_data.iloc[idx]['Ayurvedic Medicine'],
            "Diseases Cured": medicine_data.iloc[idx]['Diseases Cured'],
            "Cautions and Considerations": medicine_data.iloc[idx]['Cautions and Precautions'],
            "Properties": medicine_data.iloc[idx]['Properties'],
            "Key Ingredients": medicine_data.iloc[idx]['Key Ingredients'],
            "Mode of Action": medicine_data.iloc[idx]['Mode of Action']
        }
        recommendations.append(recommendation)

    return recommendations

# Function to recommend Ayurvedic Hospitals
def recommend_hospitals(user_state):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(hospital_data['State'])
    user_vector = vectorizer.transform([user_state])
    similarity = cosine_similarity(user_vector, tfidf_matrix)
    similar_indices = [i for i, score in enumerate(similarity[0]) if score > 0.75]

    recommendations = []
    for idx in similar_indices:
        recommendation = {
            "Name": hospital_data.iloc[idx]['Name'],
            "Address": hospital_data.iloc[idx]['Address']
        }
        recommendations.append(recommendation)

    return recommendations

# Function to recommend Ayurvedic Colleges
def recommend_colleges(user_state):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(college_data['State'])
    user_matrix = vectorizer.transform([user_state])
    similarity = cosine_similarity(user_matrix, tfidf_matrix)
    similar_indices = [i for i, score in enumerate(similarity[0]) if score > 0.5]

    recommendations = []
    for idx in similar_indices:
        recommendation = {
            "College ID": college_data.iloc[idx]['College ID'],
            "Name of the College": college_data.iloc[idx]['Name of the College'],
            "State": college_data.iloc[idx]['State']
        }
        recommendations.append(recommendation)

    return recommendations

# Streamlit UI
st.sidebar.title("Ayurvedic Recommendations")

# Sidebar section for user input
section = st.sidebar.radio("Select a Section", ["Ayurvedic Medicine", "Ayurvedic Hospitals", "Ayurvedic Colleges"])

if section == "Ayurvedic Medicine":
    user_disease = st.text_input("Enter the disease name:")
    if st.button("Recommend Medicines"):
        st.subheader("Recommended Medicines:")
        recommendations = recommend_medicines(user_disease)
        for recommendation in recommendations:
            st.write(recommendation)

elif section == "Ayurvedic Hospitals":
    user_state = st.text_input("Enter the name of your state:")
    if st.button("Recommend Hospitals"):
        st.subheader("Recommended Hospitals:")
        recommendations = recommend_hospitals(user_state)
        for recommendation in recommendations:
            st.write(recommendation)

elif section == "Ayurvedic Colleges":
    user_state = st.text_input("Enter your state name:")
    if st.button("Recommend Colleges"):
        st.subheader("Recommended Colleges:")
        recommendations = recommend_colleges(user_state)
        for recommendation in recommendations:
            st.write(recommendation)
