# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:04:04 2023

@author: Prakhar Agrawal
"""

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache
def load_data():
    return pd.read_csv("hospitals.csv")

def main():
    
    st.set_page_config(
        page_title="Hospital Recommender",
        page_icon="🏥",
        layout="wide",
    )

   
    st.title("Hospital Recommender")
    st.write("Enter your state, and we'll recommend hospitals for you.")

   
    dataset = load_data()
    dataset = dataset.dropna()

    
    user_input = st.text_input("Enter the name of your state:")

    if not user_input:
        st.warning("Please enter a state name.")
        st.stop()

    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataset['State'])
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, tfidf_matrix)
    similar_indices = [i for i, score in enumerate(similarity[0]) if score > 0.75]

    
    if similar_indices:
        st.subheader("Recommended Hospitals:")
        for idx in similar_indices:
            st.markdown(f"<span style='color: #FFB533;'>Name:</span> {dataset.iloc[idx]['Name']}", unsafe_allow_html=True)
            st.markdown(f"<span style='color: #3371FF;'>Address:</span> {dataset.iloc[idx]['Address']}", unsafe_allow_html=True)
            st.markdown("---", unsafe_allow_html=True)  
    else:
        st.warning("No records found for the given state.")

if __name__ == "__main__":
    main()
