import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def load_data():
    return pd.read_csv("C:\\Users\\Prakhar Agrawal\\Downloads\\Final_data.csv")

def main():
    
    st.set_page_config(
        page_title="Ayurvedic Medicine Recommender",
        page_icon="🌿",
        layout="wide",
    )

    
    st.title("Ayurvedic Medicine Recommender by HerbIQ")
    st.write("Enter a disease, and we'll recommend Ayurvedic medicines for you.")

   
    dataset = load_data()
    main_features = dataset.dropna()

    
    user_disease = st.text_input("Enter the disease name:")

    if not user_disease:
        st.warning("Please enter a disease name.")
        st.stop()

    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(main_features['Diseases Cured'])
    user_input_vector = tfidf_vectorizer.transform([user_disease])
    cosine_similarities = cosine_similarity(user_input_vector, tfidf_matrix)
    similar_indices = [i for i, score in enumerate(cosine_similarities[0]) if score > 0.5]

   
    if similar_indices:
        st.subheader("Recommended Medicines:")
        for idx in similar_indices:
            st.markdown(f"<span style='color: #FF5733;'>Ayurvedic Medicine:</span> {main_features.iloc[idx]['Ayurvedic Medicine']}", unsafe_allow_html=True)
            st.markdown(f"<span style='color: #3371FF;'>Used in treatment of:</span> {main_features.iloc[idx]['Diseases Cured']}", unsafe_allow_html=True)
            st.markdown(f"<span style='color: #33FF45;'>Cautions and Considerations:</span> {main_features.iloc[idx]['Cautions and Precautions']}", unsafe_allow_html=True)
            st.markdown(f"<span style='color: #FFB533;'>Properties:</span> {main_features.iloc[idx]['Properties']}", unsafe_allow_html=True)
            st.markdown(f"<span style='color: #A633FF;'>Key Ingredients:</span> {main_features.iloc[idx]['Key Ingredients']}", unsafe_allow_html=True)
            st.markdown(f"<span style='color: #FF33CE;'>Mode of Action:</span> {main_features.iloc[idx]['Mode of Action']}", unsafe_allow_html=True)
            st.markdown("---", unsafe_allow_html=True)  # Add a horizontal line between recommendations
    else:
        st.warning("No records found for the given disease.")

if __name__ == "__main__":
    main()
