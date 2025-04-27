import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

st.title("💬 Train Your Own Sentiment Analysis Model")
st.write("Upload a CSV file with two columns: `text` and `label`.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'text' in df.columns and 'label' in df.columns:
        st.success("✅ File uploaded and format verified!")
        
        # Display sample data
        st.subheader("📄 Sample Data")
        st.write(df.head())

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

        # Vectorize text
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Train model
        model = LogisticRegression()
        model.fit(X_train_vec, y_train)

        # Evaluate model
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"✅ Model trained with accuracy: **{acc:.2f}**")

        # Save model and vectorizer (optional)
        with open("sentiment_model.pkl", "wb") as f:
            pickle.dump((model, vectorizer), f)

        # Allow user to test model
        st.subheader("🧪 Test the Model")
        user_input = st.text_area("Enter a review or sentence to analyze:")

        if st.button("Analyze Sentiment"):
            input_vec = vectorizer.transform([user_input])
            prediction = model.predict(input_vec)[0]
            st.write("🔍 Prediction:", f"**{prediction}**")

    else:
        st.error("❌ CSV must contain 'text' and 'label' columns.")
