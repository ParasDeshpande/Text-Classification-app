# 💬 Sentiment Analysis Trainer App

An interactive Streamlit application that allows users to upload their own `.csv` files containing sentences and labels, trains a **Logistic Regression** model using **TF-IDF** features, and predicts sentiment in real time — even for complex sentences involving conjunctions and mixed opinions.

---

## 📋 Features

- 📂 Upload a custom CSV file (`text`, `label` columns).
- 🧠 Train a sentiment analysis model live on your data.
- ⚡ Supports complex sentences with conjunctions ("but," "although," "however," etc.).
- 🧪 Real-time testing: Enter a new sentence and predict its sentiment immediately.
- 💾 Saves the trained model and vectorizer locally (`sentiment_model.pkl`).

---

## 🛠 Technologies Used

| Purpose           | Tools                 |
|-------------------|------------------------|
| Web App           | Streamlit              |
| Machine Learning  | scikit-learn (Logistic Regression, TF-IDF) |
| Data Handling     | pandas                 |
| Deployment Ready  | Streamlit Cloud support |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/sentiment-trainer-app.git
cd sentiment-trainer-app
```

---

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Project Structure
```
sentiment-trainer-app/
├── app.py               # Streamlit App
├── requirements.txt     # Required Python packages
├── sentiment_model.pkl   # (Generated after training)
├── sample_data/
│   └── sentiment_conjunctions.csv  # Example training data
└── README.md
```

---

### 4. Run the Application
```bash
streamlit run app.py
```

The app will open automatically at [http://localhost:8501](http://localhost:8501).

---

## 🧠 About the Dataset

The app expects a `.csv` file with:
- **text**: Sentence or review (string)
- **label**: Sentiment (Positive, Negative, or any custom class)

Example:

| text                                             | label     |
|--------------------------------------------------|-----------|
| I like the design but hate the battery life      | Negative  |
| The service was slow although the food was great | Positive  |

---

## 🏁 Future Improvements

- Handle neutral/mixed sentiment classification.
- Highlight conjunction influence on model decisions.
- Export trained models directly to cloud storage (AWS, GCP).

---

## 📄 License

This project is licensed under the MIT License - feel free to use and modify it!
