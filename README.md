
# 📌 Sentiment Analysis using Machine Learning
This project applies Machine Learning techniques to classify user reviews into sentiment categories (Positive / Negative). The goal is to explore text preprocessing, feature extraction, and traditional ML algorithms for effective sentiment classification.

# 📂 Dataset
Total Samples: ~74,000 reviews for training/testing + ~1,000 reviews in a separate validation dataset.
Columns:
Reviews → Raw text data (user reviews).
Sentiment → Target label (Positive / Negative).
Preprocessing Applied:
Lowercasing
Removal of punctuation, numbers, special characters
Stopword removal
Tokenization
Stemming (Porter Stemmer)

Example Data
Review (cleaned) Sentiment
"the product was excellent and fast"	Positive
"very poor quality not recommended"	Negative

# 🔑 Key Steps

Text Preprocessing → cleaning, tokenization, stopword removal, stemming
Feature Engineering → Bag of Words (CountVectorizer), TF-IDF Vectorizer
Models Implemented → Logistic Regression, Naïve Bayes, Support Vector Machine (LinearSVC), Random Forest
Evaluation Metrics → Accuracy, Precision, Recall, F1-score
Model Saving → Trained models and vectorizers stored as .pkl files

# 📊 Results
Model	CountVectorizer (BoW)	TF-IDF	Validation (BoW)	Validation (TF-IDF)
Logistic Regression	0.88	0.89	0.86	0.87
Naïve Bayes	0.85	0.86	0.83	0.84
SVM (LinearSVC)	0.87	0.90	0.86	0.88
Random Forest	0.86	0.87	0.84	0.85

# 📌 Observations
SVM with TF-IDF achieved the best accuracy on both test and validation data.
Logistic Regression remained consistent and reliable across methods.
Naïve Bayes was computationally efficient but slightly less accurate.
Random Forest generalized decently but required more resources.

📂 Project Structure
📦 Sentiment-Analysis-ML
 ┣ 📂 data/               # Dataset (train/test/validation)
 ┣ 📂 models/             # Saved models (.pkl files)
 ┣ 📂 notebooks/          # Jupyter Notebook workflow
 ┣ 📜 requirements.txt    # Dependencies
 ┣ 📜 README.md           # Documentation

# 🔮 Future Work
This project can be extended with more advanced approaches:
Deep Learning Models 
LSTM (Long Short-Term Memory) networks
BiLSTM / GRU for richer sequence modeling
Lexicon-based Approaches
VADER Sentiment Analyzer for quick rule-based sentiment scoring
Transformer Models (Hugging Face)
BERT, DistilBERT, RoBERTa for state-of-the-art performance
Deployment
Streamlit / Flask app for real-time sentiment prediction

# ⚙️ Tech Stack
Python 🐍
Scikit-learn
Pandas, NumPy
Matplotlib, Seaborn
NLTK
Pickle / Joblib

# 🚀 How to Run
# Clone the repository
git clone https://github.com/your-username/sentiment-analysis-ml.git
cd sentiment-analysis-ml
# Install dependencies
pip install -r requirements.txt
# Run Jupyter Notebook
jupyter notebook "Sentiment Analysis ML.ipynb"

✨ This is an ongoing project. Future updates will add Deep Learning and Transformer-based models for improved accuracy.
