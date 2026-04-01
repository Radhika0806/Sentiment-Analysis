# Sentiment Analysis ML

A machine learning application for analyzing sentiment in tweets and reviews using a pre-trained Random Forest classifier with TF-IDF vectorization.

## 🎯 What This Project Does

This project predicts the sentiment of text inputs (tweets, reviews, comments) into three categories:
- **Positive** 😊
- **Negative** 😞
- **Neutral** 😐

The model was trained on a large Twitter sentiment dataset and achieves approximately **92.5% accuracy** on validation data.

## 📊 Model Details

- **Algorithm**: Random Forest Classifier (200 estimators)
- **Feature Extraction**: TF-IDF Vectorizer
- **Max Features**: 5,000
- **Training Data**: Twitter Sentiment Dataset (61,120 samples)
- **Classes**: 3 (Positive, Negative, Neutral)
- **Validation Accuracy**: 92.51%

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. Clone or download this project
2. Navigate to the project directory:
   ```bash
   cd "Sentiment Analysis ML"
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
Sentiment Analysis ML/
├── app.py                      # Streamlit deployment app
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── models/                     # Pre-trained models
│   ├── sentiment_model_ml.pkl # Random Forest model
│   ├── tfidf_ml.pkl           # TF-IDF Vectorizer
│   └── vectorizer_ml.pkl      # Count Vectorizer (alternative)
├── Dataset/                    # Training and validation data
└── Sentiment Analysis ML.ipynb # Jupyter notebook with full training pipeline
```

## 📝 How to Use

1. **Open the app** - Run `streamlit run app.py` in your terminal
2. **Enter text** - Paste or type a tweet or review in the text area
3. **Click analyze** - Press the "Analyze Sentiment" button
4. **View results** - See the predicted sentiment and confidence scores

### Example Inputs

- "This product is amazing! I love it!" → **Positive**
- "Worst customer service ever." → **Negative**
- "It works as expected." → **Neutral**

## 🛠️ Features

- **Real-time sentiment prediction** with confidence scores
- **Beautiful UI** built with Streamlit
- **Model caching** for fast predictions
- **Detailed confidence breakdown** showing probabilities for all sentiment classes
- **Responsive design** that works on desktop and mobile

## 📊 Model Training Pipeline

The full training pipeline can be found in `Sentiment Analysis ML.ipynb`. It includes:

1. **Data Loading** - Twitter sentiment dataset
2. **Data Preprocessing**:
   - Lowercasing
   - Punctuation removal
   - Number removal
   - Contraction expansion
   - Tokenization
   - Stopword removal (preserving negations)
   - Stemming (Porter Stemmer)

3. **Feature Extraction** - TF-IDF Vectorization
4. **Model Training** - Random Forest and other classifiers
5. **Model Evaluation** - Accuracy, confusion matrix, metrics
6. **Model Persistence** - Saving trained models

## 🔧 Dependencies

- **streamlit** - Web app framework
- **scikit-learn** - Machine learning library
- **joblib** - Model serialization
- **pandas** - Data manipulation
- **numpy** - Numerical computing

## 📈 Performance Metrics

| Model | Accuracy (Validation) |
|-------|----------------------|
| Logistic Regression | 94.44% |
| Random Forest | **92.51%** ✓ |
| SVM | 85.51% |
| Naive Bayes | 74.88% |

*Note: Random Forest was selected for deployment due to good balance of performance and prediction confidence*

## 🎓 Learning Resources

This project demonstrates:
- NLP text preprocessing techniques
- Feature extraction with TF-IDF
- Model training and evaluation
- Deployment with Streamlit
- Model serialization with joblib

## 🐛 Troubleshooting

**Models not loading?**
- Ensure you're running the app from the project root directory
- Check that the `models/` folder contains all three `.pkl` files

**Port 8501 already in use?**
```bash
streamlit run app.py --server.port 8502
```

**Installation issues?**
```bash
pip install --upgrade -r requirements.txt
```

## 📄 License

This project is provided as-is for educational purposes.

## 🤝 Contributing

Feel free to:
- Improve the UI/UX
- Add new preprocessing techniques
- Retrain with additional data
- Create additional deployment options (Flask, FastAPI, etc.)

---

**Built with ❤️ using Streamlit and scikit-learn**
