import streamlit as st
import joblib
import numpy as np
import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import contractions

# Download required NLTK data
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Page config
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .positive {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .negative {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .neutral {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# --------------- helpers ---------------

@st.cache_resource
def load_models():
    """Load the pre-trained model and TF-IDF vectorizer."""
    model = joblib.load("models/sentiment_model_ml.pkl")
    tfidf = joblib.load("models/tfidf_ml.pkl")
    return model, tfidf


def preprocess_text(text: str) -> str:
    """Replicate the notebook preprocessing pipeline."""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Expand contractions
    text = contractions.fix(text)
    # Tokenize
    tokens = word_tokenize(text)
    # Stopword removal (keep negations, matching notebook)
    stop_words = set(stopwords.words("english"))
    negation = {
        "no", "not", "nor", "never", "cannot",
        "don't", "didn't", "isn't", "wasn't",
        "won't", "wouldn't", "shouldn't", "couldn't",
        "aren't", "weren't", "haven't", "hasn't",
        "hadn't", "doesn't", "don't",
    }
    stop_words -= negation
    tokens = [w for w in tokens if w not in stop_words]
    # Stemming
    ps = PorterStemmer()
    tokens = [ps.stem(w) for w in tokens]
    return " ".join(tokens)


def color_class(sentiment: str) -> str:
    return {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"}.get(
        sentiment, "neutral"
    )


# --------------- main app ---------------

def main():
    st.title("Sentiment Analysis App")
    st.markdown("---")

    model, tfidf = load_models()

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown(
            """
        **Model:** Random Forest Classifier
        **Vectorizer:** TF-IDF
        **Classes:** Positive, Negative, Neutral
        **Training data:** Twitter Sentiment Dataset
        """
        )
        st.markdown("---")
        st.markdown("**Sample inputs:**")
        st.markdown(
            """
- \"This product is amazing! Highly recommended\"
- \"Worst experience ever, very disappointed\"
- \"It's okay, nothing special\"
        """
        )

    # --- Single text mode ---
    st.subheader("Single Text Analysis")
    user_input = st.text_area(
        "Paste or type a tweet / review:",
        placeholder="Example: I love this product! It's fantastic.",
        height=120,
    )

    if st.button("Analyze Sentiment", use_container_width=True, type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            processed = preprocess_text(user_input)
            vec = tfidf.transform([processed])
            prediction = model.predict(vec)[0]

            # Confidence via predict_proba (Random Forest supports it)
            probabilities = model.predict_proba(vec)[0]
            confidence = np.max(probabilities) * 100
            cls = color_class(prediction)

            st.markdown(
                f"""
                <div class="prediction-box {cls}">
                    <h2>{prediction}</h2>
                    <h3>Confidence: {confidence:.1f}%</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Per-class scores
            st.subheader("Confidence Scores")
            class_map = {c: i for i, c in enumerate(model.classes_)}
            cols = st.columns(3)
            for col, label in zip(cols, ["Positive", "Negative", "Neutral"]):
                with col:
                    prob = probabilities[class_map.get(label, 0)] * 100
                    st.metric(label, f"{prob:.1f}%")

    # --- Batch mode ---
    st.markdown("---")
    st.subheader("Batch Analysis (CSV Upload)")
    st.markdown('Upload a CSV file containing a **"Reviews"** column.')

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "Reviews" not in df.columns:
            st.error('The CSV must contain a column named "Reviews".')
        else:
            with st.spinner("Analyzing..."):
                df["Processed"] = df["Reviews"].astype(str).apply(preprocess_text)
                vectors = tfidf.transform(df["Processed"])
                df["Sentiment"] = model.predict(vectors)

            st.success(f"Analyzed {len(df)} rows.")

            # Color-code the sentiment column
            def highlight_sentiment(val):
                colors = {
                    "Positive": "background-color: #d4edda; color: #155724",
                    "Negative": "background-color: #f8d7da; color: #721c24",
                    "Neutral": "background-color: #fff3cd; color: #856404",
                }
                return colors.get(val, "")

            styled = df[["Reviews", "Sentiment"]].style.applymap(
                highlight_sentiment, subset=["Sentiment"]
            )
            st.dataframe(styled, use_container_width=True)

            # Download button
            csv_out = df[["Reviews", "Sentiment"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results as CSV",
                csv_out,
                file_name="sentiment_results.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
