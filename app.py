import streamlit as st
import joblib
import numpy as np
import pandas as pd
import string
import re
import io
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import contractions
import matplotlib.pyplot as plt

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
    """Load or retrain the model and TF-IDF vectorizer."""
    import os
    from sklearn.linear_model import LogisticRegression

    tfidf = joblib.load("models/tfidf_ml.pkl")

    model_path = "models/sentiment_model_ml.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        # Model pkl too large for git — retrain Logistic Regression on startup
        df = pd.read_csv(
            "Dataset/twitter_training.csv",
            header=None,
            names=["id", "topic", "sentiment", "text"],
        )
        df = df.dropna(subset=["text", "sentiment"])
        df = df[df["sentiment"].isin(["Positive", "Negative", "Neutral"])]
        X = tfidf.transform(df["text"].apply(preprocess_text))
        y = df["sentiment"]
        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(X, y)

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


def predict_single(text: str, model, tfidf):
    """Preprocess, vectorize, and predict a single text. Returns (label, confidence, probabilities)."""
    processed = preprocess_text(text)
    vec = tfidf.transform([processed])
    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]
    confidence = np.max(probabilities) * 100
    return prediction, confidence, probabilities


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

    # --------------- Tabs ---------------
    tab_single, tab_advanced = st.tabs(["Single Analysis", "Advanced Analysis"])

    # ==================== TAB 1: Single Analysis ====================
    with tab_single:
        st.subheader("Single Text Analysis")
        user_input = st.text_area(
            "Paste or type a tweet / review:",
            placeholder="Example: I love this product! It's fantastic.",
            height=120,
            key="single_input",
        )

        if st.button("Analyze Sentiment", use_container_width=True, type="primary", key="btn_single"):
            if not user_input.strip():
                st.warning("Please enter some text to analyze.")
            else:
                prediction, confidence, probabilities = predict_single(user_input, model, tfidf)
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

        # --- CSV Upload (original batch mode) ---
        st.markdown("---")
        st.subheader("Batch Analysis (CSV Upload)")
        st.markdown('Upload a CSV file containing a **"Reviews"** column.')

        uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_upload")
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
                    key="dl_csv_upload",
                )

    # ==================== TAB 2: Advanced Analysis ====================
    with tab_advanced:
        adv_section = st.radio(
            "Choose analysis mode:",
            ["Aspect-Based Sentiment", "Batch Text Analysis"],
            horizontal=True,
            key="adv_mode",
        )

        # ---------- Aspect-Based Sentiment ----------
        if adv_section == "Aspect-Based Sentiment":
            st.subheader("Aspect-Based Sentiment Analysis")
            st.markdown(
                "Split your text into individual sentences and see the sentiment of each part. "
                "This helps identify which aspects of a review are positive or negative."
            )

            aspect_input = st.text_area(
                "Paste a review or paragraph:",
                placeholder="Example: The food was amazing but the service was terrible. The ambiance was nice though.",
                height=150,
                key="aspect_input",
            )

            if st.button("Analyze Sentences", use_container_width=True, type="primary", key="btn_aspect"):
                if not aspect_input.strip():
                    st.warning("Please enter some text to analyze.")
                else:
                    sentences = sent_tokenize(aspect_input)
                    if not sentences:
                        st.info("No sentences detected.")
                    else:
                        results = []
                        for sent in sentences:
                            pred, conf, _ = predict_single(sent, model, tfidf)
                            results.append({
                                "Sentence": sent,
                                "Sentiment": pred,
                                "Confidence": f"{conf:.1f}%",
                            })

                        result_df = pd.DataFrame(results)

                        # Color-code sentiments in the table
                        def highlight_sent(val):
                            colors = {
                                "Positive": "background-color: #d4edda; color: #155724",
                                "Negative": "background-color: #f8d7da; color: #721c24",
                                "Neutral": "background-color: #fff3cd; color: #856404",
                            }
                            return colors.get(val, "")

                        styled = result_df.style.applymap(
                            highlight_sent, subset=["Sentiment"]
                        )
                        st.dataframe(styled, use_container_width=True)

                        # Quick summary
                        counts = result_df["Sentiment"].value_counts()
                        summary_parts = []
                        for label in ["Positive", "Negative", "Neutral"]:
                            if label in counts.index:
                                summary_parts.append(f"**{counts[label]}** {label.lower()}")
                        st.markdown(
                            f"Out of **{len(sentences)}** sentences: " + ", ".join(summary_parts)
                        )

        # ---------- Batch Text Analysis ----------
        else:
            st.subheader("Batch Text Analysis")
            st.markdown(
                "Paste multiple reviews below, **one per line**. "
                "All reviews will be analyzed at once."
            )

            batch_input = st.text_area(
                "Paste reviews (one per line):",
                placeholder="Great product, love it!\nTerrible quality, broke after a day.\nIt was okay, average.",
                height=200,
                key="batch_input",
            )

            if st.button("Analyze All", use_container_width=True, type="primary", key="btn_batch"):
                if not batch_input.strip():
                    st.warning("Please enter at least one review.")
                else:
                    lines = [line.strip() for line in batch_input.strip().splitlines() if line.strip()]
                    if not lines:
                        st.warning("No valid reviews found.")
                    else:
                        with st.spinner(f"Analyzing {len(lines)} reviews..."):
                            processed = [preprocess_text(l) for l in lines]
                            vectors = tfidf.transform(processed)
                            predictions = model.predict(vectors)
                            proba = model.predict_proba(vectors)
                            confidences = np.max(proba, axis=1) * 100

                        result_df = pd.DataFrame({
                            "Review": lines,
                            "Sentiment": predictions,
                            "Confidence": [f"{c:.1f}%" for c in confidences],
                        })

                        # --- Summary metrics ---
                        counts = pd.Series(predictions).value_counts()
                        pos_count = counts.get("Positive", 0)
                        neg_count = counts.get("Negative", 0)
                        neu_count = counts.get("Neutral", 0)

                        st.markdown("### Summary")
                        mcols = st.columns(3)
                        mcols[0].metric("Positive", pos_count)
                        mcols[1].metric("Negative", neg_count)
                        mcols[2].metric("Neutral", neu_count)

                        # --- Pie chart ---
                        st.markdown("### Sentiment Distribution")
                        fig, ax = plt.subplots(figsize=(4, 4))
                        labels = []
                        sizes = []
                        pie_colors = []
                        color_map = {
                            "Positive": "#28a745",
                            "Negative": "#dc3545",
                            "Neutral": "#ffc107",
                        }
                        for label in ["Positive", "Negative", "Neutral"]:
                            c = counts.get(label, 0)
                            if c > 0:
                                labels.append(label)
                                sizes.append(c)
                                pie_colors.append(color_map[label])

                        ax.pie(
                            sizes,
                            labels=labels,
                            colors=pie_colors,
                            autopct="%1.1f%%",
                            startangle=90,
                        )
                        ax.set_aspect("equal")
                        st.pyplot(fig)

                        # --- Results table ---
                        st.markdown("### Detailed Results")

                        def highlight_sent(val):
                            colors = {
                                "Positive": "background-color: #d4edda; color: #155724",
                                "Negative": "background-color: #f8d7da; color: #721c24",
                                "Neutral": "background-color: #fff3cd; color: #856404",
                            }
                            return colors.get(val, "")

                        styled = result_df.style.applymap(
                            highlight_sent, subset=["Sentiment"]
                        )
                        st.dataframe(styled, use_container_width=True)

                        # --- CSV download ---
                        csv_out = result_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download Results as CSV",
                            csv_out,
                            file_name="batch_sentiment_results.csv",
                            mime="text/csv",
                            key="dl_batch",
                        )


if __name__ == "__main__":
    main()
