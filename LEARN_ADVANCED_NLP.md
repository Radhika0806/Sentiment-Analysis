# Advanced NLP Features -- Concept Guide

## What's New?

Two advanced features added to the Sentiment Analysis app:

### 1. Aspect-Based Sentiment Analysis
Instead of one sentiment for the entire review, we analyze EACH SENTENCE separately.

Example:
> "The food was amazing but the service was terrible and the ambiance was okay."

| Sentence | Sentiment | Confidence |
|----------|-----------|------------|
| The food was amazing | Positive | 89% |
| the service was terrible | Negative | 92% |
| the ambiance was okay | Neutral | 67% |

This is way more useful than just "Negative" for the whole review.

**Interview answer:** "I implemented aspect-based sentiment by tokenizing input into
sentences using NLTK's sent_tokenize, then running each sentence through the TF-IDF
+ classifier pipeline independently. This gives per-aspect sentiment rather than
a single label for the whole text."

**Interview Q:** "How is this different from true aspect-based sentiment analysis?"
**Answer:** "True ABSA extracts specific aspects (food, service, price) and their
sentiments. My approach is sentence-level, which is simpler but still more granular
than document-level. For production ABSA, I'd use a fine-tuned BERT model with
aspect-sentiment pair extraction."

### 2. Batch Analysis with Export
Analyze hundreds of reviews at once, see distribution, download CSV results.
This is what companies actually need -- not one review at a time.

**Interview answer:** "I added batch processing because real business use cases involve
analyzing thousands of customer reviews at once. The system vectorizes all inputs in
one TF-IDF transform call for efficiency, generates summary statistics, visualizations,
and exportable CSV reports."

## Key Concepts

### Sentence Tokenization (sent_tokenize)
Splits text into sentences. Smarter than splitting on periods -- handles:
- "Dr. Smith" (doesn't split after Dr.)
- "U.S.A." (doesn't split after each dot)
- "Hello! How are you?" (splits on ! and ?)

### Vectorized Batch Processing
Instead of looping through reviews one by one:
```python
# Slow: loop
for review in reviews:
    vec = tfidf.transform([review])
    pred = model.predict(vec)

# Fast: vectorized
vecs = tfidf.transform(reviews)    # all at once
preds = model.predict(vecs)         # all at once
```
The vectorized version is 10-100x faster because sklearn uses numpy arrays internally.

### Confidence Scores (predict_proba)
`model.predict()` gives you a label. `model.predict_proba()` gives you probabilities.
Example: [0.1, 0.7, 0.2] means 10% Negative, 70% Positive, 20% Neutral.
The highest probability is the confidence score.

**Interview Q:** "How do you measure prediction confidence?"
**Answer:** "Random Forest's predict_proba returns the proportion of trees that voted
for each class. If 180 out of 200 trees voted Positive, confidence is 90%."

## Experiments to Try
1. Type a mixed review -- see how aspect-based catches both positive and negative parts
2. Paste 10 different reviews in batch mode -- see the pie chart distribution
3. Download the CSV -- this is what you'd show a business stakeholder

## Resources
- **NLTK tokenization:** nltk.org/api/nltk.tokenize.html
- **Aspect-Based Sentiment (research):** Search "ABSA BERT" for the state-of-the-art approach
- **CampusX NLP playlist:** YouTube, free, Hindi, covers tokenization to transformers
