# Predicting Yelp Review Star Ratings Using Text Mining and Machine Learning

## 1. Background and Motivation
Online review platforms like Yelp host millions of reviews, but the relationship between a review’s text and its star rating is complex and non-obvious. Being able to accurately predict star ratings from text alone has several benefits:

- Helps readers quickly assess sentiment without reading every review.
- Assists businesses in automatically flagging underperforming areas.
- Enables researchers to understand which features of a business influence ratings.

Previous research shows strong performance for binary sentiment classification (positive vs. negative) but much weaker results for multi-class star prediction (1–5 stars), due to:

- **Class imbalance** (many more 4–5 star reviews).
- **Different reasons for the same star rating.**
- **Noisy, user-generated text** (spelling, slang, sarcasm).

This project builds on these findings and explores methods that combine robust text preprocessing, multiple feature extraction techniques, and a range of machine learning models.

## 2. Project Objectives
- **Primary Goal:** Predict the star rating (1–5) of Yelp restaurant reviews based solely on the review text.  
- **Secondary Goals:**
  - Evaluate which feature extraction approaches (n-grams, TF-IDF, sentiment lexicons, embeddings) yield the best performance.
  - Test additional potential features (i.e. length of review, # exclamation points).
  - Address class imbalance.
  - Compare baseline models to more advanced architectures such as deep neural networks and ensembles.
  - Experiment with regressors (rounding up to nearest star rating and maintaining output between 1-5) over classifiers.

## 3. Dataset
We will use the Yelp Dataset Challenge data (restaurant reviews):

- Large sample size (~7-8 million reviews).
- Text + star rating available.
- Known imbalance: about two-thirds of reviews have ≥4 stars.

## 4. Methodology

### 4.1 Preprocessing
- Lowercasing.
- Removal of punctuation, special characters, and stop words (after extracting additional features like length of review).
- Handling negations (e.g., `not good` → `not_good`).
- Optional: Stemming.
- (Future work) Part-of-speech tagging to extract adjective–noun pairs.

### 4.2 Feature Extraction
We will compare multiple approaches:

1. **Unigrams, Bigrams, Trigrams:** Bag-of-words with TF-IDF weighting to reduce common-word dominance.  
2. **Hashing Vectorizer:** For scalable representation without storing vocabulary.  
3. **Latent Semantic Indexing (LSI):** SVD on the term-review matrix to identify latent topics.  
4. **Contextual Embeddings:** Pre-trained BERT or similar models to capture context-sensitive meanings.  
5. **Additional Features (if feasible):** Review length, counts of positive/negative words, or other metadata (“useful,” “cool” votes).

### 4.3 Modeling
We will experiment with a range of models:

- **Baseline classifiers:** Logistic Regression, Multinomial Naive Bayes, Linear/Non-Linear SVM, Passive-Aggressive Classifier.
- **Tree-based ensembles:** Random Forest, AdaBoost.
- **Neural networks:** Shallow and deep Multi-Layer Perceptrons with BERT embeddings.
- **Hybrid/ensemble approaches:** Weighted voting or stacking of baseline + MLP models.
- **(Future work) Ordinal models:** Ordered logistic regression to reflect the ordinal nature of star ratings.

### 4.4 Evaluation Metrics
- k-fold cross-validation on training data to select hyperparameters.
- Report **Accuracy** and **RMSE** on test data.
- Analyze per-class performance (especially on minority classes like 2- and 3-star reviews).

## 5. Anticipated Challenges
- **Class Imbalance:** Majority of 4–5 star reviews may skew predictions; we may try resampling, class weighting, or ordinal models.  
- **Noisy Text:** Misspellings, slang, and sarcasm reduce model effectiveness.  
- **Distinguishing Adjacent Ratings:** Harder to tell 4 vs. 5 stars or 2 vs. 3 stars from text alone. If the results for predicting exact star rating are still poor after exhausting refining options, may try relabeling as positive/negative/neutral.

## 6. Expected Outcomes
- Baseline performance similar to literature (≈60–65% accuracy for unigrams+bigrams + Logistic Regression or SVM).  
- Potential improvements from contextual embeddings + ensemble methods.  
- Insights on which features matter most for differentiating ratings, and which classes remain hardest to predict.

## 7. Future Directions
- Aspect-level sentiment analysis (e.g., food quality vs. service vs. price).  
- Sarcasm detection or spelling normalization.  
- Incorporating user- or business-level features for personalized predictions.  
- Exploring non-linear or ordinal classifiers designed for ordered star ratings.
