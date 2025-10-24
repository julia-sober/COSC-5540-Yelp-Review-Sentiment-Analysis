# Yelp Review Sentiment Analysis  
*Course: COSC 5540 -- Text Mining and Analysis*  
*Author: Julia Sober and Cole Hanrahan*  
*Date: Fall 2025*

---

## Overview  
This project analyzes Yelp reviews to predict sentiment using machine learning models.  
The primary goal is to predict each review's **1-5 star rating** based on the text content and metadata. This task is also called **Review Rating Prediction (RRP)**. 

We explore:
- Text preprocessing and feature extraction (TF-IDF, n-grams, etc.)  
- Comparative performance of models like **SVM**, **Random Forest**, and **XGBoost**  
- Interpretability through feature importance and error analysis  

---

## Motivation  
Yelp reviews contain valuable information about customer opinions and business performance.  
Understanding sentiment at scale can help:
- Businesses identify strengths and weaknesses  
- Customers make informed decisions  
- Researchers explore real-world NLP applications  

This project demonstrates how to build and evaluate a sentiment analysis pipeline from raw data to model interpretation.

---

## Dataset  
**Source:** Yelp public dataset 

**Key Features:**
- `text` — the content of each review  
- `stars` — numeric rating (1–5)  
- `date`, `business_id`, `user_id`, etc.  

**Preprocessing Steps:**
- Removing stopwords and punctuation  
- Tokenization  
- Lemmatization or stemming  
- Converting text into numerical features using TF-IDF  
- Balancing classes (if needed)

---

## ⚙️ Methodology  

### 1. **Exploratory Data Analysis (EDA)**  
- Review length distribution  
- Correlation between stars and sentiment  
- Most frequent positive/negative words  
- Word clouds and visual summaries  

### 2. **Feature Engineering**  
- TF-IDF and Bag-of-Words representations  
- N-grams for contextual features  
- Optional metadata features (e.g., business category)
