# 🎬 Movie Recommendation System (Learning Project)

## 📖 Overview

This project is a hands-on implementation of basic recommender system techniques using the MovieLens 100K dataset.

The goal is to understand how recommendation systems work, starting from simple popularity-based models to evaluation using ranking metrics.

---

## 📂 Dataset

* Dataset used: MovieLens 100K
* Contains:

  * 100,000 ratings
  * 943 users
  * 1682 movies

---

## 🚀 Features Implemented

### 1. Data Preprocessing

* Cleaned and formatted rating data
* Extracted year from release date
* Handled missing values

### 2. Exploratory Data Analysis (EDA)

* Univariate analysis of ratings and users
* Distribution of ratings and activity levels

### 3. Global Top-N Recommender

* Recommends movies based on:

  * Average rating
  * Minimum number of ratings threshold
* Same recommendations for all users

### 4. Per-User Recommendation

* Generates recommendations for each user by:

  * Taking global top-N movies
  * Excluding movies already rated by the user

### 5. Genre-Based Recommendation

* Filters movies by genre
* Recommends top-N movies within a genre

---

## 📊 Evaluation

### Metric Used: Precision@10

* For each user:

  * Generate top-10 recommendations
  * Compare with held-out test ratings
* Compute average Precision@10 across users

### Train/Test Split

* Per-user split (80% train, 20% test)
* Ensures each user appears in both sets

### Baseline Result

* Precision@10 ≈ **0.07–0.08**

---

## 🔮 Future Improvements

* User-based collaborative filtering
* Item-based collaborative filtering
* Matrix factorization (SVD)
* Incorporating genre preferences
* Using Recall@K and other metrics

---

## 🛠️ Tech Stack

* Python
* pandas
* numpy
* scikit-learn

---

## ▶️ How to Run

1. Load the dataset
2. Run preprocessing steps
3. Train the recommender functions
4. Evaluate using Precision@10

---

## 📌 Author

This project is part of my learning journey in recommender systems and machine learning.
