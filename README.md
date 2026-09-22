# CODSOFT_TASK4 — Spam SMS Detection

Classifies SMS messages as spam or legitimate using TF-IDF and classic machine learning classifiers.

## 📌 Problem Statement

Given the text of an SMS message, predict whether it is spam or legitimate (ham) — a text classification problem with moderate class imbalance (~13.4% spam).

## 📂 Dataset

[SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (Kaggle / UCI)

- `spam.csv` — 5,572 labeled SMS messages (4,825 ham, 747 spam)

## 🛠️ Approach

1. **Data Cleaning**
   - Dropped unused columns
   - Mapped labels to binary (ham=0, spam=1)
   - Stratified 80/20 train/test split to preserve the spam ratio

2. **Feature Extraction**
   - TF-IDF vectorization (English stopwords removed, top 3,000 features)
   - Fit on training data only, applied to test data — no data leakage

3. **Model Training**
   Trained and compared three classifiers:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Support Vector Machine (linear kernel)

4. **Evaluation**
   Evaluated using precision, recall, F1-score, confusion matrix, and ROC-AUC for the spam class, then tested the best model on new, unseen example messages.

## 📊 Results

| Model | Precision (spam) | Recall (spam) | F1-score | ROC-AUC |
|---|---|---|---|---|
| Naive Bayes | 0.98 | 0.81 | 0.89 | **0.989** |
| Logistic Regression | **1.00** | 0.79 | 0.88 | 0.986 |
| SVM | 0.98 | **0.88** | **0.93** ✅ (best) | 0.985 |

SVM performed best overall — highest F1-score and recall (catching 131 of 149 spam messages) while keeping precision high (only 2 false positives). Logistic Regression achieves perfect precision (zero false positives) but misses more spam, making it a reasonable alternative if minimizing false alarms is the priority.

Verified on new example messages — the model correctly classified all of them:
- ✅ "Congratulations! You've won a $1000 gift card..." → SPAM
- ✅ "Hey, are we still meeting for lunch tomorrow?" → HAM
- ✅ "URGENT: Your account has been suspended..." → SPAM
- ✅ "Can you send me the notes from today's class?" → HAM

## 📁 Repository Contents

- `CODSOFT_TASK4_Spam_SMS_Detection.ipynb` — full notebook (preprocessing, TF-IDF, model training, evaluation)
- `README.md` — this file

## ▶️ How to Run

1. Open the notebook in Google Colab or Jupyter
2. Upload `spam.csv`
3. Run all cells in order

## 🔮 Possible Improvements

- Word embeddings (Word2Vec, GloVe) instead of TF-IDF
- Hyperparameter tuning via grid search
- Handling of SMS-specific noise (abbreviations, emojis)
