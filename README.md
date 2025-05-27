# 🛡️ Spam vs Ham Classifier (Streamlit App)

A machine learning-powered web app to classify SMS messages as **Spam** or **Ham (Not Spam)**. Built with **Streamlit**, this app demonstrates the practical application of classification models, text preprocessing, and confidence scoring for real-world text data.

---

## 🔍 Problem Statement
Classify a given message as spam or ham using a machine learning model trained on a dataset of SMS messages. Spam messages are rare compared to ham, so handling **class imbalance** was a key challenge.

---

## 🧠 Models Used

- **Multinomial Naive Bayes**: Traditionally effective for text classification, especially with TF-IDF features. It performed well but was slightly less robust to false positives (spam predicted as ham).
- **Random Forest**: Performed well after upsampling but had a high training time and potential overfitting.
- ✅ **Logistic Regression (Best Performer)**:
  - **Highest F1-score** on the minority class (spam)
  - Balanced precision and recall
  - Generalized better to edge cases when trained on upsampled data
  - Fast, interpretable, and robust — ideal for deployment

---

## ⚖️ Handling Class Imbalance

- The dataset had a significant skew:
  - **Ham**: ~4825 samples
  - **Spam**: ~747 samples
- Applied **upsampling** using `sklearn.utils.resample` to balance classes in the training set.
- Models were evaluated using **F1-score**, particularly for spam detection.

---

## 🔧 Features

- TF-IDF vectorization on the text
- Streamlit UI with:
  - Text input
  - Confidence score display
- Backend pipeline: `TFIDFVectorizer` → `LogisticRegression`

---

## 📈 Performance Summary

| Model              | Spam F1-Score | Ham F1-Score | Accuracy |
|-------------------|---------------|--------------|----------|
| Logistic Regression | **0.99**       | **0.95**      | **0.99**   |
| Multinomial NB     | 0.98          | 0.90         | 0.97     |
| Random Forest      | 0.89          | 0.99         | 0.97     |

---

## 🧩 Edge Cases & Limitations

- Some borderline spam messages (e.g., messages with polite or generic language) may be misclassified as ham.
- Confidence scores provide transparency but may still be high even for misclassifications.

---

## 🚀 Future Improvements

- Use **threshold tuning** to adjust sensitivity toward spam detection.
- Implement **BERT-based transformer models** for deeper contextual understanding.
- Include **user feedback mechanism** for continuous learning.
- Add **data augmentation** techniques (e.g., synonym replacement, paraphrasing) to enhance minority class robustness.

---

## 📦 Run the App

```bash
streamlit run app.py
