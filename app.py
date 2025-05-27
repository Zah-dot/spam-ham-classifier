import streamlit as st
import joblib
import numpy as np

model = joblib.load("spam_classifier.pkl")

# App title
st.title("📨 Spam Message Classifier")

st.markdown("""
This app classifies a given message as **Spam** or **Ham** using a machine learning model trained on SMS data.  
It also shows the prediction **confidence** and flags **uncertain predictions** for better transparency.
""")

user_input = st.text_area("✍️ Enter the message to classify", height=150)

if st.button("🔍 Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        pred_proba = model.predict_proba([user_input])[0]
        pred_class = model.predict([user_input])[0]

        confidence = np.max(pred_proba)
        label = "Ham" if pred_class == 1 else "Spam"
        st.markdown(f"### 🔎 Prediction: `{label}`")
        st.markdown(f"**Confidence**: {confidence * 100:.2f}%")
        if confidence < 0.80:
            st.warning("⚠️ This prediction has **low confidence**. Please review the message manually.")

        # Optional: show probability breakdown
        st.markdown("---")
        st.markdown("#### 📊 Class Probabilities")
        st.write({
            "Spam": round(pred_proba[0]*100, 2),
            "Ham": round(pred_proba[1]*100, 2)
        })

        st.markdown("---")
        st.info("""
🧠 **How it works**:  
The model uses a TF-IDF vectorizer to convert text into numeric form and then classifies it using a trained machine learning model. The confidence is based on the model's prediction probability.
        """)
