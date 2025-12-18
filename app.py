import streamlit as st
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt_tab')

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [
        ps.stem(word) for word in words
        if word.isalnum() and word not in stop_words
    ]
    return " ".join(words)

# Streamlit UI
st.title("📩 Spam Detection using ML & NLP")
st.write("Enter a message to check whether it is **Spam** or **Not Spam**.")

user_input = st.text_area("Enter your message:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed = preprocess(user_input)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT SPAM")
