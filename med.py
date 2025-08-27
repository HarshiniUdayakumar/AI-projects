import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Text cleaning ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text.strip()

# --- Load and preprocess dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv('medquad.csv')
    df['question'] = df['question'].apply(clean_text)
    return df

# --- Symptom and synonym maps ---
generic_symptoms = {
    "stomach pain": "Stomach pain can be caused by indigestion, gas, infection, or stress. If severe, see a doctor.",
    "headache": "Headache can be due to stress, dehydration, or lack of sleep. Persistent headaches should be checked by a doctor.",
    "fever": "Fever may indicate an infection. Stay hydrated and rest. If it persists, consult a doctor.",
    "cough": "A cough can be caused by cold, flu, or allergies. If it lasts more than a week, see a doctor."
}

synonyms = {
    "abdominal pain": "stomach pain",
    "migraine": "headache",
    "temperature": "fever",
    "sore throat": "cough",
    "body ache": "fever",
    "cold": "cough"
}

# --- Load data and setup TF-IDF ---
df = load_data()
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['question'])

# --- Medical answer logic ---
def get_medical_answer(user_input):
    user_input_clean = clean_text(user_input)

    for key, val in synonyms.items():
        if key in user_input_clean:
            user_input_clean = user_input_clean.replace(key, val)

    for symptom, advice in generic_symptoms.items():
        if symptom in user_input_clean:
            return f"Generic Advice:\n{advice}"

    user_vec = vectorizer.transform([user_input_clean])
    cosine_sim = cosine_similarity(user_vec, tfidf_matrix)
    idx = np.argmax(cosine_sim)
    score = cosine_sim[0][idx]

    if score < 0.3:
        return "Sorry, I don't have information on that. Please consult a doctor."
    else:
        return f"Confidence: {score:.2f}\n\n{df.loc[idx, 'answer']}"

# --- Casual phrase handler ---
def get_response(user_input):
    user_input = user_input.lower().strip()

    if user_input in ["bye", "goodbye", "see you"]:
        return "Take care. Feel free to return if you have more questions."
    elif user_input in ["hi", "hello", "hey"]:
        return "Hello. You can ask me any health-related question."
    elif user_input in ["thanks", "thank you"]:
        return "You're welcome. Stay healthy."
    elif user_input in ["ama", "ask me anything"]:
        return "I'm ready. Ask me anything health-related."
    else:
        return get_medical_answer(user_input)

# --- Get sample questions from dataset ---
def get_sample_questions(df, n=6):
    return df['question'].sample(n).tolist()

# --- Streamlit UI ---
st.set_page_config(page_title="Healthcare Q&A Bot", page_icon="🩺")
st.title("Healthcare Q&A Bot")
st.write("Ask me any health-related question. I’ll try to help.")

# --- Example question block ---
sample_questions = get_sample_questions(df)
st.markdown("### 💡 Example Questions You Can Ask")
for q in sample_questions:
    if st.button(f"❓ {q.capitalize()}"):
        st.session_state['user_input'] = q

# --- Input box ---
user_input = st.text_input("Your question:", value=st.session_state.get('user_input', ''))

if user_input:
    response = get_response(user_input)
    st.markdown("### Bot Response:")
    st.write(response)
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 14px;
        color: #888;
    }
    </style>
    <div class="footer">🛠️ Developed with ❤️ by <b>Harshini</b></div>
""", unsafe_allow_html=True)
