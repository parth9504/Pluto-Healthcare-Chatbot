import streamlit as st
import time
import pandas as pd
import numpy as np
import nltk
from collections import Counter as ctr
import string
import re
import spacy
from nltk.corpus import stopwords
import pickle
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib



nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Get base directory (where app runs)
BASE_DIR = os.getcwd()

# Paths
PICKLE_DIR = os.path.join(BASE_DIR, "Pickle Files")
DATASET_DIR = os.path.join(BASE_DIR, "Datasets")

# Load the voting classifier
with open(os.path.join(PICKLE_DIR, "voting_classifier.pkl"), "rb") as f:
    voting_classifier = pickle.load(f)

# Load the label encoder
with open(os.path.join(PICKLE_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

# Load the TF-IDF vectorizer
with open(os.path.join(PICKLE_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# Load datasets
df_medicine = pd.read_csv(os.path.join(DATASET_DIR, "Medicine_Details.csv"))
df_details = pd.read_csv(os.path.join(DATASET_DIR, "Disease_Description.csv"))

def preprocess_text(text):
    text = text.lower()
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)

    contractions = {
        "dont": "do not", "arent": "are not", "isnt": "is not", "doesnt": "does not",
        "wont": "will not", "cant": "can not", "couldnt": "could not", "shouldnt": "should not",
        "wouldnt": "would not", "havent": "have not", "hasnt": "has not", "hadnt": "had not",
        "didnt": "did not", "mustnt": "must not", "im": "i am", "ill": "i will",
        "ive": "i have", "id": "i would", "youre": "you are", "youve": "you have",
        "youll": "you will", "youd": "you would", "hes": "he is", "hell": "he will",
        "hed": "he would", "shes": "she is", "shell": "she will", "shed": "she would",
        "its": "it is", "itll": "it will", "were": "we are", "weve": "we have",
        "well": "we will", "wed": "we would", "theyre": "they are", "theyve": "they have",
        "theyll": "they will", "theyd": "they would", "thats": "that is", "thatll": "that will",
        "whos": "who is", "wholl": "who will", "whats": "what is", "whatll": "what will",
        "wheres": "where is", "wherell": "where will", "whens": "when is", "whenll": "when will",
        "whys": "why is", "whyll": "why will", "hows": "how is", "howll": "how will"
    }

    auxiliary_verbs = {"would", "have", "should"}
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    for verb in auxiliary_verbs:
        text = text.replace(verb, "")

    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(token) for token in tokens]
    text = ' '.join(tokens)
    text = re.sub(' +', ' ', text)
    return text

# Chat helpers
def append_user(prompt):
    with st.chat_message("User"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "User", "content": prompt})

def append_assistant(prompt):
    with st.chat_message("Assistant"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "Assistant", "content": prompt})

def bot_response(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.07)

def append_bot(response):
    with st.chat_message("Assistant"):
        st.write_stream(bot_response(response))
        st.session_state.messages.append({"role": "Assistant", "content": response})

def append_med_bot(response_df):
    response_str = response_df.to_markdown()
    with st.chat_message("Assistant"):
        st.markdown(response_str)
        time.sleep(0.07)
    st.session_state.messages.append({"role": "Assistant", "content": response_str})

def predict_disease(text):
    text_processed = preprocess_text(text)
    text_transformed = tfidf_vectorizer.transform([text_processed])
    predicted_label_encoded = voting_classifier.predict(text_transformed)
    predicted_label = label_encoder.inverse_transform(predicted_label_encoded)
    append_bot("You are probably suffering from:  " + predicted_label[0])
    row = df_details[df_details['Disease'] == predicted_label[0]]
    if not row.empty:
        description = row['Description'].values[0]
        symp = row['Symptoms'].values[0]
        treatment = row['Treatment'].values[0]
        append_bot("**About the disease:**  \n  " + description)
        append_bot("**Symptoms:**  \n" + symp)
        append_bot("**Treatment:**  \n" + treatment)

def get_wikipedia_page_url(topic):
    try:
        search_results = wikipedia.search(topic)
        append_bot("Please help me choose which of the following suits your query:")
        selected_result = st.selectbox("Choose a search result:", search_results[:5], index=0)
        append_user(selected_result)
        page = wikipedia.page(selected_result)
        return page.url
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Ambiguous search term. Did you mean: {', '.join(e.options)}?"
    except wikipedia.exceptions.PageError:
        return "Page not found on Wikipedia."

def load_models(vectorizer_path, model_path):
    tfidf_vectorizer = joblib.load(os.path.join(PICKLE_DIR, vectorizer_path))
    nn_model = joblib.load(os.path.join(PICKLE_DIR, model_path))
    return tfidf_vectorizer, nn_model

def recommend_medicines_by_text(input_text, tfidf_vectorizer, nn_model, df_medicine):
    input_vector = tfidf_vectorizer.transform([input_text])
    distances, indices = nn_model.kneighbors(input_vector)
    recommended_medicines_df = df_medicine.iloc[indices[0]][[
        'Medicine Name', 'Composition', 'Uses', 'Side_effects']]
    return recommended_medicines_df

def main():
    st.title("Pluto: Healthcare Chatbot 🤖💊🏥")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_topic" not in st.session_state:
        st.session_state.selected_topic = ""
    if "section_headings" not in st.session_state:
        st.session_state.section_headings = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.write("Assistant: Hello! How may I help you today?")
    action = st.selectbox("Choose an option:", [
        "Choose", "Analyse Symptoms", "Access Information about diseases", "Medicinal Info"], index=0)

    if action == "Analyse Symptoms":
        append_user(action)
        if symptoms := st.chat_input("Enter your symptoms:"):
            append_assistant("Enter your symptoms")
            append_user(symptoms)
            predict_disease(symptoms)
        else:
            st.write("Assistant: Please provide your symptoms")

    elif action == "Access Information about diseases":
        append_user(action)
        with st.form(key='disease_info_form'):
            topic = st.text_input("Enter the disease you want to find information about:")
            if (len(topic) != 0):
                append_user(topic)
            if st.form_submit_button("Search"):
                result = get_wikipedia_page_url(topic)
                if isinstance(result, str):
                    append_bot("This is what I found, hope this helps 😊")
                    append_bot(result)
                else:
                    st.write(f"Visit the Wikipedia page: [{topic}]({result})")

    elif action == "Medicinal Info":
        append_user(action)
        if symptoms := st.chat_input("Enter disease:"):
            append_assistant("Enter your symptoms")
            append_user(symptoms)
            loaded_tfidf_vectorizer, loaded_nn_model = load_models(
                'tfidf_med_vectorizer.pkl', 'nn_model.pkl')
            recommended_medicines_df = recommend_medicines_by_text(
                symptoms, loaded_tfidf_vectorizer, loaded_nn_model, df_medicine)
            append_med_bot(recommended_medicines_df)
            append_assistant("⚠️ Please consult a physician before using any suggested medicines")
        else:
            st.write("Assistant: Please provide your symptoms")

if __name__ == "__main__":
    main()


