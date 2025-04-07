import streamlit as st
import numpy as np
import pandas as pd
import pickle
import distance
import re
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

STOP_WORDS = set(['a', 'an', 'the', 'is', 'are', 'was', 'were', 'am', 'and', 'or', 'not', 'to', 'in', 'on', 'of', 'that'])
SAFE_DIV = 0.0001
MAX_LEN = 259

# Define contractions dictionary once
contractions = { 
    "ain't": "am not", "aren't": "are not", "can't": "can not", "can't've": "can not have", "'cause": "because",
    "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have",
    "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "i'd": "i would",
    "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not",
    "it's": "it is", "let's": "let us", "ma'am": "madam", "might've": "might have", "mightn't": "might not",
    "must've": "must have", "mustn't": "must not", "needn't": "need not", "o'clock": "of the clock", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is", "should've": "should have", "shouldn't": "should not",
    "so've": "so have", "that's": "that is", "there's": "there is", "they'd": "they would", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what's": "what is", "where's": "where is", "who's": "who is", "won't": "will not",
    "would've": "would have", "wouldn't": "would not", "y'all": "you all", "you'd": "you would", "you'll": "you will",
    "you're": "you are", "you've": "you have"
}

def preprocess(q):
    q = str(q).lower().strip()
    q = q.replace('%', ' percent').replace('$', ' dollar ').replace('₹', ' rupee ').replace('€', ' euro ').replace('@', ' at ')
    q = q.replace('[math]', '')
    q = q.replace(',000,000,000', 'b').replace(',000,000', 'm').replace(',000', 'k')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    q_decontracted = [contractions[word] if word in contractions else word for word in q.split()]
    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have").replace("n't", " not").replace("'re", " are").replace("'ll", " will")

    q = BeautifulSoup(q, "html.parser").get_text()
    q = re.sub(r'\W', ' ', q).strip()
    return q

def generate_features(s1, s2):
    row = pd.DataFrame({'s1': [s1], 's2': [s2]})
    row['s1_len'] = row['s1'].str.len()
    row['s2_len'] = row['s2'].str.len()
    row['q1_num_words'] = row['s1'].apply(lambda x: len(x.split()))
    row['q2_num_words'] = row['s2'].apply(lambda x: len(x.split()))

    row['word_common'] = row.apply(lambda r: len(set(r['s1'].split()) & set(r['s2'].split())), axis=1)
    row['word_total'] = row.apply(lambda r: len(set(r['s1'].split()) | set(r['s2'].split())), axis=1)
    row['word_share'] = row['word_common'] / (row['word_total'] + SAFE_DIV)

    token_feats = row.apply(lambda r: [
        len(set(r['s1'].split()) & set(r['s2'].split())) / (min(len(set(r['s1'].split())), len(set(r['s2'].split()))) + SAFE_DIV),
        len(set(r['s1'].split()) & set(r['s2'].split())) / (max(len(set(r['s1'].split())), len(set(r['s2'].split()))) + SAFE_DIV),
        int(r['s1'].split()[-1] == r['s2'].split()[-1]),
        int(r['s1'].split()[0] == r['s2'].split()[0])
    ], axis=1, result_type="expand")
    row[['cwc_min', 'cwc_max', 'last_word_eq', 'first_word_eq']] = token_feats

    len_feats = row.apply(lambda r: [
        abs(len(r['s1'].split()) - len(r['s2'].split())),
        (len(r['s1'].split()) + len(r['s2'].split())) / 2,
        len(list(distance.lcsubstrings(r['s1'], r['s2']))[0]) / (min(len(r['s1']), len(r['s2'])) + 1)
    ], axis=1, result_type="expand")
    row[['abs_len_diff', 'mean_len', 'longest_substr_ratio']] = len_feats

    fuzz_feats = row.apply(lambda r: [
        fuzz.QRatio(r['s1'], r['s2']),
        fuzz.partial_ratio(r['s1'], r['s2']),
        fuzz.token_sort_ratio(r['s1'], r['s2']),
        fuzz.token_set_ratio(r['s1'], r['s2']),
    ], axis=1, result_type="expand")
    row[['fuzz_ratio', 'fuzz_partial_ratio', 'token_sort_ratio', 'token_set_ratio']] = fuzz_feats

    return row

def tokenize_pad(s1, s2):
    s1_seq = tokenizer.texts_to_sequences([s1])
    s2_seq = tokenizer.texts_to_sequences([s2])
    s1_pad = pad_sequences(s1_seq, maxlen=MAX_LEN, padding='post')
    s2_pad = pad_sequences(s2_seq, maxlen=MAX_LEN, padding='post')
    return np.hstack((s1_pad, s2_pad))

def predict_duplicate(q1, q2):
    q1 = preprocess(q1)
    q2 = preprocess(q2)
    df = generate_features(q1, q2)
    X_text = tokenize_pad(q1, q2)
    X_feat = df.drop(['s1', 's2'], axis=1).values
    final_input = np.hstack((X_text, X_feat)).astype('float32')
    prob = model.predict(final_input)[0][0]  # float between 0 and 1
    label = 1 if prob >= 0.5 else 0
    return label, prob
    

st.title("Duplicate Question Detector")

q1 = st.text_input("Enter question 1:")
q2 = st.text_input("Enter question 2:")

if st.button("Check Duplicate"):
    if not q1 or not q2:
        st.warning("Please enter both questions.")
    else:
        try:
            label, prob = predict_duplicate(q1, q2)
            confidence = int(prob * 100) if label == 1 else int((1 - prob) * 100)

            if label == 1:
                st.success(f"✅ Duplicate (confidence = {confidence}%)")
            else:
                st.info(f"❌ Not Duplicate (confidence = {confidence}%)")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
