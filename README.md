<div align="center">

  <a href="https://tarkptel.github.io/">
    <img src="https://img.shields.io/badge/🌐-Portfolio-blue" height="28">
  </a>
  <a href="https://www.kaggle.com/tark01/">
    <img src="https://img.shields.io/badge/-Kaggle-20BEFF?logo=kaggle&logoColor=white" height="28">
  </a>
  <a href="https://www.linkedin.com/in/tark-patel/">
    <img src="https://img.shields.io/badge/-LinkedIn-0077B5?logo=linkedin&logoColor=white" height="28">
  </a>

</div>


#  Quora Duplicate Question Detection

This project uses a machine learning model to detect whether two Quora questions are duplicates. It involves feature engineering, preprocessing, and model training to classify question pairs.

---

## 🚀 Live Demo

You can try the web app live here: **[🔗 Hugging Face Space](https://huggingface.co/spaces/tarkpatel/duplicate-question-detector)**

---

## 📂 Dataset

The dataset used is the [Quora Question Pairs](https://www.kaggle.com/competitions/quora-question-pairs/data), which contains over 400,000 question pairs.

---

## 🔍 Features Engineered
- I Preprocess Question 1 & Question 2 Column.
Custom features were created to capture the similarity between questions:

- **`q1_len`** – Number of characters in Question 1  
- **`q2_len`** – Number of characters in Question 2  
- **`q1_num_words`** – Number of words in Question 1  
- **`q2_num_words`** – Number of words in Question 2  
- **`word_common`** – Count of common words in both questions  
- **`word_total`** – Total unique words across both questions  
- **`word_share`** – Ratio of shared words to total unique words  
- **`cwc_min`** – Common word count divided by minimum of word counts  
- **`cwc_max`** – Common word count divided by maximum of word counts  
- **`csc_min`** – Ratio of common stopwords to minimum stopword count  
- **`csc_max`** – Ratio of common stopwords to maximum stopword count  
- **`ctc_min`** – Common token count to minimum token count  
- **`ctc_max`** – Common token count to maximum token count  
- **`last_word_eq`** – Whether the last words of both questions match  
- **`first_word_eq`** – Whether the first words of both questions match  
- **`abs_len_diff`** – Absolute difference in length between questions  
- **`mean_len`** – Average length of both questions  
- **`longest_substr_ratio`** – Ratio of longest common substring length to minimum question length  
- **`fuzz_ratio`** – Fuzzy string match ratio  
- **`fuzz_partial_ratio`** – Partial fuzzy match score  
- **`token_sort_ratio`** – Fuzzy match after sorting tokens  
- **`token_set_ratio`** – Fuzzy match using token sets  

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas / NumPy
- NLTK
- FuzzyWuzzy
- Streamlit (for Web App)
- Hugging Face Spaces (for deployment)

---

## Tark Patel 😎
