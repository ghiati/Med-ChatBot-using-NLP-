import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from typing import List, Tuple

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    return re.sub(r'[^ a-z]', '', text)

def lemmatize_text(tokens: List[str]) -> List[str]:
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    for token, pos in pos_tag(tokens):
        if pos.startswith('V'):
            pos_val = 'v'
        elif pos.startswith('J'):
            pos_val = 'a'
        elif pos.startswith('R'):
            pos_val = 'r'
        else:
            pos_val = 'n'
        lemmatized_words.append(lemmatizer.lemmatize(token, pos_val))
    return lemmatized_words

def normalize_text(text: str) -> str:
    preprocessed_text = preprocess_text(text)
    tokens = nltk.word_tokenize(preprocessed_text)
    lemmatized_words = lemmatize_text(tokens)
    return " ".join(lemmatized_words)

def remove_stopwords(text: str) -> str:
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    return " ".join([token for token in tokens if token not in stop_words])

class AdvancedChatbot:
    def __init__(self, data_path: str):
        self.df = pd.read_excel(data_path)
        self.df.ffill(axis=0, inplace=True)
        self.df['processed_text'] = self.df['Context'].apply(normalize_text)
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['processed_text'])

    def get_response(self, user_input: str) -> str:
        processed_input = normalize_text(remove_stopwords(user_input))
        input_vector = self.vectorizer.transform([processed_input])
        similarity_scores = cosine_similarity(self.tfidf_matrix, input_vector)
        most_similar_index = similarity_scores.argmax()
        return self.df['Text Response'].iloc[most_similar_index]
