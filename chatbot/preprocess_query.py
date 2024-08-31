import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_query(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens]
    corrected_tokens = [str(TextBlob(word).correct()) for word in filtered_tokens]
    cleaned_phrase = ' '.join(corrected_tokens)
    return cleaned_phrase

def extract_drug_names(cleaned_phrase, drug_list):
    found_drugs = [drug for drug in drug_list if drug.lower() in cleaned_phrase]
    return found_drugs
