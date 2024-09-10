import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from fuzzywuzzy import process
import spacy
from numpy import dot
from numpy.linalg import norm

from chatbot.preprocess_query import extract_entities, match_drug_name, preprocess_query

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Constants
PRICE_KEYWORDS = ['price', 'cost', 'how much', 'expensive', 'cheap', 'afford', 'pay', 'worth', 'value', 'budget', 'economical', 'rate', 'fee', 'charge', 'price tag']
DESCRIPTION_KEYWORDS = ['description', 'what is', "what's", 'info', 'information', 'details', 'describe', 'tell me about', 'explain', 'elaborate', 'specify', 'characterize', 'define', 'clarify', 'elucidate']

def preprocess_text(text):
    """Preprocess the input text."""
    text = text.lower()
    tokens = word_tokenize(text)
    # stop_words = set(stopwords.words('english')) - set(DESCRIPTION_KEYWORDS)
    tokens = [token for token in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def correct_input(text, keyword_list):
    """Correct potential input mistakes."""
    words = text.split()
    corrected_words = []
    for word in words:
        closest_match, score = process.extractOne(word, keyword_list)
        if score >= 80:
            corrected_words.append(closest_match)
        else:
            corrected_words.append(word)
    return ' '.join(corrected_words)

def detect_query_type(text):
    """Detect the type of query (price, description, or unknown)."""
    text = preprocess_text(text)
    text = correct_input(text, PRICE_KEYWORDS + DESCRIPTION_KEYWORDS)
    
    doc = nlp(text)
    
    # Check for money-related entities
    if any(ent.label_ == 'MONEY' for ent in doc.ents):
        return 'price'
    
    # Check for specific sentence structures
    for token in doc:
        if token.dep_ == 'ROOT' and token.lemma_ in ['cost', 'price', 'pay']:
            return 'price'
        if token.dep_ == 'ROOT' and token.lemma_ in ['describe', 'explain', 'mean']:
            return 'description'
    
    # Keyword matching
    if any(keyword in text for keyword in PRICE_KEYWORDS):
        return 'price'
    elif any(keyword in text for keyword in DESCRIPTION_KEYWORDS):
        return 'description'
    
    # # Word embedding similarity
    # text_vector = nlp(text).vector
    # price_vector = nlp(' '.join(PRICE_KEYWORDS)).vector
    # description_vector = nlp(' '.join(DESCRIPTION_KEYWORDS)).vector
    
    # # Check if vectors are non-empty before calculating cosine similarity
    # print("text vector ",text_vector.size , "  price vector : ",price_vector.size , " descr : ", description_vector.size)
    # if text_vector.size > 0 and price_vector.size > 1 and description_vector.size > 12:
    #     cos_sim_price = dot(text_vector, price_vector) / (norm(text_vector) * norm(price_vector))
    #     cos_sim_description = dot(text_vector, description_vector) / (norm(text_vector) * norm(description_vector))
        
    #     if cos_sim_price > cos_sim_description:
    #         return 'price'
    #     elif cos_sim_description > cos_sim_price:
    #         return 'description'
    
    return 'unknown'
def get_drug_info(text, df, last_drug_name):
    """Get drug information based on the user's query."""
    query_type = detect_query_type(text)
    cleaned_input = preprocess_query(text)
    entities = extract_entities(cleaned_input)
    substances = match_drug_name(entities, df['Name'].tolist())

    if substances:
        last_drug_name = substances[0].lower()
    elif last_drug_name:
        substances = [last_drug_name]

    response = "Sorry, I couldn't understand your request. Please ask about a specific drug's description or price."
    drug_name = None

    if substances:
        drug_name = substances[0]
        drug_info = df[df['Name'].str.lower() == drug_name.lower()]
        
        if not drug_info.empty:
            row = drug_info.iloc[0]
            if query_type == 'price':
                response = f"The international price of {row['Name']} is {row.get('International Price', 'Not available')}."
            elif query_type == 'description':
                response = f"{row['Description']}"
        else:
            response = f"Sorry, I couldn't find information about {drug_name}. Please check the spelling or try another drug name."
    
    return response, drug_name

# # Main execution (commented out for module usage)
# if __name__ == "__main__":
#     df = pd.read_csv("/home/mg/nlpchatbot/data/test_data.csv")
#     while True:
#         drug = input("test: ")
#         print("answer:", get_drug_info(drug, df, last_drug_name=None),
#               "drug name:", match_drug_name(extract_entities(preprocess_query(drug)), df['Name'].tolist()))