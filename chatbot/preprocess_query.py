import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import spacy
import pandas as pd

# Load SpaCy model for named entity recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def get_wordnet_pos(treebank_tag):
    """Convert Treebank POS tags to WordNet POS tags for lemmatization."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_query(text):
    # Convert to lowercase and remove stopwords
    text = text.lower()
    stop = set(stopwords.words('english'))
    tokens = [word for word in word_tokenize(text) if word not in stop]

    # POS tagging and lemmatization
    pos_tagged = pos_tag(tokens)
    lema = nltk.WordNetLemmatizer()

    lema_words = [lema.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tagged]
    cleaned_phrase = ' '.join(lema_words)
    
    return cleaned_phrase

def extract_entities(query):
    """Extract named entities from the query using SpaCy's NER."""
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'PERSON']]  # Focus on products (drugs)
    
    # If no entities are found, fall back to keywords extraction
    if not entities:
        entities = query.split()
    
    return entities

def match_drug_name(entities, drug_list):
    """Match entities to drugs in the list by counting matching words and return the drugs with the highest count."""
    drug_scores = {}
    
    for drug in drug_list:
        # Tokenize the drug name and convert to lowercase
        drug_words = drug.lower().split()
        match_count = 0
        
        # Count how many words from the entities match with the drug name
        for entity in entities:
            entity_words = entity.lower().split()
            for word in entity_words:
                if word in drug_words:
                    match_count += 1
        
        # Store the match count for this drug
        drug_scores[drug] = match_count
    
    # Find the highest match count
    max_match = max(drug_scores.values())
    
    # Return drugs that have the highest match count
    best_matches = [drug for drug, score in drug_scores.items() if score == max_match]
    
    return best_matches

# # Load drug dataset
# df = pd.read_csv("/home/mg/nlpchatbot/data/test_data.csv")

# # Main loop to interact with the user
# while True:
#     drug_query = input("medicine (type 'exit' to quit): ")
#     if drug_query.lower() == 'exit':
#         break
    
#     # Preprocess user input
#     cleaned_input = preprocess_query(drug_query)
    
#     # Extract entities (drug names or important keywords)
#     entities = extract_entities(cleaned_input)
    
#     # Match extracted entities with drug names
#     drug_matches = match_drug_name(entities, df['Name'].tolist())
    
#     # Output results
#     print("Drugs name: ", drug_matches if drug_matches else "No drugs found")
