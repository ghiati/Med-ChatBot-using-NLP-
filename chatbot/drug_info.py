import pandas as pd
from chatbot.preprocess_query import extract_drug_names
def detect_query_type(text):
    text = text.lower()
    price_keywords = ['price', 'cost', 'how much', 'expensive', 'cheap', 'afford']
    description_keywords = ['description', 'what is', "what's", 'info', 'information', 'details', 'describe']
    
    if any(word in text for word in price_keywords):
        return 'price'
    elif any(word in text for word in description_keywords):
        return 'description'
    else:
        return 'unknown'

def get_drug_info(text, df, last_drug_name):
    query_type = detect_query_type(text)
    substances = extract_drug_names(text, df['Name'].tolist())

    if substances:
        last_drug_name = substances[0].lower()
    elif last_drug_name:
        substances = [last_drug_name]

    if substances:
        drug_name = substances[0]
        drug_info = df[df['Name'].str.lower() == drug_name.lower()]
        
        if not drug_info.empty:
            row = drug_info.iloc[0]
            if query_type == 'price':
                return f"The international price of {row['Name']} is {row.get('International Price', 'Not available')}."
            elif query_type == 'description':
                return f"{row['Description']}"
        else:
            return f"Sorry, I couldn't find information about {drug_name}. Please check the spelling or try another drug name."

    return "Sorry, I couldn't understand your request. Please ask about a specific drug's description or price."
