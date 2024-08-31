import pandas as pd
import time
import re
from preprocess_query import preprocess_query, extract_drug_names
from chat_ansewer import AdvancedChatbot

# Load the data from the CSV file
df = pd.read_csv('/home/mg/nlpchatbot/data/test_data.csv')
last_drug_name = None  # Global variable to store last mentioned drug

# Initialize the general conversational chatbot
general_chatbot = AdvancedChatbot('/home/mg/nlpchatbot/data/dialog_talk_agent.xlsx')

def detect_query_type(text):
    text = text.lower()
    price_keywords = ['price', 'cost', 'how much', 'expensive', 'cheap', 'afford']
    description_keywords = [
        'description', 'what is', "what's", 'tell me about', 'explain',
        'info', 'information', 'details', 'describe',
        'elaborate on', 'give me details about', 'specify',
        'clarify', 'elucidate', 'expound on', 'characterize',
        'define', 'outline', 'summarize', 'brief me on',
        'provide information on', 'what do you know about',
        'can you tell me about', 'what are the details of',
        'give an overview of', 'what does it mean',
        'what is the purpose of', 'how would you describe'
    ]
    if any(word in text for word in price_keywords):
        return 'price'
    elif any(word in text for word in description_keywords):
        return 'description'
    else:
        return 'general'

def process_request(text):
    global last_drug_name
    query_type = detect_query_type(text)
    substances = extract_drug_names(text, df['Name'].tolist())
    print(f"[DEBUG] Extracted substances: {substances}")  # Debug information
    
    if substances:
        last_drug_name = substances[0].lower()
    elif last_drug_name:
        substances = [last_drug_name]
    
    print(f"[DEBUG] Using drug name: {last_drug_name}")  # Debug information
    
    if substances and query_type != 'general':
        drug_name = substances[0]
        # Make the search case-insensitive
        drug_info = df[df['Name'].str.lower() == drug_name.lower()]
        print(f"[DEBUG] Drug info: {drug_info}")  # Debug information
        
        if not drug_info.empty:
            row = drug_info.iloc[0]
            if query_type == 'price':
                return f"The international price of {row['Name']} is {row.get('International Price', 'Not available')}."
            elif query_type == 'description':
                return f"{row['Description']}"
        else:
            return f"Sorry, I couldn't find information about {drug_name}. Please check the spelling or try another drug name."
    
    elif query_type == 'general':
        return general_chatbot.get_response(text)
    
    return "Sorry, I couldn't understand your request. Please ask about a specific drug's description, price, or ask a general question."

def test_description_and_price_request():
    global last_drug_name
    last_drug_name = None
    print("Welcome to the Drug Info and General Chatbot!")
    print("Ask me about drug description, price, or anything else. Type 'exit' to stop.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Process and clean the user input using the preprocess_query function
        processed_input = preprocess_query(user_input)
        start_time = time.time()
        response = process_request(processed_input)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Bot: {response}")
        print(f"Time taken: {elapsed_time:.4f} seconds")

if __name__ == "__main__":
    test_description_and_price_request()
