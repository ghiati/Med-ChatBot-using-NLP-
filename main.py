import pandas as pd
import time
import re
from preprocess_query import preprocess_query, extract_drug_names

# Load the data from the CSV file
df = pd.read_csv('/home/mg/nlpchatbot/data/test_data.csv')

def detect_query_type(text):
    text = text.lower()
    price_keywords = ['price', 'cost', 'how much', 'expensive', 'cheap', 'afford']
    description_keywords = [
        'description', 'what is', 'tell me about', 'explain',
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
        return 'unknown'

def process_request(text):
    query_type = detect_query_type(text)
    substances = extract_drug_names(text, df['Name'].tolist())
    
    for substance in substances:
        drug_name = substance.lower()
        drug_info = df[df['Name'].str.lower() == drug_name]
        
        if not drug_info.empty:
            row = drug_info.iloc[0]
            if query_type == 'price':
                return f"The international price of {row['Name']} is {row.get('International Price', 'Not available')}."
            elif query_type == 'description':
                return f"{row['Description']}"
    
    return "Sorry, I couldn't understand your request. Please ask about a specific drug's description or price."

def test_description_and_price_request():
    print("Welcome to the Drug Info Bot!")
    print("Ask me about drug description or price. Type 'exit' to stop.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Process and clean the user input using the new preprocess_query function
        processed_input = preprocess_query(user_input)
        
        start_time = time.time()
        response = process_request(processed_input)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print(f"Bot: {response}")
        print(f"Time taken: {elapsed_time:.4f} seconds")

if __name__ == "__main__":
    test_description_and_price_request()