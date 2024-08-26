import pandas as pd
import time
import re

# Load the data from the CSV file
df = pd.read_csv('/home/mg/nlpchatbot/data/test_data.csv')

def detect_query_type(text):
    text = text.lower()
    if any(word in text for word in ['price', 'cost', 'how much']):
        return 'price'
    elif any(word in text for word in ['description', 'describe', 'what is']):
        return 'description'
    else:
        return 'unknown'

def process_request(text):
    query_type = detect_query_type(text)
    substances = potential_substance_chunk_detection2(text)
    
    for substance in substances:
        drug_name = substance.lower()
        drug_info = df[df['Name'].str.lower() == drug_name]
        
        if not drug_info.empty:
            row = drug_info.iloc[0]
            
            if query_type == 'price':
                return f"The international price of {row['Name']} is {row.get('International Price', 'Not available')}."
            elif query_type == 'description':
                return f"{row['Description']}"
    
    return "Sorry, I couldn't understand your request. Please ask about a drug's description or price."

def potential_substance_chunk_detection2(text):
    text = text.title()
    words = re.findall(r'\b[A-Za-z]+\b', text)
    return [word for word in words if word.lower() in df['Name'].str.lower().values]

def test_description_and_price_request():
    print("Welcome to the Drug Info Bot!")
    print("Ask me about drug description or price. Type 'exit' to stop.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        start_time = time.time()  # Record start time
        response = process_request(user_input)
        end_time = time.time()  # Record end time
        
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Bot: {response}")
        print(f"Time taken: {elapsed_time:.4f} seconds")  # Print elapsed time

if __name__ == "__main__":
    test_description_and_price_request()
