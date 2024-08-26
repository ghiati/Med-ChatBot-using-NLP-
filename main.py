import pandas as pd
df = pd.read_csv('/home/mg/nlpchatbot/data/test_data.csv')

def detect_description_request(text):
    keywords = ['description', 'what is', 'tell me about', 'explain']
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in keywords):
        for index, row in df.iterrows():
            drug_name = row['Name'].lower()
            if drug_name in text_lower:
                return f"The description of {row['Name']} is: {row['Description']}"
    
    return "Sorry, I couldn't understand your request. Please ask about a drug's description."

# Test the function
def test_description_request():
    print("Welcome to the Drug Description Bot!")
    print("Ask me about the description of a drug. Type 'exit' to stop.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = detect_description_request(user_input)
        print(f"Bot: {response}")


if __name__ == "__main__":
    test_description_request()
