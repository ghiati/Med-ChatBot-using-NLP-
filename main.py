import pandas as pd
from chatbot.advanced_chatbot import AdvancedChatbot
from chatbot.drug_info import get_drug_info 
from chatbot.preprocess_query import preprocess_query

def main():
    # Load the drug data
    drug_data = pd.read_csv('data/test_data.csv')  # Make sure this path is correct
    
    # Initialize the general chatbot
    general_chatbot = AdvancedChatbot(data_path='data/dialog_talk_agent.xlsx')
    
    # State to keep track of the last mentioned drug
    last_drug_name = None
    
    print("Welcome to the Pharmacy Chatbot. Ask your questions about drugs and prices.")
    
    while True:
        user_input = preprocess_query(input("You: "))
        
        # Check if the user wants to exit
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        
        # Get the drug info response
        response, last_drug_name = get_drug_info(user_input, drug_data, last_drug_name)
        
        # If the response is the default message, fall back to general chatbot
        if response.startswith("Sorry, I couldn't understand"):
            response = general_chatbot.get_response(user_input)
        
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
