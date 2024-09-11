import pandas as pd
from chatbot.preprocess_query import preprocess_query, extract_entities, match_drug_name
from chatbot.drug_info import get_drug_info, detect_query_type
from chatbot.advanced_chatbot import AdvancedChatbot

def main():
    # Load the drug dataset
    df = pd.read_csv("/home/mg/nlpchatbot/data/test_data.csv")
    
    # Initialize the AdvancedChatbot
    advanced_chatbot = AdvancedChatbot("/home/mg/nlpchatbot/data/dialog_talk_agent.xlsx")  # Replace with the actual path
    
    print("Welcome to the Drug Information Chatbot!")
    print("You can ask about drug prices, descriptions, or any other questions.")
    print("Type 'exit' to quit the chatbot.")
    
    last_drug_name = None
    
    while True:
        user_input = input("\nUser: ").strip()
        
        if user_input.lower() == 'exit':
            print("Thank you for using the Drug Information Chatbot. Goodbye!")
            break
        
        # Detect query type
        query_type = detect_query_type(user_input)
        print("detected query type : ",query_type)
        
        # If it's a follow-up question about price and we have a last_drug_name
        if query_type == 'price' and last_drug_name and not any(drug.lower() in user_input.lower() for drug in df['Name']):
            response, _ = get_drug_info(f"{last_drug_name} {user_input}", df, last_drug_name)
        else:
            # First, try to get drug-specific information
            response, drug_name = get_drug_info(user_input, df, last_drug_name)
            
            # If a specific drug was found, update last_drug_name
            if drug_name:
                last_drug_name = drug_name
            
            # If no specific drug info was found, use the advanced chatbot
            if response.startswith("Sorry, I couldn't"):
                response = advanced_chatbot.get_response(user_input)
        
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()