import pandas as pd
from chatbot.preprocess_query import preprocess_query
from chatbot.drug_info import get_drug_info
from chatbot.advanced_chatbot import AdvancedChatbot

def main():
    drug_data = pd.read_csv('data/test_data.csv')
    general_chatbot = AdvancedChatbot('data/dialog_talk_agent.xlsx')
    last_drug_name = None

    print("Welcome to the Drug Info and General Chatbot!")
    print("Ask me about drug description, price, or general questions. Type 'exit' to stop.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        processed_input = preprocess_query(user_input)
        drug_response = get_drug_info(processed_input, drug_data, last_drug_name)

        if "couldn't understand" in drug_response:
            general_response = general_chatbot.get_response(user_input)
            print(f"Bot: {general_response}")
        else:
            print(f"Bot: {drug_response}")

if __name__ == "__main__":
    main()
