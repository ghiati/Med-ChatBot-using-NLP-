import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import numpy as np

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Text normalization and processing class
class TextProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def text_normalization(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tags_list = pos_tag(tokens)
        lema_words = []
        for token, pos_token in tags_list:
            pos_val = self.get_pos_tag(pos_token)
            lema_token = self.lemmatizer.lemmatize(token, pos_val)
            lema_words.append(lema_token)
        return " ".join(lema_words)

    def get_pos_tag(self, pos_token):
        if pos_token.startswith('V'):
            return 'v'
        elif pos_token.startswith('J'):
            return 'a'
        elif pos_token.startswith('R'):
            return 'r'
        else:
            return 'n'

    def remove_stopwords(self, text):
        tokens = nltk.word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return " ".join(filtered_tokens)

# Chatbot class with text processing and response matching
class AdvancedChatbot:
    def __init__(self, data_path):
        self.df = pd.read_excel(data_path)
        self.df.ffill(axis=0, inplace=True)
        self.text_processor = TextProcessor()
        
        # Text normalization for context
        self.df['lemmatized_text'] = self.df['Context'].apply(self.text_processor.text_normalization)
        
        # Initialize TfidfVectorizer
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2))
        self.df_tfidf = self.tfidf.fit_transform(self.df['lemmatized_text'])

    def get_response(self, user_input):
        processed_input = self.text_processor.remove_stopwords(user_input)
        normalized_input = self.text_processor.text_normalization(processed_input)
        input_tfidf = self.tfidf.transform([normalized_input])
        
        # Calculate cosine similarity and find the best match
        cosine_similarities = cosine_similarity(input_tfidf, self.df_tfidf).flatten()
        best_match_index = cosine_similarities.argmax()
        
        # Check if the similarity is above a threshold
        if cosine_similarities[best_match_index] > 0.3:
            return self.df['Text Response'].iloc[best_match_index]
        else:
            return "I'm sorry, I don't have enough information to answer that question. Could you please rephrase or provide more context?"

    def get_top_responses(self, user_input, top_n=3):
        processed_input = self.text_processor.remove_stopwords(user_input)
        normalized_input = self.text_processor.text_normalization(processed_input)
        input_tfidf = self.tfidf.transform([normalized_input])
        
        cosine_similarities = cosine_similarity(input_tfidf, self.df_tfidf).flatten()
        top_indices = np.argsort(cosine_similarities)[-top_n:][::-1]
        
        top_responses = []
        for idx in top_indices:
            if cosine_similarities[idx] > 0.3:
                top_responses.append((self.df['Text Response'].iloc[idx], cosine_similarities[idx]))
        
        return top_responses

# # Running the chatbot
# if __name__ == "__main__":
#     chatbot = AdvancedChatbot('/home/mg/nlpchatbot/data/dialog_talk_agent.xlsx')
#     print("Chatbot initialized. Type 'exit' to end the conversation.")
    
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == 'exit':
#             print("Chatbot: Goodbye!")
#             break
        
#         top_responses = chatbot.get_top_responses(user_input)
        
#         if top_responses:
#             best_response = top_responses[0][0]  # Select the response with the highest similarity
#             print(f"Chatbot: {best_response}")
#         else:
#             print("Chatbot: I'm sorry, I don't have enough information to answer that question. Could you please rephrase or provide more context?")
