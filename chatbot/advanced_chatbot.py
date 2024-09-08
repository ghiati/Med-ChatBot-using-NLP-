import pandas as pd
import nltk
import re
from nltk.stem import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from nltk import pos_tag
from sklearn.metrics import pairwise_distances
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Text normalization and processing class
class TextProcessor:
    def __init__(self):
        self.lemmatizer = wordnet.WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def text_normalization(self, text):
        text = str(text).lower()  # Convert to lowercase
        text = re.sub(r'[^a-z ]', '', text)  # Remove special characters
        tokens = nltk.word_tokenize(text)  # Tokenize words
        tags_list = pos_tag(tokens)  # Part-of-speech tagging
        lema_words = []
        for token, pos_token in tags_list:
            pos_val = self.get_pos_tag(pos_token)
            lema_token = self.lemmatizer.lemmatize(token, pos_val)  # Lemmatize
            lema_words.append(lema_token)
        return " ".join(lema_words)

    def get_pos_tag(self, pos_token):
        if pos_token.startswith('V'):
            return 'v'  # Verb
        elif pos_token.startswith('J'):
            return 'a'  # Adjective
        elif pos_token.startswith('R'):
            return 'r'  # Adverb
        else:
            return 'n'  # Noun

    def remove_stopwords(self, text):
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return " ".join(filtered_tokens)


# Chatbot class with text processing and response matching
class AdvancedChatbot:
    def __init__(self, data_path):
        self.df = pd.read_excel(data_path)
        self.df.ffill(axis=0, inplace=True)  # Fill missing values
        self.text_processor = TextProcessor()  # Initialize text processor

        # Text normalization for context
        self.df['lemmatized_text'] = self.df['Context'].apply(self.text_processor.text_normalization)

        # Initialize CountVectorizer for Bag of Words
        self.cv = CountVectorizer()
        self.df_bow = self.cv.fit_transform(self.df['lemmatized_text']).toarray()  # BoW representation

    def get_response(self, user_input):
        processed_input = self.text_processor.remove_stopwords(user_input)
        normalized_input = self.text_processor.text_normalization(processed_input)
        input_bow = self.cv.transform([normalized_input]).toarray()

        # Calculate cosine similarity and find the best match
        cosine_value = 1 - pairwise_distances(self.df_bow, input_bow, metric='cosine')
        best_match_index = cosine_value.argmax()

        return self.df['Text Response'].iloc[best_match_index]


# Running the chatbot
if __name__ == "__main__":
    chatbot = AdvancedChatbot('/home/mg/nlpchatbot/data/dialog_talk_agent.xlsx')  # Initialize chatbot with the dataset

    print("Chatbot initialized. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = chatbot.get_response(user_input)
        print(f"Chatbot: {response}")
