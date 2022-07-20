import pandas as pd
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

pos_tags = {"J": wordnet.ADJ, "N": wordnet.NOUN, 
                "V": wordnet.VERB, "R": wordnet.ADV}

class Preprocessing:
        
    def __init__(self):
        super(Preprocessing, self).__init__()

    @staticmethod
    def load_data(config):
        """
        Loading data from CSV file
        """
        data = pd.read_csv(config.data_path)
        if 'Unnamed: 0'in data.columns:
            data = data.drop(['Unnamed: 0'], axis=1)
        return data

    @staticmethod
    def index_to_label(index):
        """
        converting 1 sentiment to positive and 0 sentiment to negative 
        """
        return 'positive' if index == 1 else 'negative'

    @staticmethod
    def label_to_index(label):
        """
        converting positive sentiment to 1 and negative sentiment to 0 
        """
        return 1 if label == 'positive' else 0

    @staticmethod
    def clean_text(text_string) -> str:
        """
        converting text to lowercase 
        removing all hashtags, stop_words, extra space, digits
        """
        process_text = text_string.lower()  # Convert string to lower
        process_text = re.sub(r"(@\S+)", "", process_text)  # Remove hashtags
        process_text = re.sub("<.*?>", " ", process_text) 
        process_text = ''.join([i for i in process_text if not i.isdigit()]).strip()  # Removing all digits only
        process_text = process_text.split(" ")  
        process_text = " ".join([word for word in process_text if word not in stopwords.words("english")]) # Removing all stop words
        process_text = " ".join(process_text.split())  # Removing extra spaces via splitting and joining
        return process_text

    @staticmethod
    def lemmatizing(text):
        """
        Removing punctuations and Lemmatizing Text
        """
        tokens = []
        lemmatizer = WordNetLemmatizer()
        unigrams = word_tokenize(text) # input text to unigrams
        for word in unigrams:
            pos_tag = nltk.pos_tag([word])[0][1][0].upper()
            word = lemmatizer.lemmatize(word, pos_tags.get(pos_tag, wordnet.NOUN)) # Lemmatizing all unigrams
            # removing all punctuations, empty unigrams
            if word not in string.punctuation and re.match(r'[\w+]', word):
                tokens.append(word)
        process_text = " ".join(tokens)      
        return process_text