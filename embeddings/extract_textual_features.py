# import nltk
# ----------------------------------------------------------------------
# Below code is only to avoid the error while installing nltk resources
# ----------------------------------------------------------------------
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


# Installing additional resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.models import KeyedVectors
import numpy as np
import csv
from weighted_pooling import weighted_history_pooling

# Path to the pretrained Word2Vec model file
word2vec_model_path = '/Users/viru/Documents/GitHub/Micro-Influencer-Recommendations/pretrained-models/GoogleNews-vectors-negative300.bin.gz'

# Load the pretrained Word2Vec model
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)

def preprocess_caption(caption):
    # Lowercase the caption
    caption = caption.lower()
    
    # Tokenize the caption into words
    words = word_tokenize(caption)
    
    # Remove punctuation and special characters
    words = [word for word in words if word.isalnum()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return words


# Function to extract textual features from a caption
def extract_textual_features(captions):

    all_textual_features = []

    for caption in captions:
        # preproccess the caption before getting the embeddings
        words = preprocess_caption(caption)
        
        # Initialize an empty list to store word embeddings
        embedding = []
        
        # Iterate over each word in the caption
        for word in words:
            # Check if the word is in the vocabulary of the Word2Vec model
            if word in word2vec_model:
                # If the word is in the vocabulary, add its embedding to the list
                embedding.append(word2vec_model[word])
        
        # If no embeddings were found for any word in the caption, return None
        if not embedding:
            return None
        
        # Compute the average of word embeddings to get the textual features for the caption
        textual_features = np.mean(embedding, axis=0)

        all_textual_features.append(textual_features)
    
    return all_textual_features

# list of all captions
captions_list = [
    "Stepping into the weekend like a fashion icon 💃✨ Embracing the latest trends and letting my style shine bright! #FashionForward #WeekendVibes",
    "Living the luxe life in the city ✨ From chic street style to glamorous nights out, every moment is a fashion statement! #CityLife #Glamorous",
    "Sunday brunching in style with my squad 🥂✨ Bringing together fashion, friends, and fabulous moments! #BrunchGoals #Fashionista",
    "Exploring new horizons with a touch of elegance 💼🌟 From city streets to exotic destinations, every journey is a fashion adventure! #Wanderlust #FashionExplorer",
    "From runway to real life, embracing my unique style 💁‍♀️✨ Mixing trends, textures, and colors to create my own fashion story! #Fashionista #StyleInspo"
]

# Extracting the textual features
textual_features = extract_textual_features(captions_list)

# Applying the Weighted History Pooling method to the textual features
et = weighted_history_pooling(textual_features, 0.333)


# Save textual features to CSV
with open('embeddings/extracted_features_files/textual_features.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(et)