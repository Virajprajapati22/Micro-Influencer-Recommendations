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

import re
import string
from typing import List, Union

import numpy as np
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load the pretrained Word2Vec model
word2vec_model_path = '/Users/viru/Documents/GitHub/Micro-Influencer-Recommendations/pretrained-models/GoogleNews-vectors-negative300.bin.gz'
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)

def preprocess_caption(caption: str) -> List[str]:
    """Preprocess a caption string."""
    # Remove emojis
    caption = remove_emojis(caption)
    # Remove URLs
    caption = re.sub(r'http\S+', '', caption)
    # Convert to lowercase
    caption = caption.lower()
    # Remove punctuation
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the caption
    tokens = word_tokenize(caption)
    # Remove stopwords, single character tokens, and numeric tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1 and not token.isnumeric()]
    return tokens

def remove_emojis(text: str) -> str:
    """Remove emojis from a text."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def extract_textual_features(captions: List[str]) -> Union[List[np.ndarray], None]:
    """Extract textual features from captions."""
    all_textual_features = []
    for caption in captions:
        words = preprocess_caption(caption)
        # Initialize an empty list to store word embeddings
        embeddings = []
        for word in words:
            # Check if the word is in the vocabulary of the Word2Vec model
            if word in word2vec_model:
                embeddings.append(word2vec_model[word])
        # If no embeddings were found for any word in the caption, return None
        if not embeddings:
            return None
        # Compute the average of word embeddings to get the textual features for the caption
        textual_features = np.mean(embeddings, axis=0)
        all_textual_features.append(textual_features)
    return all_textual_features