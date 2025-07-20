"""
Text preprocessing utilities
"""
import re
import string
from typing import List, Union
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """Text preprocessing for ticket data"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Custom stop words for support tickets
        self.custom_stop_words = {
            'please', 'help', 'need', 'want', 'thanks', 'thank',
            'hi', 'hello', 'regards', 'sincerely', 'best'
        }
        self.stop_words.update(self.custom_stop_words)
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        return word_tokenize(text)
    
        def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stop words from token list"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str, remove_stop=True, lemmatize=True) -> str:
        """Full preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stop words if requested
        if remove_stop:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize if requested
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        # Join tokens back
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str], **kwargs) -> List[str]:
        """Preprocess multiple texts"""
        return [self.preprocess(text, **kwargs) for text in texts]
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract top keywords from text"""
        # Preprocess
        processed = self.preprocess(text)
        tokens = self.tokenize(processed)
        
        # Count word frequency
        word_freq = {}
        for token in tokens:
            if len(token) > 2:  # Skip very short words
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N keywords
        return [word for word, freq in sorted_words[:top_n]]
    
    def get_text_features(self, text: str) -> dict:
        """Extract various text features for analysis"""
        tokens = self.tokenize(text)
        
        features = {
            'length': len(text),
            'word_count': len(tokens),
            'avg_word_length': sum(len(token) for token in tokens) / len(tokens) if tokens else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'has_url': bool(re.search(r'http\S+|www.\S+', text)),
            'has_email': bool(re.search(r'\S+@\S+', text))
        }
        
        return features