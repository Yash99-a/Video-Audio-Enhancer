import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import langdetect
from typing import Dict, List, Optional

class MultilingualTextCleaningModel:
    def __init__(self, languages: List[str] = ['en', 'es', 'fr', 'de', 'mr', 'hi', 'ja', 'zh']):
        self.languages = languages
        self.models: Dict[str, Dict] = {}
        
        # Initialize models for each language
        for lang in languages:
            self.models[lang] = {
                'vectorizer': TfidfVectorizer(max_features=5000),
                'classifier': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500),
                'is_trained': False
            }

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text"""
        try:
            detected_lang = langdetect.detect(text)
            return detected_lang if detected_lang in self.languages else 'en'
        except:
            return 'en'

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text"""
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Ensure that spaces between words are properly maintained
        text = text.strip()
        return text

    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks while preserving special characters like '!' and '?'"""
        # Replace multiple punctuation marks with a single one
        text = re.sub(r'!+', '!', text)
        text = re.sub(r'\?+', '?', text)
        text = re.sub(r'\.+', '.', text)
        
        # Add space after punctuation if not present (preserve ? and ! as they are)
        text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)
        return text

    def _clean_asian_text(self, text: str) -> str:
        """Clean Asian languages (Chinese, Japanese)"""
        # Remove extra spaces between Asian characters
        text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
        # Normalize Asian punctuation
        text = text.replace('。', '。 ')
        text = text.replace('、', '、 ')
        text = text.replace('！', '！ ')
        text = text.replace('？', '？ ')
        return text.strip()

    def _clean_indic_text(self, text: str) -> str:
        """Clean Indic languages (Hindi, Marathi)"""
        # Ensure spaces between Devanagari characters are preserved
        # Remove extra spaces between Devanagari characters (but ensure proper word separation)
        text = re.sub(r'([\u0900-\u097F])\s+([\u0900-\u097F])', r'\1 \2', text)
        
        # Normalize Devanagari punctuation
        text = text.replace('।', '। ')  # Add space after punctuation if needed
        return text.strip()

    def process_transcription(self, text: str, language: Optional[str] = None) -> str:
        """Process transcribed text in the specified or detected language"""
        if not text:
            return ""

        # Detect language
        lang = language if language in self.languages else self.detect_language(text)

        # Initial cleaning steps
        text = self._normalize_whitespace(text)
        text = self._normalize_punctuation(text)

        # Language-specific cleaning
        if lang in ['zh', 'ja']:
            text = self._clean_asian_text(text)
        elif lang in ['hi', 'mr']:
            text = self._clean_indic_text(text)

        # Capitalize first letter of each sentence for languages that use Latin script
        if lang in ['en', 'es', 'fr', 'de']:
            sentences = text.split('. ')
            text = '. '.join(s.strip().capitalize() for s in sentences if s.strip())
        # Handle sentence-ending punctuation for Indic languages (Hindi, Marathi)
        elif lang in ['hi', 'mr']:
            if not text.endswith('।'):
                text += '।'

        # Final normalization
        text = self._normalize_whitespace(text)
        
        # Add appropriate ending punctuation based on language
        if text:
            if lang in ['ja', 'zh']:
                if not text[-1] in '。！？':
                    text += '。'
            elif lang in ['hi', 'mr']:
                if not text[-1] in '।':
                    text += '।'
            elif not text[-1] in '.!?':
                text += '.'

        return text

    def train(self, dirty_texts: List[str], clean_texts: List[str], language: str):
        """Train the model for a specific language"""
        if language not in self.languages:
            raise ValueError(f"Unsupported language: {language}")
            
        try:
            # Prepare data
            dirty_texts = [self.process_transcription(text, language) for text in dirty_texts]
            clean_texts = [self.process_transcription(text, language) for text in clean_texts]
            
            # Update vectorizer and train classifier
            all_texts = dirty_texts + clean_texts
            X = self.models[language]['vectorizer'].fit_transform(all_texts)
            y = np.array([0] * len(dirty_texts) + [1] * len(clean_texts))
            
            self.models[language]['classifier'].fit(X, y)
            self.models[language]['is_trained'] = True
            print(f"Model trained successfully for {language}")
            
        except Exception as e:
            print(f"Training failed for {language}: {str(e)}")
            self.models[language]['is_trained'] = False

def main():
    # Test the multilingual model
    model = MultilingualTextCleaningModel()
    
    # Test with different languages
    test_texts = {
        'en': "ThIs   is   an EXampLE  of a  mESSy text!!!",
        'es': "ESte   es   un   EJemplo   DE texto   DESORDENADO!!!",
        'mr': "हे   एक   गोंधळलेले   वाक्य   आहे!!!",
        'hi': "यह   एक   अव्यवस्थित   वाक्य   है!!!",
        'ja': "これは  めちゃくちゃな  テキスト  です！！！",
        'zh': "这是    一个    乱七八糟    的    文本！！！"
    }
    
    for lang, text in test_texts.items():
        cleaned_text = model.process_transcription(text, lang)
        print(f"\nLanguage: {lang}")
        print(f"Original text: {text}")
        print(f"Cleaned text: {cleaned_text}")

if __name__ == "__main__":
    main()
