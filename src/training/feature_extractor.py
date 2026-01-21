"""
Feature Extractor for Sentinel-AI

Extracts linguistic, semantic, and lexical features from text for model training.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional
import logging
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Extracts features from text data for content moderation training.
    """
    
    def __init__(self, use_embeddings: bool = True, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the feature extractor.
        
        Args:
            use_embeddings: Whether to use sentence embeddings (requires heavy download)
            embedding_model: Name of the sentence-transformer model to use
        """
        self.use_embeddings = use_embeddings
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )
        
        self.embedding_model = None
        if use_embeddings:
            logger.info(f"Loading embedding model: {embedding_model}...")
            # Lazy load in separate method or here? doing it here for simplicity
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}. Embeddings will be skipped.")
                self.use_embeddings = False
                
        self.is_fitted = False

    def fit_transform(self, texts: List[str]) -> pd.DataFrame:
        """
        Fit vectorizers and extract features for training data.
        
        Args:
            texts: List of text samples
            
        Returns:
            DataFrame with all features
        """
        logger.info(f"Fitting features for {len(texts)} samples...")
        
        # 1. Fit TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.is_fitted = True
        
        # Convert extracted features to DataFrame
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
        )
        # To save memory, we might want to reduce this or use sparse matrices
        # but for this implementation we'll keep it explicit.
        
        # 2. Extract other features
        other_features = self._batch_extract_basic_features(texts)
        features_df = pd.DataFrame(other_features)
        
        # 3. Embeddings
        if self.use_embeddings and self.embedding_model:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            embed_cols = [f"embed_{i}" for i in range(embeddings.shape[1])]
            embed_df = pd.DataFrame(embeddings, columns=embed_cols)
            # Concatenate
            final_df = pd.concat([features_df, embed_df], axis=1)
            # Note: We are NOT concatenating TF-IDF here to keep dimensions manageable
            # In a real pipeline, we'd probably use a Pipeline object or only top K TF-IDF dimensions
            # For this 'agentic' implementation, let's return combined non-sparse features 
            # and let the Trainer handle TF-IDF separately or allow user to choose.
            
            # Let's combine EVERYTHING for maximum information as requested
            # Warning: this might create a very wide DF
            final_df = pd.concat([final_df, tfidf_df], axis=1)
            
        else:
            final_df = pd.concat([features_df, tfidf_df], axis=1)
            
        return final_df

    def transform(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract features for inference (using fitted vectorizers).
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before calling transform()")
            
        # 1. TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
        )
        
        # 2. Basic Features
        other_features = self._batch_extract_basic_features(texts)
        features_df = pd.DataFrame(other_features)
        
        # 3. Embeddings
        if self.use_embeddings and self.embedding_model:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            embed_cols = [f"embed_{i}" for i in range(embeddings.shape[1])]
            embed_df = pd.DataFrame(embeddings, columns=embed_cols)
            final_df = pd.concat([features_df, embed_df, tfidf_df], axis=1)
        else:
            final_df = pd.concat([features_df, tfidf_df], axis=1)
            
        return final_df

    def _batch_extract_basic_features(self, texts: List[str]) -> List[Dict[str, float]]:
        """Extract linguistic and lexical features for a batch."""
        return [self._extract_single_features(text) for text in texts]

    def _extract_single_features(self, text: str) -> Dict[str, float]:
        """
        Extract detailed linguistic and lexical features for a single text.
        Similar to sentiment analysis features + specialized moderation metrics.
        """
        features = {}
        blob = TextBlob(text)
        
        # --- Linguistic Features ---
        features['polarity'] = blob.sentiment.polarity
        features['subjectivity'] = blob.sentiment.subjectivity
        
        # --- Lexical Features ---
        length = len(text)
        words = text.split()
        word_count = len(words)
        
        features['char_count'] = length
        features['word_count'] = word_count
        features['avg_word_length'] = length / word_count if word_count > 0 else 0
        
        # Aggression indicators
        caps_count = sum(1 for c in text if c.isupper())
        features['caps_ratio'] = caps_count / length if length > 0 else 0
        
        exclamation_count = text.count('!')
        features['exclamation_ratio'] = exclamation_count / length if length > 0 else 0
        
        question_count = text.count('?')
        features['question_ratio'] = question_count / length if length > 0 else 0
        
        # --- Multilingual Hints (Simple Heuristics) ---
        # Devanagari Unicode Range: \u0900-\u097F
        devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
        features['devanagari_ratio'] = devanagari_chars / length if length > 0 else 0
        
        return features
