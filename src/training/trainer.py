"""
Model Trainer for Sentinel-AI

Handles training, evaluation, and persistence of content moderation models.
Uses XGBoost with MultiOutputClassifier for multi-label classification.
"""

import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
import xgboost as xgb
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentModerationTrainer:
    """
    Trainer for identifying policy violations.
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize trainer.
        
        Args:
            model_type: Type of model to use (currently only 'xgboost')
        """
        self.model_type = model_type
        self.model = None
        self.labels_list = None
        
    def train(self, X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2):
        """
        Train the model.
        
        Args:
            X: Feature DataFrame
            y: Label DataFrame (multi-hot encoded)
            test_size: Fraction of data to use for validation
        
        Returns:
            Dict containing evaluation metrics
        """
        logger.info(f"Starting training with {len(X)} samples. Model: {self.model_type}")
        self.labels_list = y.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Initialize model
        if self.model_type == 'xgboost':
            # Create a MultiOutputClassifier with XGBoost
            # Using basic parameters for now
            base_estimator = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                n_jobs=-1
            )
            self.model = MultiOutputClassifier(base_estimator)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        # Train
        logger.info("Fitting model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        logger.info("Evaluating model...")
        predictions = self.model.predict(X_test)
        
        metrics = self._calculate_metrics(y_test, predictions)
        logger.info(f"Training completed. Accuracy (Subset match): {metrics['accuracy']:.4f}")
        
        return metrics

    def _calculate_metrics(self, y_true: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate detailed metrics for multi-label classification."""
        
        # Subset accuracy (exact match of all labels) can be harsh, but useful
        subset_accuracy = accuracy_score(y_true, y_pred)
        
        # Hamming loss (fraction of wrong labels)
        h_loss = hamming_loss(y_true, y_pred)
        
        # Per-class report
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.labels_list, 
            output_dict=True,
            zero_division=0
        )
        
        return {
            "accuracy": subset_accuracy,
            "hamming_loss": h_loss,
            "classification_report": report
        }

    def save_model(self, output_dir: str):
        """Save model and metadata."""
        if not self.model:
            raise ValueError("Model not trained yet.")
            
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model object
        model_path = os.path.join(output_dir, f"sentinel_model_{self.model_type}_{timestamp}.joblib")
        joblib.dump(self.model, model_path)
        
        # Save feature/label metadata
        metadata = {
            "model_type": self.model_type,
            "labels": self.labels_list,
            "timestamp": timestamp,
            "path": model_path
        }
        meta_path = os.path.join(output_dir, "model_metadata.json")
        joblib.dump(metadata, meta_path) # Using joblib for simplicity/consistency
        
        logger.info(f"Model saved to {model_path}")
        return model_path

    def load_model(self, model_path: str):
        """Load a trained model."""
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
