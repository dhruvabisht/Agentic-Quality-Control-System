"""
Training Pipeline for Sentinel-AI

CLI script to run the end-to-end training process:
1. Load LMSYS dataset
2. Extract features
3. Train XGBoost model
4. Save model and metrics
"""

import sys
import os
import argparse
import logging
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.data_loader import LMSYSDataLoader
from src.training.feature_extractor import FeatureExtractor
from src.training.trainer import ContentModerationTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Sentinel-AI Training Pipeline")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of samples to use (default: 1000)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size")
    parser.add_argument("--model-type", type=str, default="xgboost", help="Model type to train")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--use-embeddings", action="store_true", help="Enable sentence embeddings (slower)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory for dataset")
    
    args = parser.parse_args()
    
    logger.info("Starting training pipeline...")
    logger.info(f"Sample Size: {args.sample_size}")
    logger.info(f"Use Embeddings: {args.use_embeddings}")
    
    # 1. Load Data
    try:
        loader = LMSYSDataLoader(cache_dir=args.cache_dir)
        loader.load_dataset(split="train", sample_size=args.sample_size)
        samples = loader.process_data(languages=["en", "hi"])
        
        if not samples:
            logger.error("No samples found. Exiting.")
            sys.exit(1)
            
        logger.info(f"Loaded {len(samples)} valid training samples.")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
        
    # 2. Prepare Feature Data
    texts = [s.content for s in samples]
    
    # Process Labels (Multi-label)
    # Convert dict labels to list of active labels for Binarizer
    # We treat any label with score > 0.5 as "present" for binary classification
    # Or just simply 'if key exists' depending on loader logic. 
    # Loader logic: labels[mapped_cat] = score. 
    # Let's say threshold is 0.1 to catch everything for training
    label_lists = []
    for s in samples:
        active_labels = [k for k, v in s.labels.items() if v > 0.0]
        # If clean, maybe add 'clean'? Or just empty list implies clean.
        # Strategically for multi-label, empty list = clean.
        label_lists.append(active_labels)
        
    # Binarize labels
    mlb = MultiLabelBinarizer()
    y_matrix = mlb.fit_transform(label_lists)
    y_df = pd.DataFrame(y_matrix, columns=mlb.classes_)
    logger.info(f"Label classes: {mlb.classes_}")
    
    # 3. Extract Features
    extractor = FeatureExtractor(use_embeddings=args.use_embeddings)
    try:
        X_df = extractor.fit_transform(texts)
        logger.info(f"Extracted features shape: {X_df.shape}")
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        sys.exit(1)
    
    # 4. Train Model
    trainer = ContentModerationTrainer(model_type=args.model_type)
    try:
        metrics = trainer.train(X_df, y_df, test_size=args.test_size)
        
        # Print key metrics
        print("\n=== Training Results ===")
        print(f"Subset Accuracy: {metrics['accuracy']:.4f}")
        print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
        
    # 5. Save Model
    try:
        trainer.save_model(args.output_dir)
        # Also save the label encoder logic? The trainer saves 'labels_list'.
        # We might need to save MultiLabelBinarizer if we want to reverse transform easily.
        # For now, relying on model metadata is okay.
    except Exception as e:
        logger.error(f"Model saving failed: {e}")
        
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
