"""
Data Loader for LMSYS-Chat-1M Dataset

Handles loading, preprocessing, and label extraction from the Huggingface dataset.
"""

import json
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import datasets
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingSample:
    """A single training sample with text and labels."""
    content: str
    language: str
    labels: Dict[str, float]  # Labels and their severity/confidence (0.0 to 1.0)
    source_id: str

class LMSYSDataLoader:
    """
    Loader for LMSYS-Chat-1M dataset from Huggingface.
    """
    
    DATASET_NAME = "lmsys/lmsys-chat-1m"
    
    # Mapping OpenAI moderation categories to Sentinel-AI policies
    CATEGORY_MAPPING = {
        "hate": "hate_speech",
        "hate/threatening": "hate_speech",
        "harassment": "harassment",
        "harassment/threatening": "harassment",
        "self-harm": "self_harm",
        "sexual": "adult_content",
        "sexual/minors": "adult_content",
        "violence": "graphic_violence",
        "violence/graphic": "graphic_violence"
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the loader.
        
        Args:
            cache_dir: Optional directory to cache the dataset
        """
        self.cache_dir = cache_dir
        self.dataset = None

    def load_dataset(self, split: str = "train", sample_size: Optional[int] = None):
        """
        Load the dataset from Huggingface.
        
        Args:
            split: Dataset split to load
            sample_size: Optional number of samples to load (streaming mode)
        """
        logger.info(f"Loading dataset {self.DATASET_NAME}...")
        
        try:
            # Load in streaming mode if sample size is small to save bandwidth
            streaming = sample_size is not None and sample_size < 10000
            
            self.dataset = datasets.load_dataset(
                self.DATASET_NAME, 
                split=split,
                cache_dir=self.cache_dir,
                streaming=streaming
            )
            
            if sample_size:
                if streaming:
                    self.dataset = self.dataset.take(sample_size)
                else:
                    self.dataset = self.dataset.select(range(sample_size))
                    
            logger.info(f"Dataset loaded. Type: {type(self.dataset)}")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def process_data(self, languages: List[str] = ["en", "hi"]) -> List[TrainingSample]:
        """
        Process the loaded dataset and extract training samples.
        
        Args:
            languages: List of language codes to filter by (e.g. ['en', 'hi'])
            
        Returns:
            List of TrainingSample objects
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        samples = []
        skipped_count = 0
        
        logger.info(f"Processing samples for languages: {languages}")
        
        # Iterate through dataset
        iterator = self.dataset
        if not isinstance(self.dataset, datasets.IterableDataset):
            iterator = tqdm(self.dataset)
            
        for row in iterator:
            try:
                # 1. Filter by language
                # LMSYS dataset has 'language' field (e.g. 'English', 'Hindi')
                # We need to normalize or check logic. The dataset description says 'detected language tag'.
                # Let's be permissive and check if our target langs are in the row language string.
                row_lang = str(row.get("language", "")).lower()
                
                # Check if it matches any of our target languages
                # 'en' matches 'english', 'hi' matches 'hindi'
                target_lang = None
                for lang in languages:
                    full_lang_name = "english" if lang == "en" else "hindi" if lang == "hi" else lang
                    if lang in row_lang or full_lang_name in row_lang:
                        target_lang = lang
                        break
                        
                if not target_lang:
                    continue
                
                # 2. Extract conversation
                # conversation is a list of dicts: [{"role": "user", "content": "..."}, ...]
                conversation = row.get("conversation", [])
                if not conversation:
                    continue
                
                # 3. Extract moderation labels
                # openai_moderation is also a list, usually corresponding to messages?
                # Or it might be a single object. 
                # Re-reading dataset docs: "included the OpenAI moderation API output for each message"
                openai_mod = row.get("openai_moderation", [])
                
                # Find user messages and their corresponding moderation tags
                for i, msg in enumerate(conversation):
                    if msg.get("role") != "user":
                        continue
                        
                    content = msg.get("content", "")
                    if not content or len(content) < 5:  # Skip very short content
                        continue
                        
                    # Get corresponding moderation result if available
                    # Assuming alignment by index or structure
                    labels = {}
                    if i < len(openai_mod):
                        mod_result = openai_mod[i]
                        
                        # Handle potential different structures of mod_result
                        # It usually has 'categories' (bool) and 'category_scores' (float)
                        if isinstance(mod_result, dict):
                            categories = mod_result.get("categories", {})
                            scores = mod_result.get("category_scores", {})
                            
                            for cat, is_flagged in categories.items():
                                if is_flagged and cat in self.CATEGORY_MAPPING:
                                    mapped_cat = self.CATEGORY_MAPPING[cat]
                                    # Use the score if available, otherwise 1.0 for flagged
                                    score = scores.get(cat, 1.0)
                                    # specific logic: keep max score if mapping conflicts (e.g. hate vs hate/threatening)
                                    labels[mapped_cat] = max(labels.get(mapped_cat, 0.0), score)
                    
                    # Add to samples
                    samples.append(TrainingSample(
                        content=content,
                        language=target_lang,
                        labels=labels,
                        source_id=f"{row.get('conversation_id', 'unknown')}_msg_{i}"
                    ))
                    
            except Exception as e:
                skipped_count += 1
                continue
                
        logger.info(f"Processed {len(samples)} samples. Skipped {skipped_count} errors.")
        return samples
