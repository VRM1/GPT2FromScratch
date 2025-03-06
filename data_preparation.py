"""
Data preparation for multitask GPT-2 training
"""

import os
import argparse
import random
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm

class MultitaskDataset(Dataset):
    """Dataset for training GPT-2 on multiple tasks with zero-shot formatting."""
    
    def __init__(self, tokenizer, max_length=1024, cache_dir=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load translation dataset
        print("Loading translation dataset...")
        translation_dataset = load_dataset("wmt14", "fr-en", split="train", cache_dir=cache_dir)
        
        # Load sentiment dataset
        print("Loading sentiment dataset...")
        sentiment_dataset = load_dataset("sst2", split="train", cache_dir=cache_dir)
        
        # Process translation examples
        print("Processing translation examples...")
        translation_examples = []
        for example in tqdm(translation_dataset.select(range(min(200000, len(translation_dataset)))), desc="Translation"):
            fr_text = example['translation']['fr']
            en_text = example['translation']['en']
            
            # Format with prompt
            text = f"French: {fr_text} English: {en_text}"
            
            # Tokenize and append
            translation_examples.append(text)
        
        # Process sentiment examples
        print("Processing sentiment examples...")
        sentiment_examples = []
        for example in tqdm(sentiment_dataset, desc="Sentiment"):
            text = example['sentence']
            label = "Positive" if example['label'] == 1 else "Negative"
            
            # Format with prompt
            formatted_text = f"Review: {text} Sentiment: {label}"
            
            # Tokenize and append
            sentiment_examples.append(formatted_text)
        
        # Balance datasets
        min_size = max(len(translation_examples), len(sentiment_examples))
        translation_examples = translation_examples[:min_size]
        sentiment_examples = sentiment_examples[:min_size]
        
        print(f"Using {min_size} examples from each dataset")
        
        # Combine and shuffle examples
        all_examples = translation_examples + sentiment_examples
        random.shuffle(all_examples)
        
        # Tokenize all examples
        print("Tokenizing all examples...")
        for text in tqdm(all_examples, desc="Tokenizing"):
            tokenized = tokenizer.encode(text)
            
            # Skip examples that are too long
            if len(tokenized) <= max_length:
                self.examples.append(tokenized)
            
        print(f"Final dataset has {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


class EvalTranslationDataset(Dataset):
    """Dataset for evaluating GPT-2 on translation task."""
    
    def __init__(self, tokenizer, split="validation", max_length=1024, cache_dir=None, num_examples=1000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.targets = []
        
        # Load translation dataset
        print(f"Loading translation {split} dataset...")
        translation_dataset = load_dataset("wmt14", "fr-en", split=split, cache_dir=cache_dir)
        
        # Select subset for evaluation
        translation_dataset = translation_dataset.select(range(min(num_examples, len(translation_dataset))))
        
        # Process examples
        print("Processing translation examples...")
        for example in tqdm(translation_dataset, desc="Translation Eval"):
            fr_text = example['translation']['fr']
            en_text = example['translation']['en']
            
            # Format with prompt
            input_text = f"French: {fr_text} English:"
            
            # Tokenize
            tokenized_input = tokenizer.encode(input_text)
            
            # Skip if too long
            if len(tokenized_input) < max_length - 10:  # Leave room for generation
                self.inputs.append(tokenized_input)
                self.targets.append(en_text)
        
        print(f"Prepared {len(self.inputs)} evaluation examples")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.inputs[idx], dtype=torch.long),
            "target_text": self.targets[idx]
        }


class EvalSentimentDataset(Dataset):
    """Dataset for evaluating GPT-2 on sentiment task."""
    
    def __init__(self, tokenizer, split="validation", max_length=1024, cache_dir=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.targets = []
        
        # Load sentiment dataset
        print(f"Loading sentiment {split} dataset...")
        sentiment_dataset = load_dataset("sst2", split=split, cache_dir=cache_dir)
        
        # Process examples
        print("Processing sentiment examples...")
        for example in tqdm(sentiment_dataset, desc="Sentiment Eval"):
            text = example['sentence']
            label = "Positive" if example['label'] == 1 else "Negative"
            
            # Format with prompt
            input_text = f"Review: {text} Sentiment:"
            
            # Tokenize
            tokenized_input = tokenizer.encode(input_text)
            
            # Skip if too long
            if len(tokenized_input) < max_length - 10:  # Leave room for generation
                self.inputs.append(tokenized_input)
                self.targets.append(label)
        
        print(f"Prepared {len(self.inputs)} evaluation examples")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.inputs[idx], dtype=torch.long),
            "target_text": self.targets[idx]
        }


def prepare_datasets(tokenizer_name="gpt2", cache_dir=None, output_dir="data"):
    """Prepare and save the datasets for training and evaluation."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_dataset = MultitaskDataset(tokenizer, cache_dir=cache_dir)
    eval_translation = EvalTranslationDataset(tokenizer, cache_dir=cache_dir)
    eval_sentiment = EvalSentimentDataset(tokenizer, cache_dir=cache_dir)
    
    # Save tokenizer for future use
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    
    return train_dataset, eval_translation, eval_sentiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for GPT-2 multitask training")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer to use")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for datasets")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    
    args = parser.parse_args()
    
    prepare_datasets(args.tokenizer, args.cache_dir, args.output_dir)
