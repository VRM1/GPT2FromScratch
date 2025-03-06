"""
Interactive testing script for GPT-2 model
"""

import os
import argparse
import yaml
import torch
from transformers import GPT2Tokenizer
import logging
import sys

from transformer.gpt2 import GPT2

# Set up logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path, config, device):
    """Load the trained model from checkpoint."""
    # Initialize model
    model = GPT2(
        vocab_size=config['model']['vocab_size'],
        pad_idx=None,  # We don't need padding for inference
        d_model=config['model']['n_embd'],
        n_layers=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        d_k=config['model']['n_embd'] // config['model']['n_head'],
        d_v=config['model']['n_embd'] // config['model']['n_head'],
        d_inner=config['model']['d_inner'],
        dropout=0.0,  # No dropout for inference
        max_position_embeddings=config['model']['n_positions']
    ).to(device)
    
    # Load weights
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def generate_text(model, tokenizer, prompt, config, device):
    """Generate text using the model."""
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_length=input_ids.size(1) + config['test']['max_length'],
            temperature=config['test']['temperature'],
            top_k=config['test']['top_k'],
            top_p=config['test']['top_p']
        )
    
    # Decode
    original_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the prompt)
    generated_text = full_text[len(original_prompt):]
    
    return original_prompt, generated_text


def interactive_mode(model, tokenizer, config, device):
    """Run interactive mode where user can input prompts."""
    print("\n" + "="*50)
    print("GPT-2 Interactive Testing Mode")
    print("Type 'quit', 'exit', or press Ctrl+C to end the session")
    print("="*50)
    
    while True:
        # Prompt user for input
        try:
            task = input("\nChoose task (1=Translation, 2=Sentiment, 3=Custom): ")
            
            if task.lower() in ['quit', 'exit']:
                break
                
            if task == '1':
                french_text = input("Enter French text: ")
                prompt = f"French: {french_text} English:"
            elif task == '2':
                review = input("Enter review text: ")
                prompt = f"Review: {review} Sentiment:"
            elif task == '3':
                prompt = input("Enter custom prompt: ")
            else:
                print("Invalid option. Please choose 1, 2, or 3.")
                continue
            
            if prompt.lower() in ['quit', 'exit']:
                break
                
            # Generate and display text
            original_prompt, generated_text = generate_text(
                model, tokenizer, prompt, config, device
            )
            
            print("\n" + "-"*50)
            print(f"Prompt: {original_prompt}")
            print(f"Generated: {generated_text}")
            print("-"*50)
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test GPT-2 model")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to config file")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint (overrides config)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override model path if provided
    if args.model_path:
        config['test']['model_path'] = args.model_path
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config['data']['tokenizer_path'])
    
    # Load model
    model = load_model(config['test']['model_path'], config, device)
    
    # Override mode if specified
    if args.interactive:
        config['test']['mode'] = 'interactive'
    
    # Run the requested mode
    if config['test']['mode'] == 'interactive':
        interactive_mode(model, tokenizer, config, device)
    else:
        logger.error(f"Unknown test mode: {config['test']['mode']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
