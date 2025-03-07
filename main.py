"""
GPT-2 Training Script for Multitask Learning
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import logging
import time
import random
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score

from transformer.gpt2 import GPT2
from data_preparation import MultitaskDataset, EvalTranslationDataset, EvalSentimentDataset

# Set up logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def collate_batch(batch):
    """Collate function for DataLoader that handles variable length sequences."""
    pad_id = 50256  # Default GPT-2 pad_id (same as eos_id)
    
    # Pad sequences to the same length
    padded_batch = pad_sequence([item for item in batch], batch_first=True, padding_value=pad_id)
    
    return padded_batch


def collate_eval(batch):
    """Collate function for evaluation DataLoader."""
    input_ids = [item["input_ids"] for item in batch]
    target_texts = [item["target_text"] for item in batch]
    
    # Pad input_ids
    pad_id = 50256  # Default GPT-2 pad_id
    padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    
    return {
        "input_ids": padded_inputs,
        "target_texts": target_texts
    }


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """Create a schedule with linear learning rate warmup followed by linear decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load checkpoint to resume training."""
    logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Get the step we're resuming from
        start_step = checkpoint.get('step', 0)
        
        # Get loss if available
        loss = checkpoint.get('loss', None)
        if loss is not None:
            logger.info(f"Resuming from step {start_step} with loss {loss:.4f}")
        else:
            logger.info(f"Resuming from step {start_step}")
            
        return start_step
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        logger.warning("Starting training from scratch...")
        return 0


def train(model, train_dataloader, optimizer, scheduler, config, device, writer, start_step=0):
    """Train the model."""
    model.train()
    
    global_step = start_step
    total_loss = 0.0
    
    # Calculate number of steps per epoch and total steps
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * config['training']['epochs']
    
    # Calculate starting epoch
    start_epoch = global_step // steps_per_epoch
    
    progress_bar = tqdm(range(total_steps - global_step))
    
    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        # Skip steps we've already processed in the first epoch when resuming
        skip_steps = global_step % steps_per_epoch if epoch == start_epoch else 0
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps already processed in the current epoch
            if step < skip_steps:
                continue
                
            batch = batch.to(device)
            
            # Shift input and target for language modeling
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            # Forward pass
            logits = model(inputs)
            
            # Calculate loss (cross-entropy)
            loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
            
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            
            # Update learning rate schedule
            scheduler.step()
            
            total_loss += loss.item()
            epoch_loss += loss.item()
            
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1}/{config['training']['epochs']} - Loss: {loss.item():.4f}")
            
            # Logging
            if global_step % config['output']['log_interval'] == 0:
                avg_loss = total_loss / config['output']['log_interval']
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar('train/loss', avg_loss, global_step)
                writer.add_scalar('train/lr', lr, global_step)
                logger.info(f"Step {global_step} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")
                total_loss = 0.0
            
            # Evaluation & Save checkpoint
            if global_step % config['output']['eval_interval'] == 0:
                # Save model checkpoint
                save_model(model, optimizer, scheduler, global_step, loss.item(),
                           os.path.join(config['output']['output_dir'], f"checkpoint-{global_step}.pt"))
                
                # Run evaluation
                model.eval()
                logger.info("Running evaluation...")
                evaluate_tasks(model, config, device, writer, global_step)
                model.train()
        
        # Log epoch stats
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / (len(train_dataloader) - skip_steps if epoch == start_epoch else len(train_dataloader))
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']} completed in {epoch_time:.2f}s | Avg Loss: {avg_epoch_loss:.4f}")
    
    # Save final model
    save_model(model, optimizer, scheduler, global_step, avg_epoch_loss,
              os.path.join(config['output']['output_dir'], "final_model.pt"))
    
    return global_step


def evaluate_tasks(model, config, device, writer, global_step):
    """Evaluate the model on translation and sentiment tasks."""
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config['data']['tokenizer_path'])
    
    # Evaluate translation
    logger.info("Evaluating translation task...")
    translation_dataset = EvalTranslationDataset(
        tokenizer, 
        split="validation", 
        max_length=config['data']['max_seq_length'],
        num_examples=50  # Use smaller number for quick evaluation
    )
    
    translation_loader = DataLoader(
        translation_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_eval
    )
    
    bleu_scores = []
    example_count = 0
    
    for batch in tqdm(translation_loader, desc="Translation Eval"):
        input_ids = batch["input_ids"].to(device)
        target_texts = batch["target_texts"]
        
        # Generate translations
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_length=input_ids.size(1) + 50,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )
        
        # Decode the generated text
        generated_texts = [tokenizer.decode(ids[input_ids.size(1):], skip_special_tokens=True).strip() 
                          for ids in generated_ids]
        
        # Calculate BLEU score
        for gen_text, target_text in zip(generated_texts, target_texts):
            # Always log some examples regardless of BLEU score
            if example_count < 5:
                logger.info(f"Translation example:")
                logger.info(f"  Input: {tokenizer.decode(input_ids[example_count % len(input_ids)], skip_special_tokens=True)}")
                logger.info(f"  Generated: {gen_text}")
                logger.info(f"  Target: {target_text}")
                example_count += 1
            
            reference = [target_text.lower().split()]
            candidate = gen_text.lower().split()
            
            if len(candidate) > 0:  # Skip empty generations
                try:
                    bleu = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))  # BLEU-1 for simplicity
                    bleu_scores.append(bleu)
                    
                    # Log BLEU score for examples we've already printed
                    if example_count <= 5 and example_count > 0:
                        logger.info(f"  BLEU: {bleu:.4f}")
                except Exception as e:
                    logger.warning(f"Error calculating BLEU: {e}")
    
    # Log BLEU if available, otherwise log zeros
    if bleu_scores:
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        logger.info(f"Average BLEU score: {avg_bleu:.4f}")
        writer.add_scalar('eval/bleu', avg_bleu, global_step)
    else:
        logger.info("No valid BLEU scores calculated. Model may need more training.")
        writer.add_scalar('eval/bleu', 0.0, global_step)  # Log zero to track progress
    
    # Evaluate sentiment
    logger.info("Evaluating sentiment task...")
    sentiment_dataset = EvalSentimentDataset(
        tokenizer, 
        split="validation", 
        max_length=config['data']['max_seq_length']
    )
    
    sentiment_loader = DataLoader(
        sentiment_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_eval
    )
    
    predictions = []
    targets = []
    example_count = 0
    
    for batch in tqdm(sentiment_loader, desc="Sentiment Eval"):
        input_ids = batch["input_ids"].to(device)
        target_texts = batch["target_texts"]
        
        # Generate sentiment predictions
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_length=input_ids.size(1) + 10,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )
        
        # Decode the generated text
        generated_texts = [tokenizer.decode(ids[input_ids.size(1):], skip_special_tokens=True).strip() 
                          for ids in generated_ids]
        
        # Process predictions and targets
        for gen_text, target_text in zip(generated_texts, target_texts):
            # Always log some examples
            if example_count < 5:
                logger.info(f"Sentiment example:")
                logger.info(f"  Input: {tokenizer.decode(input_ids[example_count % len(input_ids)], skip_special_tokens=True)}")
                logger.info(f"  Generated: {gen_text}")
                logger.info(f"  Target: {target_text}")
                example_count += 1
            
            # Simple matching for positive/negative
            pred = None
            if "positive" in gen_text.lower():
                pred = "Positive"
            elif "negative" in gen_text.lower():
                pred = "Negative"
            
            if pred is not None:
                predictions.append(pred)
                targets.append(target_text)
                
                # Log prediction for examples we've already printed
                if example_count <= 5 and example_count > 0:
                    logger.info(f"  Prediction: {pred}")
    
    # Calculate accuracy if possible
    if predictions:
        accuracy = accuracy_score([1 if t == "Positive" else 0 for t in targets], 
                                 [1 if p == "Positive" else 0 for p in predictions])
        logger.info(f"Sentiment accuracy: {accuracy:.4f}")
        writer.add_scalar('eval/sentiment_accuracy', accuracy, global_step)
    else:
        logger.warning("No valid sentiment predictions - model may need more training")
        writer.add_scalar('eval/sentiment_accuracy', 0.0, global_step)


def save_model(model, optimizer, scheduler, step, loss, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'step': step,
        'loss': loss
    }, path)
    
    logger.info(f"Model saved to {path}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train GPT-2 for multitask learning")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prepare_data", action="store_true", help="Prepare datasets before training")
    parser.add_argument("--resume", action="store_true", help="Resume training (overrides config)")
    parser.add_argument("--resume_checkpoint", type=str, help="Checkpoint to resume from (overrides config)")
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Override resume settings if provided in command line
    if args.resume:
        config['training']['resume_training'] = True
    if args.resume_checkpoint:
        config['training']['resume_checkpoint'] = args.resume_checkpoint
    
    # Create output directories
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    os.makedirs(config['output']['tensorboard_dir'], exist_ok=True)
    
    # Set up tensorboard
    writer = SummaryWriter(config['output']['tensorboard_dir'])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config['data']['tokenizer_path'])
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare datasets if requested
    if args.prepare_data:
        from data_preparation import prepare_datasets
        logger.info("Preparing datasets...")
        train_dataset, _, _ = prepare_datasets(
            tokenizer_name=config['data']['tokenizer_path'],
            cache_dir=None,
            output_dir="data"
        )
    else:
        # Load dataset
        logger.info("Loading dataset...")
        train_dataset = MultitaskDataset(
            tokenizer=tokenizer,
            max_length=config['data']['max_seq_length']
        )
    
    # Prepare dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_batch
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    
    # Initialize model with attention configuration
    model = GPT2(
        vocab_size=config['model']['vocab_size'],
        pad_idx=tokenizer.pad_token_id,
        d_model=config['model']['n_embd'],
        n_layers=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        d_k=config['model']['n_embd'] // config['model']['n_head'],
        d_v=config['model']['n_embd'] // config['model']['n_head'],
        d_inner=config['model']['d_inner'],
        dropout=config['model']['dropout'],
        max_position_embeddings=config['model']['n_positions'],
        attention_type=config['model'].get('attention_type', 'full'),  # Default to full attention if not specified
        window_size=config['model'].get('window_size', 256)            # Default window size of 256 if not specified
    ).to(device)
    
    # Log attention configuration
    logger.info(f"Using attention type: {config['model'].get('attention_type', 'full')}")
    if config['model'].get('attention_type', 'full') in ['windowed', 'causal_windowed']:
        logger.info(f"Window size: {config['model'].get('window_size', 256)}")
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * config['training']['epochs']
    
    # Initialize scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Start step for training (0 by default)
    start_step = 0
    
    # Resume training if requested
    if config['training']['resume_training'] and config['training']['resume_checkpoint']:
        start_step = load_checkpoint(
            model, 
            optimizer, 
            scheduler, 
            config['training']['resume_checkpoint'],
            device
        )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model size: {total_params/1e6:.2f}M parameters")
    
    # Train the model with the starting step
    logger.info("Starting training...")
    train(model, train_dataloader, optimizer, scheduler, config, device, writer, start_step=start_step)
    
    logger.info("Training complete!")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    evaluate_tasks(model, config, device, writer, total_steps)
    
    writer.close()


if __name__ == "__main__":
    main()