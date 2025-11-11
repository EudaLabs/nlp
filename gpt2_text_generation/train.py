"""
Fine-tuning script for GPT-2 on custom text data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)
from typing import List, Optional
import logging
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Custom dataset for GPT-2 fine-tuning."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: GPT2Tokenizer,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.examples = []
        
        logger.info(f"Tokenizing {len(texts)} texts...")
        for text in tqdm(texts):
            # Tokenize and add EOS token
            tokens = tokenizer.encode(
                text + tokenizer.eos_token,
                max_length=max_length,
                truncation=True
            )
            self.examples.append(tokens)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


class GPT2Trainer:
    """Trainer for fine-tuning GPT-2 models."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing trainer with {model_name} on {self.device}")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        self.model.to(self.device)
    
    def prepare_data(
        self,
        train_texts: List[str],
        val_texts: Optional[List[str]] = None,
        max_length: int = 512
    ):
        """
        Prepare datasets for training.
        
        Args:
            train_texts: Training texts
            val_texts: Validation texts
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (train_dataset, val_dataset, data_collator)
        """
        train_dataset = TextDataset(train_texts, self.tokenizer, max_length)
        
        val_dataset = None
        if val_texts:
            val_dataset = TextDataset(val_texts, self.tokenizer, max_length)
        
        # Data collator for dynamic padding
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # GPT-2 uses causal LM, not masked LM
        )
        
        return train_dataset, val_dataset, data_collator
    
    def train(
        self,
        train_dataset: Dataset,
        data_collator,
        val_dataset: Optional[Dataset] = None,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        output_dir: str = "./gpt2_finetuned",
        save_steps: int = 1000,
        eval_steps: int = 500,
        logging_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0
    ):
        """
        Fine-tune GPT-2 model.
        
        Args:
            train_dataset: Training dataset
            data_collator: Data collator for batching
            val_dataset: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps for learning rate scheduler
            output_dir: Directory to save model
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            logging_steps: Log every N steps
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_collator
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=data_collator
            )
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        global_step = 0
        best_val_loss = float('inf')
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Total steps: {total_steps}")
        
        for epoch in range(epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item()
                
                # Update weights
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Logging
                    if global_step % logging_steps == 0:
                        avg_loss = epoch_loss / (step + 1)
                        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    
                    # Evaluation
                    if val_loader and global_step % eval_steps == 0:
                        val_loss = self.evaluate(val_loader)
                        logger.info(f"Step {global_step} - Validation loss: {val_loss:.4f}")
                        
                        # Save best model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.save_model(output_path / "best_model")
                            logger.info("Saved best model")
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if global_step % save_steps == 0:
                        self.save_model(output_path / f"checkpoint-{global_step}")
                        logger.info(f"Saved checkpoint at step {global_step}")
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} - Average loss: {avg_epoch_loss:.4f}")
            
            # Evaluate at end of epoch
            if val_loader:
                val_loss = self.evaluate(val_loader)
                logger.info(f"Epoch {epoch + 1} - Validation loss: {val_loss:.4f}")
        
        # Save final model
        self.save_model(output_path / "final_model")
        logger.info("Training completed!")
    
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def save_model(self, path: Path):
        """Save model and tokenizer."""
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load_model(self, path: Path):
        """Load model and tokenizer."""
        self.model = GPT2LMHeadModel.from_pretrained(path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(path)
        self.model.to(self.device)


def load_text_file(file_path: str) -> List[str]:
    """Load and split text file into training examples."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split by double newlines (paragraphs) or sentences
    examples = [p.strip() for p in text.split('\n\n') if p.strip()]
    return examples


def main():
    """Example training script."""
    # Example: Load data from file or use sample data
    train_texts = [
        "This is an example training text for GPT-2 fine-tuning.",
        "GPT-2 is a powerful language model developed by OpenAI.",
        "Fine-tuning allows you to adapt the model to your specific domain.",
    ]
    
    # Initialize trainer
    trainer = GPT2Trainer(model_name="gpt2")
    
    # Prepare data
    train_dataset, _, data_collator = trainer.prepare_data(train_texts)
    
    # Train
    trainer.train(
        train_dataset,
        data_collator,
        epochs=3,
        batch_size=2,
        output_dir="./gpt2_example_model"
    )


if __name__ == "__main__":
    main()
