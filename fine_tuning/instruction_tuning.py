"""
Instruction Tuning Implementation

This module demonstrates instruction tuning, which trains language models to follow
instructions by fine-tuning on instruction-response pairs. This is a key component
of making language models more helpful and controllable.

Key concepts:
- Instruction-response pair training
- Template-based instruction formatting
- Instruction following evaluation
- Generalization to unseen instructions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass
import json

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_data.sample_data import get_sample_data
from utils.common import (
    print_colored, time_function, calculate_metrics, plot_training_progress,
    SimpleModel, compare_methods, set_random_seed
)

@dataclass
class InstructionExample:
    """Data structure for instruction-response pairs"""
    instruction: str
    input_text: str
    output_text: str
    task_type: str

class InstructionDataset(Dataset):
    """Dataset for instruction tuning"""
    
    def __init__(self, examples: List[InstructionExample], tokenizer_vocab_size: int = 1000):
        self.examples = examples
        self.vocab_size = tokenizer_vocab_size
        self.max_length = 128
        
        # Simple tokenization for demonstration
        self.word_to_id = {}
        self.id_to_word = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Build vocabulary from examples"""
        all_text = []
        for example in self.examples:
            all_text.extend(example.instruction.lower().split())
            all_text.extend(example.input_text.lower().split())
            all_text.extend(example.output_text.lower().split())
        
        # Add special tokens
        vocab = ['<pad>', '<unk>', '<start>', '<end>'] + list(set(all_text))
        vocab = vocab[:self.vocab_size]
        
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_word = {i: word for i, word in enumerate(vocab)}
    
    def _tokenize(self, text: str) -> List[int]:
        """Simple tokenization"""
        tokens = text.lower().split()
        return [self.word_to_id.get(token, 1) for token in tokens]  # 1 is <unk>
    
    def _format_instruction(self, example: InstructionExample) -> str:
        """Format instruction following the standard template"""
        if example.input_text:
            return f"Instruction: {example.instruction}\nInput: {example.input_text}\nOutput: {example.output_text}"
        else:
            return f"Instruction: {example.instruction}\nOutput: {example.output_text}"
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format the full instruction-response pair
        formatted_text = self._format_instruction(example)
        
        # Create input (instruction + input) and target (output)
        input_part = f"Instruction: {example.instruction}"
        if example.input_text:
            input_part += f"\nInput: {example.input_text}"
        input_part += "\nOutput:"
        
        # Tokenize
        input_tokens = self._tokenize(input_part)
        target_tokens = self._tokenize(example.output_text)
        
        # Pad/truncate to max length
        input_tokens = input_tokens[:self.max_length//2]
        target_tokens = target_tokens[:self.max_length//2]
        
        # Pad
        input_tokens += [0] * (self.max_length//2 - len(input_tokens))
        target_tokens += [0] * (self.max_length//2 - len(target_tokens))
        
        return {
            'input': torch.tensor(input_tokens, dtype=torch.long),
            'target': torch.tensor(target_tokens, dtype=torch.long),
            'task_type': example.task_type,
            'instruction': example.instruction
        }

class InstructionTuningModel(nn.Module):
    """Model for instruction tuning"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Encoder for instruction processing
        self.instruction_encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # Decoder for response generation
        self.response_decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Attention mechanism for instruction conditioning
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_seq: torch.Tensor, target_seq: torch.Tensor = None):
        """
        Forward pass for instruction tuning
        
        Args:
            input_seq: Instruction + input tokens
            target_seq: Expected output tokens (for training)
        """
        batch_size = input_seq.size(0)
        
        # Encode instruction
        input_embedded = self.embedding(input_seq)
        instruction_hidden, _ = self.instruction_encoder(input_embedded)
        
        if target_seq is not None:
            # Training mode: teacher forcing
            target_embedded = self.embedding(target_seq)
            
            # Decode with attention to instruction
            decoder_output, _ = self.response_decoder(target_embedded)
            
            # Apply attention to instruction context
            attended_output, _ = self.attention(
                decoder_output, instruction_hidden, instruction_hidden
            )
            
            # Apply dropout
            attended_output = self.dropout(attended_output)
            
            # Project to vocabulary
            logits = self.output_projection(attended_output)
            
            return logits
        else:
            # Inference mode: autoregressive generation
            generated_tokens = []
            hidden = None
            
            # Start with a start token
            current_input = torch.zeros(batch_size, 1, self.embed_dim).to(input_seq.device)
            
            for _ in range(32):  # Max generation length
                decoder_output, hidden = self.response_decoder(current_input, hidden)
                
                # Apply attention to instruction
                attended_output, _ = self.attention(
                    decoder_output, instruction_hidden, instruction_hidden
                )
                
                # Get next token
                logits = self.output_projection(attended_output)
                next_token = torch.argmax(logits, dim=-1)
                generated_tokens.append(next_token)
                
                # Prepare next input
                current_input = self.embedding(next_token)
            
            return torch.cat(generated_tokens, dim=1)

class InstructionTuner:
    """Main class for instruction tuning"""
    
    def __init__(self, model: InstructionTuningModel, dataset: InstructionDataset):
        self.model = model
        self.dataset = dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training components
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Tracking
        self.training_history = {
            'loss': [],
            'instruction_following_score': [],
            'task_diversity_score': [],
            'generalization_score': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # Move to device
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            logits = self.model(inputs, targets)
            
            # Calculate loss
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {'loss': total_loss / num_batches}
    
    def evaluate_instruction_following(self, test_examples: List[InstructionExample]) -> Dict[str, float]:
        """Evaluate instruction following capability"""
        self.model.eval()
        
        task_type_accuracy = {}
        total_correct = 0
        total_examples = 0
        
        with torch.no_grad():
            for example in test_examples:
                # Create input
                input_text = f"Instruction: {example.instruction}"
                if example.input_text:
                    input_text += f"\nInput: {example.input_text}"
                input_text += "\nOutput:"
                
                # Tokenize and predict
                input_tokens = self.dataset._tokenize(input_text)
                input_tokens += [0] * (64 - len(input_tokens))  # Pad to fixed length
                input_tensor = torch.tensor(input_tokens[:64], dtype=torch.long).unsqueeze(0).to(self.device)
                
                # Generate response
                generated = self.model(input_tensor)
                
                # Convert back to text (simplified evaluation)
                predicted_tokens = generated[0].cpu().numpy()
                predicted_text = ' '.join([
                    self.dataset.id_to_word.get(token, '<unk>') 
                    for token in predicted_tokens if token != 0
                ])
                
                # Simple accuracy check (contains key words from expected output)
                expected_words = set(example.output_text.lower().split())
                predicted_words = set(predicted_text.lower().split())
                
                # Calculate word overlap
                if expected_words:
                    overlap = len(expected_words & predicted_words) / len(expected_words)
                    is_correct = overlap > 0.5
                else:
                    is_correct = False
                
                if is_correct:
                    total_correct += 1
                
                # Track by task type
                if example.task_type not in task_type_accuracy:
                    task_type_accuracy[example.task_type] = {'correct': 0, 'total': 0}
                
                task_type_accuracy[example.task_type]['correct'] += int(is_correct)
                task_type_accuracy[example.task_type]['total'] += 1
                total_examples += 1
        
        # Calculate metrics
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0
        
        task_accuracies = {}
        for task_type, stats in task_type_accuracy.items():
            task_accuracies[task_type] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        return {
            'overall_accuracy': overall_accuracy,
            'task_accuracies': task_accuracies,
            'instruction_following_score': overall_accuracy,
            'task_diversity_score': len(task_accuracies) / 10.0,  # Normalize by expected task types
            'generalization_score': min(task_accuracies.values()) if task_accuracies else 0
        }
    
    @time_function
    def train(self, num_epochs: int = 10, batch_size: int = 8) -> Dict[str, List[float]]:
        """Train the instruction tuning model"""
        print_colored("Starting Instruction Tuning Training", "blue")
        
        # Create train/test split
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Extract test examples for evaluation
        test_examples = [self.dataset.examples[i] for i in test_dataset.indices]
        
        for epoch in range(num_epochs):
            print_colored(f"Epoch {epoch + 1}/{num_epochs}", "cyan")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Evaluate
            eval_metrics = self.evaluate_instruction_following(test_examples)
            
            # Record metrics
            self.training_history['loss'].append(train_metrics['loss'])
            self.training_history['instruction_following_score'].append(eval_metrics['instruction_following_score'])
            self.training_history['task_diversity_score'].append(eval_metrics['task_diversity_score'])
            self.training_history['generalization_score'].append(eval_metrics['generalization_score'])
            
            # Print progress
            print(f"Loss: {train_metrics['loss']:.4f}")
            print(f"Instruction Following: {eval_metrics['instruction_following_score']:.4f}")
            print(f"Task Diversity: {eval_metrics['task_diversity_score']:.4f}")
            print(f"Generalization: {eval_metrics['generalization_score']:.4f}")
            
            # Print task-specific accuracies
            for task_type, accuracy in eval_metrics['task_accuracies'].items():
                print(f"  {task_type}: {accuracy:.4f}")
            
            print()
        
        return self.training_history

def prepare_instruction_data() -> List[InstructionExample]:
    """Prepare instruction tuning data from shared datasets"""
    data = get_sample_data()
    instruction_examples = []
    
    # Convert math problems to instruction format
    for problem in data['math_problems']:
        instruction_examples.append(InstructionExample(
            instruction="Solve the following math problem step by step.",
            input_text=problem['problem'],
            output_text=problem['solution'],
            task_type="math"
        ))
    
    # Convert reasoning tasks
    for task in data['reasoning_tasks']:
        instruction_examples.append(InstructionExample(
            instruction="Answer the following reasoning question with an explanation.",
            input_text=task['question'],
            output_text=task['answer'],
            task_type="reasoning"
        ))
    
    # Convert instruction examples
    for example in data['instruction_data']:
        instruction_examples.append(InstructionExample(
            instruction=example['instruction'],
            input_text=example.get('input', ''),
            output_text=example['output'],
            task_type="instruction"
        ))
    
    # Convert code problems
    for problem in data['code_problems']:
        instruction_examples.append(InstructionExample(
            instruction="Write a Python function to solve the following problem.",
            input_text=problem['problem'],
            output_text=problem['solution'],
            task_type="code"
        ))
    
    return instruction_examples

def demonstrate_instruction_tuning():
    """Demonstrate instruction tuning with examples"""
    print_colored("=== Instruction Tuning Demonstration ===", "green")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Prepare data
    print_colored("Preparing instruction tuning data...", "cyan")
    instruction_examples = prepare_instruction_data()
    print(f"Created {len(instruction_examples)} instruction examples")
    
    # Show example instructions
    print_colored("\nSample instruction examples:", "yellow")
    for i, example in enumerate(instruction_examples[:3]):
        print(f"\n{i+1}. Task: {example.task_type}")
        print(f"   Instruction: {example.instruction}")
        print(f"   Input: {example.input_text}")
        print(f"   Output: {example.output_text}")
    
    # Create dataset
    dataset = InstructionDataset(instruction_examples)
    
    # Create model
    model = InstructionTuningModel(vocab_size=len(dataset.word_to_id))
    
    # Create trainer
    trainer = InstructionTuner(model, dataset)
    
    # Train
    print_colored("\nTraining instruction tuning model...", "cyan")
    history = trainer.train(num_epochs=5, batch_size=4)
    
    # Plot training progress
    print_colored("\nPlotting training progress...", "cyan")
    
    plt.figure(figsize=(15, 10))
    
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(history['loss'], 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Instruction Following Score
    plt.subplot(2, 2, 2)
    plt.plot(history['instruction_following_score'], 'g-', label='Instruction Following')
    plt.title('Instruction Following Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Task Diversity Score
    plt.subplot(2, 2, 3)
    plt.plot(history['task_diversity_score'], 'r-', label='Task Diversity')
    plt.title('Task Diversity Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Generalization Score
    plt.subplot(2, 2, 4)
    plt.plot(history['generalization_score'], 'm-', label='Generalization')
    plt.title('Generalization Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('instruction_tuning_training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print_colored("\nInstruction tuning demonstration completed!", "green")
    print("Key insights:")
    print("1. Instruction tuning teaches models to follow diverse instructions")
    print("2. The model learns to handle different task types (math, reasoning, code)")
    print("3. Generalization to unseen instructions is a key challenge")
    print("4. Task diversity helps improve overall instruction following")

if __name__ == "__main__":
    demonstrate_instruction_tuning() 