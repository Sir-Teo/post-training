"""
Direct Preference Optimization (DPO) Implementation

This module demonstrates Direct Preference Optimization, which trains language models
to align with human preferences without requiring explicit reward modeling.
DPO directly optimizes the policy using preference pairs.

Key concepts:
- Direct preference optimization without reward models
- Bradley-Terry preference modeling
- KL divergence regularization
- Stable policy optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass
import math

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
class PreferencePair:
    """Data structure for preference pairs"""
    query: str
    preferred_response: str
    rejected_response: str
    preference_strength: float = 1.0  # How strong the preference is

class DPODataset(Dataset):
    """Dataset for DPO training"""
    
    def __init__(self, preference_pairs: List[PreferencePair], tokenizer_vocab_size: int = 1000):
        self.preference_pairs = preference_pairs
        self.vocab_size = tokenizer_vocab_size
        self.max_length = 128
        
        # Simple tokenization for demonstration
        self.word_to_id = {}
        self.id_to_word = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Build vocabulary from preference pairs"""
        all_text = []
        for pair in self.preference_pairs:
            all_text.extend(pair.query.lower().split())
            all_text.extend(pair.preferred_response.lower().split())
            all_text.extend(pair.rejected_response.lower().split())
        
        # Add special tokens
        vocab = ['<pad>', '<unk>', '<start>', '<end>', '<sep>'] + list(set(all_text))
        vocab = vocab[:self.vocab_size]
        
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_word = {i: word for i, word in enumerate(vocab)}
    
    def _tokenize(self, text: str) -> List[int]:
        """Simple tokenization"""
        tokens = text.lower().split()
        return [self.word_to_id.get(token, 1) for token in tokens]  # 1 is <unk>
    
    def _pad_tokens(self, tokens: List[int], max_length: int) -> List[int]:
        """Pad tokens to max length"""
        tokens = tokens[:max_length]
        tokens += [0] * (max_length - len(tokens))
        return tokens
    
    def __len__(self):
        return len(self.preference_pairs)
    
    def __getitem__(self, idx):
        pair = self.preference_pairs[idx]
        
        # Tokenize query and responses
        query_tokens = self._tokenize(pair.query)
        preferred_tokens = self._tokenize(pair.preferred_response)
        rejected_tokens = self._tokenize(pair.rejected_response)
        
        # Create input sequences (query + response)
        preferred_seq = query_tokens + [4] + preferred_tokens  # 4 is <sep>
        rejected_seq = query_tokens + [4] + rejected_tokens
        
        # Pad sequences
        preferred_seq = self._pad_tokens(preferred_seq, self.max_length)
        rejected_seq = self._pad_tokens(rejected_seq, self.max_length)
        
        return {
            'query_tokens': torch.tensor(query_tokens[:self.max_length//2], dtype=torch.long),
            'preferred_tokens': torch.tensor(preferred_seq, dtype=torch.long),
            'rejected_tokens': torch.tensor(rejected_seq, dtype=torch.long),
            'preference_strength': torch.tensor(pair.preference_strength, dtype=torch.float),
            'query': pair.query,
            'preferred': pair.preferred_response,
            'rejected': pair.rejected_response
        }

class DPOModel(nn.Module):
    """Language model for DPO training"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Encoder for processing input
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=2)
        
        # Language modeling head
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_tokens: torch.Tensor, return_logits: bool = False):
        """
        Forward pass for DPO model
        
        Args:
            input_tokens: Input token sequence
            return_logits: Whether to return logits for each token
        """
        # Embed tokens
        embedded = self.embedding(input_tokens)
        
        # Apply dropout
        embedded = self.dropout(embedded)
        
        # Encode sequence
        encoded, _ = self.encoder(embedded)
        
        # Get logits for each position
        logits = self.lm_head(encoded)
        
        if return_logits:
            return logits
        else:
            # Return log probabilities for DPO loss
            return F.log_softmax(logits, dim=-1)
    
    def get_log_probs(self, input_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for target tokens given input"""
        log_probs = self(input_tokens, return_logits=False)
        
        # Get log probabilities for target tokens
        # Shift by one for language modeling
        target_log_probs = torch.gather(log_probs[:, :-1], 2, target_tokens[:, 1:].unsqueeze(-1))
        
        return target_log_probs.squeeze(-1)

class DPOTrainer:
    """Main class for DPO training"""
    
    def __init__(self, model: DPOModel, reference_model: DPOModel, dataset: DPODataset, beta: float = 0.1):
        self.model = model
        self.reference_model = reference_model
        self.dataset = dataset
        self.beta = beta  # KL regularization strength
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.reference_model.to(self.device)
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        # Training components
        self.optimizer = optim.Adam(model.parameters(), lr=5e-4)
        
        # Tracking
        self.training_history = {
            'loss': [],
            'dpo_loss': [],
            'kl_divergence': [],
            'preference_accuracy': [],
            'win_rate': []
        }
    
    def compute_dpo_loss(self, batch) -> Dict[str, torch.Tensor]:
        """Compute DPO loss for a batch"""
        # Get log probabilities for preferred and rejected responses
        preferred_tokens = batch['preferred_tokens'].to(self.device)
        rejected_tokens = batch['rejected_tokens'].to(self.device)
        
        # Model log probabilities
        preferred_log_probs = self.model.get_log_probs(preferred_tokens, preferred_tokens)
        rejected_log_probs = self.model.get_log_probs(rejected_tokens, rejected_tokens)
        
        # Reference model log probabilities
        with torch.no_grad():
            ref_preferred_log_probs = self.reference_model.get_log_probs(preferred_tokens, preferred_tokens)
            ref_rejected_log_probs = self.reference_model.get_log_probs(rejected_tokens, rejected_tokens)
        
        # Average log probabilities over sequence length (ignore padding)
        preferred_mask = (preferred_tokens != 0).float()
        rejected_mask = (rejected_tokens != 0).float()
        
        preferred_log_prob = torch.sum(preferred_log_probs * preferred_mask[:, 1:], dim=1) / torch.sum(preferred_mask[:, 1:], dim=1)
        rejected_log_prob = torch.sum(rejected_log_probs * rejected_mask[:, 1:], dim=1) / torch.sum(rejected_mask[:, 1:], dim=1)
        
        ref_preferred_log_prob = torch.sum(ref_preferred_log_probs * preferred_mask[:, 1:], dim=1) / torch.sum(preferred_mask[:, 1:], dim=1)
        ref_rejected_log_prob = torch.sum(ref_rejected_log_probs * rejected_mask[:, 1:], dim=1) / torch.sum(rejected_mask[:, 1:], dim=1)
        
        # Calculate KL divergence terms
        kl_preferred = preferred_log_prob - ref_preferred_log_prob
        kl_rejected = rejected_log_prob - ref_rejected_log_prob
        
        # DPO loss using Bradley-Terry model
        # log(sigma(beta * (log_pi(y_w|x) - log_pi(y_l|x) - log_pi_ref(y_w|x) + log_pi_ref(y_l|x))))
        logits = self.beta * (kl_preferred - kl_rejected)
        
        # Sigmoid loss (equivalent to log-sigmoid)
        dpo_loss = -F.logsigmoid(logits).mean()
        
        # Calculate preference accuracy
        preference_accuracy = (logits > 0).float().mean()
        
        # Calculate KL divergence for monitoring
        kl_divergence = (kl_preferred.abs() + kl_rejected.abs()).mean()
        
        return {
            'dpo_loss': dpo_loss,
            'kl_divergence': kl_divergence,
            'preference_accuracy': preference_accuracy,
            'logits': logits
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_dpo_loss = 0
        total_kl_div = 0
        total_pref_acc = 0
        num_batches = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # Compute DPO loss
            loss_dict = self.compute_dpo_loss(batch)
            
            # Total loss is just DPO loss (no additional regularization needed)
            loss = loss_dict['dpo_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_dpo_loss += loss_dict['dpo_loss'].item()
            total_kl_div += loss_dict['kl_divergence'].item()
            total_pref_acc += loss_dict['preference_accuracy'].item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'dpo_loss': total_dpo_loss / num_batches,
            'kl_divergence': total_kl_div / num_batches,
            'preference_accuracy': total_pref_acc / num_batches
        }
    
    def evaluate_preferences(self, test_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate preference learning on test set"""
        self.model.eval()
        
        total_accuracy = 0
        total_win_rate = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_dataloader:
                loss_dict = self.compute_dpo_loss(batch)
                
                # Calculate win rate (how often preferred response is actually preferred)
                logits = loss_dict['logits']
                win_rate = (logits > 0).float().mean()
                
                total_accuracy += loss_dict['preference_accuracy'].item()
                total_win_rate += win_rate.item()
                num_batches += 1
        
        return {
            'preference_accuracy': total_accuracy / num_batches,
            'win_rate': total_win_rate / num_batches
        }
    
    def generate_response(self, query: str, max_length: int = 32) -> str:
        """Generate response for a query using the trained model"""
        self.model.eval()
        
        # Tokenize query
        query_tokens = self.dataset._tokenize(query)
        query_tensor = torch.tensor(query_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Generate response
        generated_tokens = query_tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get current sequence
                current_seq = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
                
                # Get logits
                logits = self.model(current_seq, return_logits=True)
                
                # Get next token (simple greedy sampling)
                next_token = torch.argmax(logits[0, -1, :]).item()
                
                # Stop if end token or padding
                if next_token == 0 or next_token == 3:  # pad or end
                    break
                
                generated_tokens.append(next_token)
        
        # Convert back to text
        response_tokens = generated_tokens[len(query_tokens):]
        response = ' '.join([self.dataset.id_to_word.get(token, '<unk>') for token in response_tokens])
        
        return response.strip()
    
    @time_function
    def train(self, num_epochs: int = 10, batch_size: int = 4) -> Dict[str, List[float]]:
        """Train the DPO model"""
        print_colored(f"Starting DPO Training (beta={self.beta})", "blue")
        
        # Create train/test split
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        for epoch in range(num_epochs):
            print_colored(f"Epoch {epoch + 1}/{num_epochs}", "cyan")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Evaluate
            eval_metrics = self.evaluate_preferences(test_loader)
            
            # Record metrics
            self.training_history['loss'].append(train_metrics['loss'])
            self.training_history['dpo_loss'].append(train_metrics['dpo_loss'])
            self.training_history['kl_divergence'].append(train_metrics['kl_divergence'])
            self.training_history['preference_accuracy'].append(train_metrics['preference_accuracy'])
            self.training_history['win_rate'].append(eval_metrics['win_rate'])
            
            # Print progress
            print(f"Loss: {train_metrics['loss']:.4f}")
            print(f"DPO Loss: {train_metrics['dpo_loss']:.4f}")
            print(f"KL Divergence: {train_metrics['kl_divergence']:.4f}")
            print(f"Preference Accuracy: {train_metrics['preference_accuracy']:.4f}")
            print(f"Win Rate: {eval_metrics['win_rate']:.4f}")
            print()
        
        return self.training_history

def prepare_dpo_data() -> List[PreferencePair]:
    """Prepare DPO data from shared datasets"""
    data = get_sample_data()
    preference_pairs = []
    
    # Use preference data directly
    for pref_data in data['preference_data']:
        preference_pairs.append(PreferencePair(
            query=pref_data['query'],
            preferred_response=pref_data['preferred'],
            rejected_response=pref_data['rejected'],
            preference_strength=1.0
        ))
    
    # Create synthetic preference pairs from math problems
    math_problems = data['math_problems']
    for i in range(len(math_problems) - 1):
        problem1 = math_problems[i]
        problem2 = math_problems[i + 1]
        
        # Create preference pair (assume first solution is better)
        preference_pairs.append(PreferencePair(
            query=problem1['problem'],
            preferred_response=problem1['solution'],
            rejected_response=problem2['solution'],  # Use different solution as rejected
            preference_strength=0.8
        ))
    
    # Create preference pairs from reasoning tasks
    reasoning_tasks = data['reasoning_tasks']
    for i in range(len(reasoning_tasks) - 1):
        task1 = reasoning_tasks[i]
        task2 = reasoning_tasks[i + 1]
        
        # Create preference pair
        preference_pairs.append(PreferencePair(
            query=task1['question'],
            preferred_response=task1['answer'],
            rejected_response=task2['answer'],  # Use different answer as rejected
            preference_strength=0.7
        ))
    
    return preference_pairs

def demonstrate_dpo():
    """Demonstrate DPO with examples"""
    print_colored("=== Direct Preference Optimization (DPO) Demonstration ===", "green")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Prepare data
    print_colored("Preparing DPO data...", "cyan")
    preference_pairs = prepare_dpo_data()
    print(f"Created {len(preference_pairs)} preference pairs")
    
    # Show example preference pairs
    print_colored("\nSample preference pairs:", "yellow")
    for i, pair in enumerate(preference_pairs[:2]):
        print(f"\n{i+1}. Query: {pair.query}")
        print(f"   Preferred: {pair.preferred_response}")
        print(f"   Rejected: {pair.rejected_response}")
        print(f"   Strength: {pair.preference_strength}")
    
    # Create dataset
    dataset = DPODataset(preference_pairs)
    
    # Create models
    model = DPOModel(vocab_size=len(dataset.word_to_id))
    reference_model = DPOModel(vocab_size=len(dataset.word_to_id))
    
    # Initialize reference model with same weights as model
    reference_model.load_state_dict(model.state_dict())
    
    # Test different beta values
    beta_values = [0.1, 0.5, 1.0]
    results = {}
    
    for beta in beta_values:
        print_colored(f"\nTraining DPO with beta={beta}...", "cyan")
        
        # Create fresh model for each beta
        current_model = DPOModel(vocab_size=len(dataset.word_to_id))
        current_model.load_state_dict(model.state_dict())
        
        # Create trainer
        trainer = DPOTrainer(current_model, reference_model, dataset, beta=beta)
        
        # Train
        history = trainer.train(num_epochs=5, batch_size=2)
        results[beta] = history
        
        # Test response generation
        print_colored(f"Testing response generation (beta={beta}):", "yellow")
        test_query = "What is 2 + 2?"
        response = trainer.generate_response(test_query)
        print(f"Query: {test_query}")
        print(f"Response: {response}")
        print()
    
    # Plot comparison of different beta values
    print_colored("Plotting training comparison...", "cyan")
    
    plt.figure(figsize=(15, 10))
    
    # DPO Loss comparison
    plt.subplot(2, 2, 1)
    for beta in beta_values:
        plt.plot(results[beta]['dpo_loss'], label=f'β={beta}')
    plt.title('DPO Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('DPO Loss')
    plt.legend()
    plt.grid(True)
    
    # Preference Accuracy comparison
    plt.subplot(2, 2, 2)
    for beta in beta_values:
        plt.plot(results[beta]['preference_accuracy'], label=f'β={beta}')
    plt.title('Preference Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # KL Divergence comparison
    plt.subplot(2, 2, 3)
    for beta in beta_values:
        plt.plot(results[beta]['kl_divergence'], label=f'β={beta}')
    plt.title('KL Divergence vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.legend()
    plt.grid(True)
    
    # Win Rate comparison
    plt.subplot(2, 2, 4)
    for beta in beta_values:
        plt.plot(results[beta]['win_rate'], label=f'β={beta}')
    plt.title('Win Rate vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('dpo_beta_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print_colored("\nDPO demonstration completed!", "green")
    print("Key insights:")
    print("1. DPO directly optimizes preferences without explicit reward modeling")
    print("2. Beta parameter controls the strength of KL regularization")
    print("3. Higher beta values lead to more conservative updates")
    print("4. Win rate measures how often the model prefers the preferred response")
    print("5. DPO is more stable and easier to train than PPO")

if __name__ == "__main__":
    demonstrate_dpo() 