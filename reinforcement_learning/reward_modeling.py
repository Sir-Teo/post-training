"""
Reward Modeling Implementation

This module demonstrates reward modeling for reinforcement learning in language models.
It implements both process-based and outcome-based reward models, which are crucial
for training language models with RLHF and other RL approaches.

Key concepts:
- Process vs Outcome reward modeling
- Preference learning from human feedback
- Reward hacking mitigation
- Iterative reward model training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import random
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_data.sample_data import get_sample_data
from utils.common import (
    print_colored, time_function, calculate_metrics, plot_training_progress,
    SimpleModel, compare_methods, set_random_seed
)

class RewardType(Enum):
    """Types of reward modeling"""
    PROCESS = "process"  # Rewards intermediate steps
    OUTCOME = "outcome"  # Rewards final outcomes only

@dataclass
class RewardExample:
    """Data structure for reward modeling examples"""
    query: str
    response: str
    steps: List[str]  # For process rewards
    reward_score: float
    preference_pair: Optional[Tuple[str, str]] = None  # For preference learning
    reward_type: RewardType = RewardType.OUTCOME

class RewardDataset(Dataset):
    """Dataset for reward modeling"""
    
    def __init__(self, examples: List[RewardExample], tokenizer_vocab_size: int = 1000):
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
            all_text.extend(example.query.lower().split())
            all_text.extend(example.response.lower().split())
            for step in example.steps:
                all_text.extend(step.lower().split())
        
        # Add special tokens
        vocab = ['<pad>', '<unk>', '<start>', '<end>', '<step>'] + list(set(all_text))
        vocab = vocab[:self.vocab_size]
        
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_word = {i: word for i, word in enumerate(vocab)}
    
    def _tokenize(self, text: str) -> List[int]:
        """Simple tokenization"""
        tokens = text.lower().split()
        return [self.word_to_id.get(token, 1) for token in tokens]  # 1 is <unk>
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize query and response
        query_tokens = self._tokenize(example.query)
        response_tokens = self._tokenize(example.response)
        
        # For process rewards, include steps
        if example.reward_type == RewardType.PROCESS and example.steps:
            step_tokens = []
            for step in example.steps:
                step_tokens.extend([4])  # <step> token
                step_tokens.extend(self._tokenize(step))
        else:
            step_tokens = []
        
        # Combine all tokens
        all_tokens = query_tokens + response_tokens + step_tokens
        
        # Pad/truncate to max length
        all_tokens = all_tokens[:self.max_length]
        all_tokens += [0] * (self.max_length - len(all_tokens))
        
        return {
            'tokens': torch.tensor(all_tokens, dtype=torch.long),
            'reward_score': torch.tensor(example.reward_score, dtype=torch.float),
            'reward_type': example.reward_type.value,
            'query': example.query,
            'response': example.response,
            'steps': example.steps
        }

class ProcessRewardModel(nn.Module):
    """Reward model that evaluates intermediate steps"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Step-wise encoder
        self.step_encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # Attention for step importance
        self.step_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Reward predictor for each step
        self.step_reward_predictor = nn.Linear(hidden_dim, 1)
        
        # Final aggregation
        self.final_aggregator = nn.Linear(hidden_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, tokens: torch.Tensor, return_step_rewards: bool = False):
        """
        Forward pass for process reward modeling
        
        Args:
            tokens: Input tokens
            return_step_rewards: Whether to return individual step rewards
        """
        # Embed tokens
        embedded = self.embedding(tokens)
        
        # Encode sequence
        encoded, _ = self.step_encoder(embedded)
        
        # Apply attention to identify important steps
        attended, attention_weights = self.step_attention(encoded, encoded, encoded)
        
        # Apply dropout
        attended = self.dropout(attended)
        
        # Predict step-wise rewards
        step_rewards = self.step_reward_predictor(attended)
        
        # Aggregate final reward
        # Use attention-weighted average
        final_features = torch.mean(attended, dim=1)
        final_reward = self.final_aggregator(final_features)
        
        if return_step_rewards:
            return final_reward, step_rewards, attention_weights
        else:
            return final_reward

class OutcomeRewardModel(nn.Module):
    """Reward model that evaluates final outcomes only"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Encoder
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, tokens: torch.Tensor):
        """
        Forward pass for outcome reward modeling
        
        Args:
            tokens: Input tokens
        """
        # Embed tokens
        embedded = self.embedding(tokens)
        
        # Encode sequence
        encoded, _ = self.encoder(embedded)
        
        # Use final hidden state
        final_state = encoded[:, -1, :]
        
        # Predict reward
        reward = self.reward_predictor(final_state)
        
        return reward

class RewardModelTrainer:
    """Main class for training reward models"""
    
    def __init__(self, model: nn.Module, dataset: RewardDataset, reward_type: RewardType):
        self.model = model
        self.dataset = dataset
        self.reward_type = reward_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training components
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # For preference learning
        self.preference_criterion = nn.BCEWithLogitsLoss()
        
        # Tracking
        self.training_history = {
            'loss': [],
            'reward_accuracy': [],
            'preference_accuracy': [],
            'reward_calibration': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_reward_accuracy = 0
        num_batches = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # Move to device
            tokens = batch['tokens'].to(self.device)
            reward_scores = batch['reward_score'].to(self.device)
            
            # Forward pass
            if self.reward_type == RewardType.PROCESS:
                predicted_rewards = self.model(tokens)
            else:
                predicted_rewards = self.model(tokens)
            
            # Calculate loss
            loss = self.criterion(predicted_rewards.squeeze(), reward_scores)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Calculate accuracy (within threshold)
            with torch.no_grad():
                accuracy = torch.mean((torch.abs(predicted_rewards.squeeze() - reward_scores) < 0.1).float())
            
            total_loss += loss.item()
            total_reward_accuracy += accuracy.item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'reward_accuracy': total_reward_accuracy / num_batches
        }
    
    def evaluate_calibration(self, test_dataloader: DataLoader) -> float:
        """Evaluate reward model calibration"""
        self.model.eval()
        predicted_rewards = []
        actual_rewards = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                tokens = batch['tokens'].to(self.device)
                reward_scores = batch['reward_score'].to(self.device)
                
                if self.reward_type == RewardType.PROCESS:
                    predicted = self.model(tokens)
                else:
                    predicted = self.model(tokens)
                
                predicted_rewards.extend(predicted.squeeze().cpu().numpy())
                actual_rewards.extend(reward_scores.cpu().numpy())
        
        # Calculate correlation as calibration metric
        predicted_rewards = np.array(predicted_rewards)
        actual_rewards = np.array(actual_rewards)
        
        correlation = np.corrcoef(predicted_rewards, actual_rewards)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def analyze_reward_distribution(self, test_dataloader: DataLoader) -> Dict[str, float]:
        """Analyze reward distribution to detect potential reward hacking"""
        self.model.eval()
        predicted_rewards = []
        actual_rewards = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                tokens = batch['tokens'].to(self.device)
                reward_scores = batch['reward_score'].to(self.device)
                
                if self.reward_type == RewardType.PROCESS:
                    predicted = self.model(tokens)
                else:
                    predicted = self.model(tokens)
                
                predicted_rewards.extend(predicted.squeeze().cpu().numpy())
                actual_rewards.extend(reward_scores.cpu().numpy())
        
        predicted_rewards = np.array(predicted_rewards)
        actual_rewards = np.array(actual_rewards)
        
        # Calculate distribution metrics
        pred_mean = np.mean(predicted_rewards)
        pred_std = np.std(predicted_rewards)
        actual_mean = np.mean(actual_rewards)
        actual_std = np.std(actual_rewards)
        
        # Detect potential reward hacking (over-optimization)
        reward_hacking_score = max(0, (pred_mean - actual_mean) / actual_std)
        
        return {
            'pred_mean': pred_mean,
            'pred_std': pred_std,
            'actual_mean': actual_mean,
            'actual_std': actual_std,
            'reward_hacking_score': reward_hacking_score
        }
    
    @time_function
    def train(self, num_epochs: int = 10, batch_size: int = 8) -> Dict[str, List[float]]:
        """Train the reward model"""
        print_colored(f"Starting {self.reward_type.value.title()} Reward Model Training", "blue")
        
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
            calibration = self.evaluate_calibration(test_loader)
            reward_analysis = self.analyze_reward_distribution(test_loader)
            
            # Record metrics
            self.training_history['loss'].append(train_metrics['loss'])
            self.training_history['reward_accuracy'].append(train_metrics['reward_accuracy'])
            self.training_history['reward_calibration'].append(calibration)
            
            # Print progress
            print(f"Loss: {train_metrics['loss']:.4f}")
            print(f"Reward Accuracy: {train_metrics['reward_accuracy']:.4f}")
            print(f"Calibration: {calibration:.4f}")
            print(f"Reward Hacking Score: {reward_analysis['reward_hacking_score']:.4f}")
            print()
        
        return self.training_history

def prepare_reward_data() -> Tuple[List[RewardExample], List[RewardExample]]:
    """Prepare reward modeling data from shared datasets"""
    data = get_sample_data()
    process_examples = []
    outcome_examples = []
    
    # Convert math problems to reward examples
    for i, problem in enumerate(data['math_problems']):
        # Create process reward example (step-by-step)
        steps = problem['solution'].split('. ')
        reward_score = 0.8 + 0.2 * (i % 3) / 3  # Simulate varying quality
        
        process_examples.append(RewardExample(
            query=problem['problem'],
            response=problem['solution'],
            steps=steps,
            reward_score=reward_score,
            reward_type=RewardType.PROCESS
        ))
        
        # Create outcome reward example (final answer only)
        outcome_examples.append(RewardExample(
            query=problem['problem'],
            response=problem['solution'],
            steps=[],
            reward_score=reward_score,
            reward_type=RewardType.OUTCOME
        ))
    
    # Convert reasoning tasks
    for i, task in enumerate(data['reasoning_tasks']):
        reward_score = 0.7 + 0.3 * (i % 4) / 4  # Simulate varying quality
        
        # Process reward with reasoning steps
        steps = task['answer'].split('. ')
        process_examples.append(RewardExample(
            query=task['question'],
            response=task['answer'],
            steps=steps,
            reward_score=reward_score,
            reward_type=RewardType.PROCESS
        ))
        
        # Outcome reward
        outcome_examples.append(RewardExample(
            query=task['question'],
            response=task['answer'],
            steps=[],
            reward_score=reward_score,
            reward_type=RewardType.OUTCOME
        ))
    
    # Add some preference pairs for preference learning
    preference_data = data['preference_data']
    for i, pref in enumerate(preference_data):
        # Preferred response gets higher reward
        process_examples.append(RewardExample(
            query=pref['query'],
            response=pref['preferred'],
            steps=pref['preferred'].split('. '),
            reward_score=0.9,
            preference_pair=(pref['preferred'], pref['rejected']),
            reward_type=RewardType.PROCESS
        ))
        
        # Rejected response gets lower reward
        process_examples.append(RewardExample(
            query=pref['query'],
            response=pref['rejected'],
            steps=pref['rejected'].split('. '),
            reward_score=0.3,
            preference_pair=(pref['preferred'], pref['rejected']),
            reward_type=RewardType.PROCESS
        ))
    
    return process_examples, outcome_examples

def demonstrate_reward_modeling():
    """Demonstrate reward modeling with examples"""
    print_colored("=== Reward Modeling Demonstration ===", "green")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Prepare data
    print_colored("Preparing reward modeling data...", "cyan")
    process_examples, outcome_examples = prepare_reward_data()
    
    print(f"Created {len(process_examples)} process reward examples")
    print(f"Created {len(outcome_examples)} outcome reward examples")
    
    # Show example rewards
    print_colored("\nSample reward examples:", "yellow")
    for i, example in enumerate(process_examples[:2]):
        print(f"\n{i+1}. Query: {example.query}")
        print(f"   Response: {example.response}")
        print(f"   Steps: {example.steps}")
        print(f"   Reward Score: {example.reward_score:.3f}")
        print(f"   Type: {example.reward_type.value}")
    
    # Create datasets
    process_dataset = RewardDataset(process_examples)
    outcome_dataset = RewardDataset(outcome_examples)
    
    # Create models
    process_model = ProcessRewardModel(vocab_size=len(process_dataset.word_to_id))
    outcome_model = OutcomeRewardModel(vocab_size=len(outcome_dataset.word_to_id))
    
    # Create trainers
    process_trainer = RewardModelTrainer(process_model, process_dataset, RewardType.PROCESS)
    outcome_trainer = RewardModelTrainer(outcome_model, outcome_dataset, RewardType.OUTCOME)
    
    # Train both models
    print_colored("\nTraining process reward model...", "cyan")
    process_history = process_trainer.train(num_epochs=5, batch_size=4)
    
    print_colored("\nTraining outcome reward model...", "cyan")
    outcome_history = outcome_trainer.train(num_epochs=5, batch_size=4)
    
    # Compare models
    print_colored("\nComparing reward models...", "cyan")
    
    plt.figure(figsize=(15, 10))
    
    # Loss comparison
    plt.subplot(2, 2, 1)
    plt.plot(process_history['loss'], 'b-', label='Process Reward Model')
    plt.plot(outcome_history['loss'], 'r-', label='Outcome Reward Model')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy comparison
    plt.subplot(2, 2, 2)
    plt.plot(process_history['reward_accuracy'], 'b-', label='Process Reward Model')
    plt.plot(outcome_history['reward_accuracy'], 'r-', label='Outcome Reward Model')
    plt.title('Reward Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Calibration comparison
    plt.subplot(2, 2, 3)
    plt.plot(process_history['reward_calibration'], 'b-', label='Process Reward Model')
    plt.plot(outcome_history['reward_calibration'], 'r-', label='Outcome Reward Model')
    plt.title('Reward Calibration Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Calibration (Correlation)')
    plt.legend()
    plt.grid(True)
    
    # Final performance comparison
    plt.subplot(2, 2, 4)
    models = ['Process\nReward', 'Outcome\nReward']
    final_accuracy = [process_history['reward_accuracy'][-1], outcome_history['reward_accuracy'][-1]]
    final_calibration = [process_history['reward_calibration'][-1], outcome_history['reward_calibration'][-1]]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, final_accuracy, width, label='Accuracy', alpha=0.7)
    plt.bar(x + width/2, final_calibration, width, label='Calibration', alpha=0.7)
    plt.title('Final Performance Comparison')
    plt.xlabel('Reward Model Type')
    plt.ylabel('Score')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reward_modeling_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Demonstrate step-wise reward analysis for process model
    print_colored("\nAnalyzing step-wise rewards (Process Model)...", "cyan")
    
    # Get a sample and analyze step rewards
    sample_tokens = process_dataset[0]['tokens'].unsqueeze(0).to(process_trainer.device)
    process_model.eval()
    
    with torch.no_grad():
        final_reward, step_rewards, attention_weights = process_model(
            sample_tokens, return_step_rewards=True
        )
    
    print(f"Sample Query: {process_dataset[0]['query']}")
    print(f"Sample Response: {process_dataset[0]['response']}")
    print(f"Final Reward: {final_reward.item():.3f}")
    print(f"Step Rewards Shape: {step_rewards.shape}")
    print(f"Attention Weights Shape: {attention_weights.shape}")
    
    print_colored("\nReward modeling demonstration completed!", "green")
    print("Key insights:")
    print("1. Process rewards evaluate intermediate steps, outcome rewards evaluate final results")
    print("2. Process rewards can provide more fine-grained feedback for training")
    print("3. Calibration is crucial to prevent reward hacking")
    print("4. Different reward models may be better for different tasks")

if __name__ == "__main__":
    demonstrate_reward_modeling() 