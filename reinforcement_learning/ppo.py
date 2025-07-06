"""
Proximal Policy Optimization (PPO) Implementation

This module demonstrates Proximal Policy Optimization for language model training.
PPO is a popular policy gradient method that uses a clipped surrogate objective
to ensure stable training and prevent large policy updates.

Key concepts:
- Policy gradient with clipped surrogate objective
- Value function estimation
- Advantage estimation
- KL divergence regularization
- Actor-Critic architecture
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
class PPOExperience:
    """Data structure for PPO experiences"""
    query: str
    response: str
    reward: float
    log_prob: float
    value: float
    advantage: float = 0.0

class PPODataset(Dataset):
    """Dataset for PPO training"""
    
    def __init__(self, experiences: List[PPOExperience], tokenizer_vocab_size: int = 1000):
        self.experiences = experiences
        self.vocab_size = tokenizer_vocab_size
        self.max_length = 128
        
        # Simple tokenization for demonstration
        self.word_to_id = {}
        self.id_to_word = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Build vocabulary from experiences"""
        all_text = []
        for exp in self.experiences:
            all_text.extend(exp.query.lower().split())
            all_text.extend(exp.response.lower().split())
        
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
        return len(self.experiences)
    
    def __getitem__(self, idx):
        exp = self.experiences[idx]
        
        # Tokenize query and response
        query_tokens = self._tokenize(exp.query)
        response_tokens = self._tokenize(exp.response)
        
        # Create full sequence (query + response)
        full_seq = query_tokens + [4] + response_tokens  # 4 is <sep>
        full_seq = self._pad_tokens(full_seq, self.max_length)
        
        return {
            'tokens': torch.tensor(full_seq, dtype=torch.long),
            'query_tokens': torch.tensor(query_tokens[:self.max_length//2], dtype=torch.long),
            'response_tokens': torch.tensor(response_tokens[:self.max_length//2], dtype=torch.long),
            'reward': torch.tensor(exp.reward, dtype=torch.float),
            'old_log_prob': torch.tensor(exp.log_prob, dtype=torch.float),
            'value': torch.tensor(exp.value, dtype=torch.float),
            'advantage': torch.tensor(exp.advantage, dtype=torch.float),
            'query': exp.query,
            'response': exp.response
        }

class PPOActor(nn.Module):
    """Actor network for PPO (policy)"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Policy network
        self.policy_net = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=2)
        
        # Output layer for action probabilities
        self.action_head = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, tokens: torch.Tensor, return_logits: bool = False):
        """
        Forward pass for actor
        
        Args:
            tokens: Input tokens
            return_logits: Whether to return raw logits
        """
        # Embed tokens
        embedded = self.embedding(tokens)
        embedded = self.dropout(embedded)
        
        # Pass through policy network
        policy_output, _ = self.policy_net(embedded)
        
        # Get action logits
        logits = self.action_head(policy_output)
        
        if return_logits:
            return logits
        else:
            # Return log probabilities
            return F.log_softmax(logits, dim=-1)
    
    def get_action_log_probs(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for specific actions"""
        log_probs = self(tokens, return_logits=False)
        
        # Get log probabilities for actions (shifted for language modeling)
        action_log_probs = torch.gather(log_probs[:, :-1], 2, actions[:, 1:].unsqueeze(-1))
        
        return action_log_probs.squeeze(-1)

class PPOCritic(nn.Module):
    """Critic network for PPO (value function)"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Value network
        self.value_net = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=2)
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, tokens: torch.Tensor):
        """
        Forward pass for critic
        
        Args:
            tokens: Input tokens
        """
        # Embed tokens
        embedded = self.embedding(tokens)
        embedded = self.dropout(embedded)
        
        # Pass through value network
        value_output, _ = self.value_net(embedded)
        
        # Get value estimate (use final hidden state)
        value = self.value_head(value_output[:, -1, :])
        
        return value.squeeze(-1)

class PPOTrainer:
    """Main class for PPO training"""
    
    def __init__(self, actor: PPOActor, critic: PPOCritic, dataset: PPODataset, 
                 clip_ratio: float = 0.2, vf_coef: float = 0.5, ent_coef: float = 0.01):
        self.actor = actor
        self.critic = critic
        self.dataset = dataset
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef  # Value function coefficient
        self.ent_coef = ent_coef  # Entropy coefficient
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)
        
        # Tracking
        self.training_history = {
            'actor_loss': [],
            'critic_loss': [],
            'total_loss': [],
            'clip_fraction': [],
            'kl_divergence': [],
            'entropy': [],
            'explained_variance': []
        }
    
    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor, 
                          gamma: float = 0.99, lam: float = 0.95) -> torch.Tensor:
        """Compute advantages using GAE (Generalized Advantage Estimation)"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
        
        return advantages
    
    def compute_ppo_loss(self, batch) -> Dict[str, torch.Tensor]:
        """Compute PPO loss for a batch"""
        tokens = batch['tokens'].to(self.device)
        rewards = batch['reward'].to(self.device)
        old_log_probs = batch['old_log_prob'].to(self.device)
        advantages = batch['advantage'].to(self.device)
        old_values = batch['value'].to(self.device)
        
        # Get current policy log probabilities
        current_log_probs = self.actor.get_action_log_probs(tokens, tokens)
        
        # Average over sequence length (ignore padding)
        mask = (tokens != 0).float()
        current_log_probs = torch.sum(current_log_probs * mask[:, 1:], dim=1) / torch.sum(mask[:, 1:], dim=1)
        
        # Get current values
        current_values = self.critic(tokens)
        
        # Compute ratio
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # Compute surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        
        # Actor loss (negative because we want to maximize)
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic loss
        returns = advantages + old_values
        critic_loss = F.mse_loss(current_values, returns)
        
        # Entropy loss (for exploration)
        logits = self.actor(tokens, return_logits=True)
        entropy = -torch.mean(torch.sum(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1), dim=-1))
        
        # Total loss
        total_loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
        
        # Compute metrics
        clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_ratio).float())
        kl_divergence = torch.mean(old_log_probs - current_log_probs)
        explained_variance = 1 - torch.var(returns - current_values) / torch.var(returns)
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'total_loss': total_loss,
            'clip_fraction': clip_fraction,
            'kl_divergence': kl_divergence,
            'entropy': entropy,
            'explained_variance': explained_variance
        }
    
    def train_epoch(self, dataloader: DataLoader, num_updates: int = 4) -> Dict[str, float]:
        """Train for one epoch with multiple updates per batch"""
        self.actor.train()
        self.critic.train()
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_loss = 0
        total_clip_fraction = 0
        total_kl_div = 0
        total_entropy = 0
        total_explained_var = 0
        num_batches = 0
        
        for batch in dataloader:
            # Multiple updates per batch (common in PPO)
            for _ in range(num_updates):
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                # Compute PPO loss
                loss_dict = self.compute_ppo_loss(batch)
                
                # Backward pass
                loss_dict['total_loss'].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                
                # Update parameters
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Track metrics
                total_actor_loss += loss_dict['actor_loss'].item()
                total_critic_loss += loss_dict['critic_loss'].item()
                total_loss += loss_dict['total_loss'].item()
                total_clip_fraction += loss_dict['clip_fraction'].item()
                total_kl_div += loss_dict['kl_divergence'].item()
                total_entropy += loss_dict['entropy'].item()
                total_explained_var += loss_dict['explained_variance'].item()
                num_batches += 1
        
        return {
            'actor_loss': total_actor_loss / num_batches,
            'critic_loss': total_critic_loss / num_batches,
            'total_loss': total_loss / num_batches,
            'clip_fraction': total_clip_fraction / num_batches,
            'kl_divergence': total_kl_div / num_batches,
            'entropy': total_entropy / num_batches,
            'explained_variance': total_explained_var / num_batches
        }
    
    def generate_response(self, query: str, max_length: int = 32) -> Tuple[str, float, float]:
        """Generate response for a query using the trained policy"""
        self.actor.eval()
        self.critic.eval()
        
        # Tokenize query
        query_tokens = self.dataset._tokenize(query)
        query_tensor = torch.tensor(query_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Generate response
        generated_tokens = query_tokens.copy()
        total_log_prob = 0
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get current sequence
                current_seq = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
                
                # Get action probabilities
                logits = self.actor(current_seq, return_logits=True)
                probs = F.softmax(logits[0, -1, :], dim=-1)
                
                # Sample action
                action = torch.multinomial(probs, 1).item()
                
                # Stop if end token or padding
                if action == 0 or action == 3:  # pad or end
                    break
                
                # Add to sequence
                generated_tokens.append(action)
                
                # Track log probability
                total_log_prob += torch.log(probs[action]).item()
            
            # Get value estimate
            full_seq = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
            value = self.critic(full_seq).item()
        
        # Convert back to text
        response_tokens = generated_tokens[len(query_tokens):]
        response = ' '.join([self.dataset.id_to_word.get(token, '<unk>') for token in response_tokens])
        
        return response.strip(), total_log_prob, value
    
    @time_function
    def train(self, num_epochs: int = 10, batch_size: int = 4) -> Dict[str, List[float]]:
        """Train the PPO model"""
        print_colored(f"Starting PPO Training (clip_ratio={self.clip_ratio})", "blue")
        
        # Create train/test split
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            print_colored(f"Epoch {epoch + 1}/{num_epochs}", "cyan")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Record metrics
            self.training_history['actor_loss'].append(train_metrics['actor_loss'])
            self.training_history['critic_loss'].append(train_metrics['critic_loss'])
            self.training_history['total_loss'].append(train_metrics['total_loss'])
            self.training_history['clip_fraction'].append(train_metrics['clip_fraction'])
            self.training_history['kl_divergence'].append(train_metrics['kl_divergence'])
            self.training_history['entropy'].append(train_metrics['entropy'])
            self.training_history['explained_variance'].append(train_metrics['explained_variance'])
            
            # Print progress
            print(f"Actor Loss: {train_metrics['actor_loss']:.4f}")
            print(f"Critic Loss: {train_metrics['critic_loss']:.4f}")
            print(f"Total Loss: {train_metrics['total_loss']:.4f}")
            print(f"Clip Fraction: {train_metrics['clip_fraction']:.4f}")
            print(f"KL Divergence: {train_metrics['kl_divergence']:.4f}")
            print(f"Entropy: {train_metrics['entropy']:.4f}")
            print(f"Explained Variance: {train_metrics['explained_variance']:.4f}")
            print()
        
        return self.training_history

def prepare_ppo_data(reward_model=None) -> List[PPOExperience]:
    """Prepare PPO data from shared datasets"""
    data = get_sample_data()
    experiences = []
    
    # Convert instruction data to PPO experiences
    for i, inst in enumerate(data['instruction_data']):
        # Simple reward function (length-based for demonstration)
        reward = min(1.0, len(inst['output'].split()) / 20.0)
        
        # Mock log probability and value
        log_prob = -2.0 + 0.5 * np.random.randn()
        value = reward + 0.1 * np.random.randn()
        
        experiences.append(PPOExperience(
            query=inst['instruction'],
            response=inst['output'],
            reward=reward,
            log_prob=log_prob,
            value=value
        ))
    
    # Convert math problems
    for i, problem in enumerate(data['math_problems']):
        reward = 0.8 + 0.2 * (i % 3) / 3  # Simulate varying quality
        log_prob = -1.5 + 0.5 * np.random.randn()
        value = reward + 0.1 * np.random.randn()
        
        experiences.append(PPOExperience(
            query=problem['problem'],
            response=problem['solution'],
            reward=reward,
            log_prob=log_prob,
            value=value
        ))
    
    # Convert reasoning tasks
    for i, task in enumerate(data['reasoning_tasks']):
        reward = 0.7 + 0.3 * (i % 4) / 4  # Simulate varying quality
        log_prob = -1.8 + 0.5 * np.random.randn()
        value = reward + 0.1 * np.random.randn()
        
        experiences.append(PPOExperience(
            query=task['question'],
            response=task['answer'],
            reward=reward,
            log_prob=log_prob,
            value=value
        ))
    
    # Compute advantages using GAE
    rewards = torch.tensor([exp.reward for exp in experiences])
    values = torch.tensor([exp.value for exp in experiences])
    
    # Simple advantage calculation (reward - value)
    advantages = rewards - values
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Update experiences with advantages
    for i, exp in enumerate(experiences):
        exp.advantage = advantages[i].item()
    
    return experiences

def demonstrate_ppo():
    """Demonstrate PPO with examples"""
    print_colored("=== Proximal Policy Optimization (PPO) Demonstration ===", "green")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Prepare data
    print_colored("Preparing PPO data...", "cyan")
    experiences = prepare_ppo_data()
    print(f"Created {len(experiences)} PPO experiences")
    
    # Show example experiences
    print_colored("\nSample PPO experiences:", "yellow")
    for i, exp in enumerate(experiences[:2]):
        print(f"\n{i+1}. Query: {exp.query}")
        print(f"   Response: {exp.response}")
        print(f"   Reward: {exp.reward:.3f}")
        print(f"   Log Prob: {exp.log_prob:.3f}")
        print(f"   Value: {exp.value:.3f}")
        print(f"   Advantage: {exp.advantage:.3f}")
    
    # Create dataset
    dataset = PPODataset(experiences)
    
    # Create models
    actor = PPOActor(vocab_size=len(dataset.word_to_id))
    critic = PPOCritic(vocab_size=len(dataset.word_to_id))
    
    # Test different clip ratios
    clip_ratios = [0.1, 0.2, 0.3]
    results = {}
    
    for clip_ratio in clip_ratios:
        print_colored(f"\nTraining PPO with clip_ratio={clip_ratio}...", "cyan")
        
        # Create fresh models for each clip ratio
        current_actor = PPOActor(vocab_size=len(dataset.word_to_id))
        current_critic = PPOCritic(vocab_size=len(dataset.word_to_id))
        
        # Create trainer
        trainer = PPOTrainer(current_actor, current_critic, dataset, clip_ratio=clip_ratio)
        
        # Train
        history = trainer.train(num_epochs=5, batch_size=2)
        results[clip_ratio] = history
        
        # Test response generation
        print_colored(f"Testing response generation (clip_ratio={clip_ratio}):", "yellow")
        test_query = "What is the capital of France?"
        response, log_prob, value = trainer.generate_response(test_query)
        print(f"Query: {test_query}")
        print(f"Response: {response}")
        print(f"Log Prob: {log_prob:.3f}")
        print(f"Value: {value:.3f}")
        print()
    
    # Plot comparison of different clip ratios
    print_colored("Plotting training comparison...", "cyan")
    
    plt.figure(figsize=(15, 12))
    
    # Actor Loss comparison
    plt.subplot(2, 3, 1)
    for clip_ratio in clip_ratios:
        plt.plot(results[clip_ratio]['actor_loss'], label=f'clip={clip_ratio}')
    plt.title('Actor Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Actor Loss')
    plt.legend()
    plt.grid(True)
    
    # Critic Loss comparison
    plt.subplot(2, 3, 2)
    for clip_ratio in clip_ratios:
        plt.plot(results[clip_ratio]['critic_loss'], label=f'clip={clip_ratio}')
    plt.title('Critic Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Critic Loss')
    plt.legend()
    plt.grid(True)
    
    # Clip Fraction comparison
    plt.subplot(2, 3, 3)
    for clip_ratio in clip_ratios:
        plt.plot(results[clip_ratio]['clip_fraction'], label=f'clip={clip_ratio}')
    plt.title('Clip Fraction vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Clip Fraction')
    plt.legend()
    plt.grid(True)
    
    # KL Divergence comparison
    plt.subplot(2, 3, 4)
    for clip_ratio in clip_ratios:
        plt.plot(results[clip_ratio]['kl_divergence'], label=f'clip={clip_ratio}')
    plt.title('KL Divergence vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.legend()
    plt.grid(True)
    
    # Entropy comparison
    plt.subplot(2, 3, 5)
    for clip_ratio in clip_ratios:
        plt.plot(results[clip_ratio]['entropy'], label=f'clip={clip_ratio}')
    plt.title('Entropy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Entropy')
    plt.legend()
    plt.grid(True)
    
    # Explained Variance comparison
    plt.subplot(2, 3, 6)
    for clip_ratio in clip_ratios:
        plt.plot(results[clip_ratio]['explained_variance'], label=f'clip={clip_ratio}')
    plt.title('Explained Variance vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Explained Variance')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ppo_clip_ratio_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print_colored("\nPPO demonstration completed!", "green")
    print("Key insights:")
    print("1. PPO uses clipped surrogate objective to prevent large policy updates")
    print("2. Actor-Critic architecture combines policy and value function learning")
    print("3. Clip ratio controls the aggressiveness of policy updates")
    print("4. Entropy regularization encourages exploration")
    print("5. Explained variance measures how well the critic predicts returns")

if __name__ == "__main__":
    demonstrate_ppo() 