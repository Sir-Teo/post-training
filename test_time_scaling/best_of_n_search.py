"""
Best-of-N Search Implementation

This module demonstrates Best-of-N search, a test-time scaling method that generates
multiple responses for a given query and selects the best one based on a reward model
or heuristic. This is also known as rejection sampling.

Key concepts:
- Multiple response generation
- Response scoring and ranking
- Reward model integration
- Quality vs computational cost trade-off
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Callable
import random
from dataclasses import dataclass
import time
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
class GeneratedResponse:
    """Data structure for generated responses"""
    query: str
    response: str
    score: float
    generation_time: float
    tokens: List[int]
    log_prob: float

class SimpleGenerator(nn.Module):
    """Simple language model for response generation"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Language model
        self.lm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=2)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, tokens: torch.Tensor, temperature: float = 1.0):
        """
        Forward pass for generation
        
        Args:
            tokens: Input tokens
            temperature: Sampling temperature
        """
        # Embed tokens
        embedded = self.embedding(tokens)
        embedded = self.dropout(embedded)
        
        # Pass through language model
        output, _ = self.lm(embedded)
        
        # Get logits
        logits = self.output_layer(output)
        
        # Apply temperature
        logits = logits / temperature
        
        return logits

class SimpleRewardModel(nn.Module):
    """Simple reward model for scoring responses"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Encoder
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=2)
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, tokens: torch.Tensor):
        """
        Forward pass for reward estimation
        
        Args:
            tokens: Input tokens (query + response)
        """
        # Embed tokens
        embedded = self.embedding(tokens)
        
        # Encode
        encoded, _ = self.encoder(embedded)
        
        # Use final hidden state
        final_state = encoded[:, -1, :]
        
        # Get reward
        reward = self.reward_head(final_state)
        
        return reward.squeeze(-1)

class BestOfNSearcher:
    """Main class for Best-of-N search"""
    
    def __init__(self, generator: SimpleGenerator, reward_model: SimpleRewardModel, 
                 vocab_size: int, tokenizer_vocab_size: int = 1000):
        self.generator = generator
        self.reward_model = reward_model
        self.vocab_size = vocab_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        self.generator.to(self.device)
        self.reward_model.to(self.device)
        
        # Simple tokenization for demonstration
        self.word_to_id = {}
        self.id_to_word = {}
        self._build_vocab(tokenizer_vocab_size)
        
        # Set models to evaluation mode
        self.generator.eval()
        self.reward_model.eval()
    
    def _build_vocab(self, vocab_size: int):
        """Build vocabulary for tokenization"""
        # Create a simple vocabulary
        words = ['<pad>', '<unk>', '<start>', '<end>', '<sep>']
        words.extend([f'word_{i}' for i in range(vocab_size - 5)])
        
        self.word_to_id = {word: i for i, word in enumerate(words)}
        self.id_to_word = {i: word for i, word in enumerate(words)}
    
    def _tokenize(self, text: str) -> List[int]:
        """Simple tokenization"""
        tokens = text.lower().split()
        return [self.word_to_id.get(token, 1) for token in tokens]  # 1 is <unk>
    
    def _detokenize(self, tokens: List[int]) -> str:
        """Convert tokens back to text"""
        words = [self.id_to_word.get(token, '<unk>') for token in tokens]
        return ' '.join(words).replace('<pad>', '').strip()
    
    def generate_response(self, query: str, max_length: int = 32, 
                         temperature: float = 1.0, top_k: int = 50) -> GeneratedResponse:
        """Generate a single response for a query"""
        start_time = time.time()
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        current_tokens = query_tokens + [4]  # Add separator
        
        # Generate response
        generated_tokens = []
        total_log_prob = 0.0
        
        with torch.no_grad():
            for _ in range(max_length):
                # Convert to tensor
                input_tensor = torch.tensor(current_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
                
                # Get logits
                logits = self.generator(input_tensor, temperature=temperature)
                next_logits = logits[0, -1, :]
                
                # Apply top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
                    probs = F.softmax(top_k_logits, dim=-1)
                    
                    # Sample from top-k
                    next_token_idx = torch.multinomial(probs, 1).item()
                    next_token = top_k_indices[next_token_idx].item()
                else:
                    # Sample from full distribution
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                
                # Stop conditions
                if next_token == 0 or next_token == 3:  # <pad> or <end>
                    break
                
                # Add to sequence
                current_tokens.append(next_token)
                generated_tokens.append(next_token)
                
                # Track log probability
                log_prob = F.log_softmax(next_logits, dim=-1)[next_token]
                total_log_prob += log_prob.item()
        
        # Convert to text
        response = self._detokenize(generated_tokens)
        
        # Score the response
        full_tokens = query_tokens + [4] + generated_tokens
        score = self._score_response(full_tokens)
        
        generation_time = time.time() - start_time
        
        return GeneratedResponse(
            query=query,
            response=response,
            score=score,
            generation_time=generation_time,
            tokens=generated_tokens,
            log_prob=total_log_prob
        )
    
    def _score_response(self, tokens: List[int]) -> float:
        """Score a response using the reward model"""
        # Pad tokens
        padded_tokens = tokens[:128]  # Truncate if too long
        padded_tokens += [0] * (128 - len(padded_tokens))  # Pad
        
        # Convert to tensor
        token_tensor = torch.tensor(padded_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Get reward
        with torch.no_grad():
            reward = self.reward_model(token_tensor)
        
        return reward.item()
    
    def best_of_n_search(self, query: str, n: int = 5, temperature: float = 1.0, 
                        top_k: int = 50) -> Tuple[GeneratedResponse, List[GeneratedResponse]]:
        """
        Perform Best-of-N search
        
        Args:
            query: Input query
            n: Number of responses to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
        
        Returns:
            Tuple of (best_response, all_responses)
        """
        print_colored(f"Generating {n} responses for query: {query}", "cyan")
        
        responses = []
        
        for i in range(n):
            response = self.generate_response(query, temperature=temperature, top_k=top_k)
            responses.append(response)
            print(f"Response {i+1}: {response.response} (Score: {response.score:.3f})")
        
        # Sort by score (descending)
        responses.sort(key=lambda x: x.score, reverse=True)
        best_response = responses[0]
        
        print_colored(f"Best response: {best_response.response} (Score: {best_response.score:.3f})", "green")
        
        return best_response, responses
    
    def analyze_scaling_behavior(self, queries: List[str], n_values: List[int], 
                               num_trials: int = 3) -> Dict[str, List[float]]:
        """Analyze how performance scales with N"""
        results = {
            'n_values': n_values,
            'avg_best_score': [],
            'avg_worst_score': [],
            'score_variance': [],
            'total_time': [],
            'success_rate': []
        }
        
        for n in n_values:
            print_colored(f"Analyzing N={n}...", "cyan")
            
            trial_best_scores = []
            trial_worst_scores = []
            trial_times = []
            trial_success = []
            
            for trial in range(num_trials):
                for query in queries:
                    start_time = time.time()
                    
                    # Perform Best-of-N search
                    best_response, all_responses = self.best_of_n_search(
                        query, n=n, temperature=0.8
                    )
                    
                    # Track metrics
                    scores = [resp.score for resp in all_responses]
                    trial_best_scores.append(max(scores))
                    trial_worst_scores.append(min(scores))
                    trial_times.append(time.time() - start_time)
                    trial_success.append(1 if best_response.score > 0.5 else 0)
            
            # Average across trials
            results['avg_best_score'].append(np.mean(trial_best_scores))
            results['avg_worst_score'].append(np.mean(trial_worst_scores))
            results['score_variance'].append(np.var(trial_best_scores))
            results['total_time'].append(np.mean(trial_times))
            results['success_rate'].append(np.mean(trial_success))
        
        return results

def create_heuristic_reward_model(vocab_size: int) -> SimpleRewardModel:
    """Create a simple heuristic reward model"""
    model = SimpleRewardModel(vocab_size)
    
    # Initialize with some reasonable weights for demonstration
    with torch.no_grad():
        # Make longer responses slightly preferred
        model.reward_head[0].weight.data.fill_(0.1)
        model.reward_head[0].bias.data.fill_(0.5)
        model.reward_head[3].weight.data.fill_(0.1)
        model.reward_head[3].bias.data.fill_(0.0)
    
    return model

def demonstrate_best_of_n_search():
    """Demonstrate Best-of-N search with examples"""
    print_colored("=== Best-of-N Search Demonstration ===", "green")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Create models
    vocab_size = 1000
    generator = SimpleGenerator(vocab_size)
    reward_model = create_heuristic_reward_model(vocab_size)
    
    # Create Best-of-N searcher
    searcher = BestOfNSearcher(generator, reward_model, vocab_size)
    
    # Test queries
    test_queries = [
        "What is the capital of France?",
        "How do you solve a quadratic equation?",
        "Explain machine learning in simple terms.",
        "What are the benefits of renewable energy?"
    ]
    
    print_colored("Testing individual queries...", "cyan")
    
    # Test each query with different N values
    for query in test_queries[:2]:  # Limit to 2 queries for demo
        print_colored(f"\nTesting query: {query}", "yellow")
        
        # Test with different N values
        for n in [1, 3, 5]:
            print_colored(f"\nN = {n}:", "magenta")
            best_response, all_responses = searcher.best_of_n_search(query, n=n)
            
            # Show score distribution
            scores = [resp.score for resp in all_responses]
            print(f"Score range: {min(scores):.3f} - {max(scores):.3f}")
            print(f"Score std: {np.std(scores):.3f}")
            print()
    
    # Analyze scaling behavior
    print_colored("Analyzing scaling behavior...", "cyan")
    
    scaling_results = searcher.analyze_scaling_behavior(
        queries=test_queries[:2],  # Use fewer queries for demo
        n_values=[1, 2, 3, 5, 8],
        num_trials=2
    )
    
    # Plot scaling analysis
    print_colored("Plotting scaling analysis...", "cyan")
    
    plt.figure(figsize=(15, 10))
    
    # Best score vs N
    plt.subplot(2, 2, 1)
    plt.plot(scaling_results['n_values'], scaling_results['avg_best_score'], 'b-o', label='Best Score')
    plt.plot(scaling_results['n_values'], scaling_results['avg_worst_score'], 'r-o', label='Worst Score')
    plt.title('Score vs N')
    plt.xlabel('N (Number of Generations)')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Score variance vs N
    plt.subplot(2, 2, 2)
    plt.plot(scaling_results['n_values'], scaling_results['score_variance'], 'g-o')
    plt.title('Score Variance vs N')
    plt.xlabel('N (Number of Generations)')
    plt.ylabel('Score Variance')
    plt.grid(True)
    
    # Total time vs N
    plt.subplot(2, 2, 3)
    plt.plot(scaling_results['n_values'], scaling_results['total_time'], 'm-o')
    plt.title('Total Time vs N')
    plt.xlabel('N (Number of Generations)')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    
    # Success rate vs N
    plt.subplot(2, 2, 4)
    plt.plot(scaling_results['n_values'], scaling_results['success_rate'], 'c-o')
    plt.title('Success Rate vs N')
    plt.xlabel('N (Number of Generations)')
    plt.ylabel('Success Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('best_of_n_scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Demonstrate quality vs computational cost trade-off
    print_colored("Quality vs Computational Cost Analysis:", "cyan")
    
    n_values = scaling_results['n_values']
    scores = scaling_results['avg_best_score']
    times = scaling_results['total_time']
    
    print(f"{'N':>3} | {'Avg Best Score':>15} | {'Total Time':>12} | {'Score/Time':>12}")
    print("-" * 50)
    
    for i, n in enumerate(n_values):
        score_per_time = scores[i] / times[i] if times[i] > 0 else 0
        print(f"{n:>3} | {scores[i]:>15.3f} | {times[i]:>12.3f} | {score_per_time:>12.3f}")
    
    print_colored("\nBest-of-N search demonstration completed!", "green")
    print("Key insights:")
    print("1. Best-of-N search improves response quality by generating multiple candidates")
    print("2. Performance typically improves with larger N, but with diminishing returns")
    print("3. There's a trade-off between quality and computational cost")
    print("4. The reward model is crucial for effective candidate selection")
    print("5. Temperature and top-k sampling affect the diversity of generated responses")

if __name__ == "__main__":
    demonstrate_best_of_n_search() 