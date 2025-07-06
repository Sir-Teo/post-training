"""
Common utilities for post-training method demonstrations.
This module provides shared functions and classes used across all implementations.
"""

import json
import pickle
import random
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Colors:
    """ANSI color codes for terminal output."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # If torch is available, set torch seed too
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def print_colored(text: str, color: str = Colors.BLUE, bold: bool = False):
    """Print colored text to terminal."""
    if bold:
        print(f"{Colors.BOLD}{color}{text}{Colors.END}")
    else:
        print(f"{color}{text}{Colors.END}")

def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print_colored(f"  {title}", Colors.BLUE, bold=True)
    print("="*60)

def print_subsection_header(title: str):
    """Print a formatted subsection header."""
    print("\n" + "-"*40)
    print_colored(f"  {title}", Colors.GREEN, bold=True)
    print("-"*40)

def save_results(results: Dict[str, Any], filename: str, directory: str = "results"):
    """Save results to JSON file."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    filepath = Path(directory) / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print_colored(f"Results saved to {filepath}", Colors.GREEN)

def load_results(filename: str, directory: str = "results") -> Dict[str, Any]:
    """Load results from JSON file."""
    filepath = Path(directory) / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results

def calculate_metrics(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """Calculate basic evaluation metrics."""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    # Exact match accuracy
    exact_matches = sum(1 for p, t in zip(predictions, targets) if p.strip() == t.strip())
    accuracy = exact_matches / len(targets)
    
    # Token-level overlap (simplified)
    token_overlaps = []
    for p, t in zip(predictions, targets):
        p_tokens = set(p.lower().split())
        t_tokens = set(t.lower().split())
        if len(t_tokens) > 0:
            overlap = len(p_tokens.intersection(t_tokens)) / len(t_tokens)
            token_overlaps.append(overlap)
    
    avg_token_overlap = np.mean(token_overlaps) if token_overlaps else 0.0
    
    return {
        "accuracy": accuracy,
        "exact_matches": exact_matches,
        "total_samples": len(targets),
        "avg_token_overlap": avg_token_overlap
    }

def plot_training_curves(losses: List[float], rewards: Optional[List[float]] = None, 
                        save_path: Optional[str] = None):
    """Plot training curves for loss and rewards."""
    fig, axes = plt.subplots(1, 2 if rewards else 1, figsize=(12, 5))
    
    if rewards:
        axes[0].plot(losses, label='Loss', color='red')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        axes[1].plot(rewards, label='Reward', color='green')
        axes[1].set_title('Training Reward')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Reward')
        axes[1].legend()
    else:
        if isinstance(axes, np.ndarray):
            ax = axes[0]
        else:
            ax = axes
        ax.plot(losses, label='Loss', color='red')
        ax.set_title('Training Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_colored(f"Training curves saved to {save_path}", Colors.GREEN)
    
    plt.show()

def plot_training_progress(history: Dict[str, List[float]], title: str = "Training Progress", 
                         save_path: Optional[str] = None):
    """Plot training progress with multiple metrics."""
    num_metrics = len(history)
    cols = min(3, num_metrics)
    rows = (num_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if num_metrics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten() if num_metrics > 1 else axes
    
    for idx, (metric, values) in enumerate(history.items()):
        ax = axes_flat[idx] if num_metrics > 1 else axes_flat[0]
        ax.plot(values, label=metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_colored(f"Training progress plot saved to {save_path}", Colors.GREEN)
    
    plt.show()

def plot_comparison_results(results: Dict[str, Dict[str, float]], 
                          title: str = "Method Comparison",
                          save_path: Optional[str] = None):
    """Plot comparison results between different methods."""
    methods = list(results.keys())
    metrics = list(results[methods[0]].keys())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        values = [results[method][metric] for method in methods]
        axes[i].bar(methods, values, color=plt.cm.viridis(np.linspace(0, 1, len(methods))))
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_colored(f"Comparison plot saved to {save_path}", Colors.GREEN)
    
    plt.show()

def time_function(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print_colored(f"Starting {func.__name__}...", Colors.YELLOW)
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        duration = end_time - start_time
        print_colored(f"Completed {func.__name__} in {duration:.2f} seconds", Colors.GREEN)
        
        return result
    return wrapper

class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print_colored(f"Starting {self.description}...", Colors.YELLOW)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print_colored(f"Completed {self.description} in {duration:.2f} seconds", Colors.GREEN)

class SimpleModel:
    """Simple neural network model for demonstrations."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights randomly
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
        # Store gradients
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)  # Hidden layer activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)  # Output layer activation
        return self.a2
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray):
        """Backward pass to compute gradients."""
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = output - y
        self.dW2 = np.dot(self.a1.T, dz2) / m
        self.db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (1 - np.tanh(self.z1)**2)  # Derivative of tanh
        self.dW1 = np.dot(X.T, dz1) / m
        self.db1 = np.sum(dz1, axis=0, keepdims=True) / m
    
    def update_weights(self, learning_rate: float):
        """Update weights using gradients."""
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Prevent log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Prevent overflow
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get model parameters."""
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy()
        }
    
    def set_parameters(self, params: Dict[str, np.ndarray]):
        """Set model parameters."""
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()

def create_embeddings(texts: List[str], vocab_size: int = 1000) -> Tuple[np.ndarray, Dict[str, int]]:
    """Create simple bag-of-words embeddings for text data."""
    # Build vocabulary
    all_words = []
    for text in texts:
        words = text.lower().split()
        all_words.extend(words)
    
    word_counts = {}
    for word in all_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Select top words
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {word: i for i, (word, _) in enumerate(sorted_words[:vocab_size])}
    
    # Create embeddings
    embeddings = np.zeros((len(texts), vocab_size))
    for i, text in enumerate(texts):
        words = text.lower().split()
        for word in words:
            if word in vocab:
                embeddings[i, vocab[word]] += 1
    
    return embeddings, vocab

def demonstrate_method(method_name: str, method_func, *args, **kwargs):
    """Wrapper to demonstrate a post-training method."""
    print_section_header(f"Demonstrating {method_name}")
    
    with Timer(f"{method_name} demonstration"):
        results = method_func(*args, **kwargs)
    
    print_colored(f"\n{method_name} demonstration completed!", Colors.GREEN, bold=True)
    return results

def compare_methods(results: Dict[str, Dict[str, Any]], 
                   method_names: List[str] = None,
                   save_comparison: bool = True):
    """Compare results from multiple methods."""
    if method_names is None:
        method_names = list(results.keys())
    
    print_section_header("Method Comparison")
    
    # Extract metrics for comparison
    comparison_data = {}
    for method in method_names:
        if method in results:
            comparison_data[method] = results[method]
    
    # Print comparison table
    print("\nComparison Results:")
    print("-" * 80)
    
    # Get all metrics
    all_metrics = set()
    for method_results in comparison_data.values():
        if 'metrics' in method_results:
            all_metrics.update(method_results['metrics'].keys())
    
    # Print headers
    print(f"{'Method':<20} {'Accuracy':<12} {'Token Overlap':<15} {'Exact Matches':<15}")
    print("-" * 80)
    
    # Print results
    for method, method_results in comparison_data.items():
        metrics = method_results.get('metrics', {})
        accuracy = metrics.get('accuracy', 0.0)
        token_overlap = metrics.get('avg_token_overlap', 0.0)
        exact_matches = metrics.get('exact_matches', 0)
        
        print(f"{method:<20} {accuracy:<12.3f} {token_overlap:<15.3f} {exact_matches:<15}")
    
    # Plot comparison if there are numerical metrics
    if comparison_data:
        metric_data = {}
        for method, method_results in comparison_data.items():
            if 'metrics' in method_results:
                metric_data[method] = method_results['metrics']
        
        if metric_data:
            plot_comparison_results(metric_data, "Post-Training Methods Comparison")
    
    # Save comparison results
    if save_comparison:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(comparison_data, f"method_comparison_{timestamp}.json")
    
    return comparison_data

def create_progress_bar(total: int, description: str = "Progress"):
    """Create a simple progress bar."""
    def update_progress(current: int):
        percent = (current / total) * 100
        bar_length = 30
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f'\r{description}: |{bar}| {percent:.1f}% Complete', end='')
        if current == total:
            print()  # New line when complete
    
    return update_progress

# Example usage functions
def example_usage():
    """Show examples of how to use the utility functions."""
    print_section_header("Utility Functions Example")
    
    # Example data
    predictions = ["The answer is 42", "Hello world", "AI is fascinating"]
    targets = ["The answer is 42", "Hello, world!", "AI is very interesting"]
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, targets)
    print("\nExample Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    
    # Example embedding creation
    texts = ["hello world", "artificial intelligence", "machine learning"]
    embeddings, vocab = create_embeddings(texts, vocab_size=10)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Example model
    model = SimpleModel(input_size=10, hidden_size=5, output_size=1)
    X = np.random.randn(5, 10)
    output = model.forward(X)
    print(f"\nModel output shape: {output.shape}")

if __name__ == "__main__":
    example_usage() 