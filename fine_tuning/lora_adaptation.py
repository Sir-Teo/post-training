"""
LoRA (Low-Rank Adaptation) Implementation

This module demonstrates LoRA, a parameter-efficient fine-tuning technique that
adapts large models by learning low-rank decomposition matrices while keeping
the original weights frozen.

Key Concepts:
- Parameter efficiency (only a small fraction of parameters are trained)
- Low-rank matrix decomposition
- Minimal impact on inference speed
- Preservation of original model capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import (
    print_section_header, print_subsection_header, print_colored, Colors,
    Timer, calculate_metrics, set_random_seed, create_embeddings
)
from shared_data.sample_data import get_sample_data

class LoRALayer:
    """
    LoRA layer implementation that adds low-rank adaptation to existing weights.
    
    For a weight matrix W, LoRA learns two smaller matrices A and B such that:
    W_adapted = W + B @ A
    where A is (d, r) and B is (r, d) with r << d (rank)
    """
    
    def __init__(self, input_dim: int, output_dim: int, rank: int = 4, alpha: float = 16.0):
        """
        Initialize LoRA layer.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            rank: Rank of the low-rank decomposition
            alpha: Scaling factor for LoRA weights
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Initialize LoRA matrices
        # A is initialized with random Gaussian, B with zeros
        self.A = np.random.normal(0, 0.02, (input_dim, rank))
        self.B = np.zeros((rank, output_dim))
        
        # Gradients
        self.dA = np.zeros_like(self.A)
        self.dB = np.zeros_like(self.B)
        
        print_colored(f"LoRA layer initialized: {input_dim}×{output_dim}, rank={rank}", Colors.GREEN)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through LoRA adaptation."""
        # Compute LoRA adaptation: x @ A @ B
        return x @ self.A @ self.B * self.scaling
    
    def backward(self, x: np.ndarray, grad_output: np.ndarray):
        """Backward pass to compute gradients."""
        # Gradients for B: A^T @ x^T @ grad_output
        self.dB = self.A.T @ x.T @ grad_output * self.scaling
        # Gradients for A: x^T @ grad_output @ B^T
        self.dA = x.T @ grad_output @ self.B.T * self.scaling
    
    def update_weights(self, learning_rate: float):
        """Update LoRA weights."""
        self.A -= learning_rate * self.dA
        self.B -= learning_rate * self.dB
    
    def get_parameter_count(self) -> int:
        """Get number of trainable parameters."""
        return self.A.size + self.B.size

class LoRAModel:
    """
    Simple model with LoRA adaptations.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, rank: int = 4):
        """
        Initialize model with LoRA layers.
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension
            rank: LoRA rank
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rank = rank
        
        # Frozen base weights (simulate pre-trained weights)
        self.W1_base = np.random.randn(input_size, hidden_size) * 0.1
        self.b1_base = np.zeros((1, hidden_size))
        self.W2_base = np.random.randn(hidden_size, output_size) * 0.1
        self.b2_base = np.zeros((1, output_size))
        
        # LoRA adaptations (only these will be trained)
        self.lora1 = LoRALayer(input_size, hidden_size, rank)
        self.lora2 = LoRALayer(hidden_size, output_size, rank)
        
        # Store intermediate activations for backprop
        self.z1 = None
        self.a1 = None
        self.z2 = None
        
        total_base_params = (self.W1_base.size + self.b1_base.size + 
                           self.W2_base.size + self.b2_base.size)
        lora_params = self.lora1.get_parameter_count() + self.lora2.get_parameter_count()
        
        print_colored(f"Base model parameters (frozen): {total_base_params}", Colors.BLUE)
        print_colored(f"LoRA parameters (trainable): {lora_params}", Colors.GREEN)
        print_colored(f"Parameter efficiency: {lora_params/total_base_params:.3%}", Colors.YELLOW)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the model."""
        # First layer: base + LoRA
        z1_base = np.dot(x, self.W1_base) + self.b1_base
        z1_lora = self.lora1.forward(x)
        self.z1 = z1_base + z1_lora
        self.a1 = np.tanh(self.z1)
        
        # Second layer: base + LoRA
        z2_base = np.dot(self.a1, self.W2_base) + self.b2_base
        z2_lora = self.lora2.forward(self.a1)
        self.z2 = z2_base + z2_lora
        
        return self.sigmoid(self.z2)
    
    def backward(self, x: np.ndarray, y: np.ndarray, output: np.ndarray):
        """Backward pass (only updates LoRA parameters)."""
        m = x.shape[0]
        
        # Output layer gradients
        dz2 = (output - y) / m
        
        # LoRA2 gradients
        self.lora2.backward(self.a1, dz2)
        
        # Hidden layer gradients
        # Only consider gradients flowing through LoRA (base weights are frozen)
        da1 = dz2 @ self.lora2.B.T @ self.lora2.A.T * self.lora2.scaling
        dz1 = da1 * (1 - np.tanh(self.z1)**2)
        
        # LoRA1 gradients
        self.lora1.backward(x, dz1)
    
    def update_weights(self, learning_rate: float):
        """Update only LoRA weights."""
        self.lora1.update_weights(learning_rate)
        self.lora2.update_weights(learning_rate)
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def get_adaptation_magnitude(self) -> Dict[str, float]:
        """Compute the magnitude of LoRA adaptations."""
        lora1_magnitude = np.linalg.norm(self.lora1.A @ self.lora1.B * self.lora1.scaling)
        lora2_magnitude = np.linalg.norm(self.lora2.A @ self.lora2.B * self.lora2.scaling)
        
        base1_magnitude = np.linalg.norm(self.W1_base)
        base2_magnitude = np.linalg.norm(self.W2_base)
        
        return {
            'lora1_magnitude': lora1_magnitude,
            'lora2_magnitude': lora2_magnitude,
            'base1_magnitude': base1_magnitude,
            'base2_magnitude': base2_magnitude,
            'adaptation_ratio_1': lora1_magnitude / base1_magnitude,
            'adaptation_ratio_2': lora2_magnitude / base2_magnitude
        }

class LoRAFineTuner:
    """
    LoRA fine-tuning implementation.
    """
    
    def __init__(self, model_config: Dict[str, int]):
        """Initialize LoRA fine-tuner."""
        self.model = LoRAModel(**model_config)
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'adaptation_ratios': []
        }
        
        print_colored("✓ LoRA Fine-tuner initialized", Colors.GREEN)
    
    def prepare_data(self, data_type: str = "instruction") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data."""
        print_subsection_header("Preparing Training Data")
        
        sample_data = get_sample_data()
        
        if data_type == "instruction":
            data = sample_data.instruction_data
            texts = [f"{item['instruction']} {item['input']}" for item in data]
            targets = [item['output'] for item in data]
        elif data_type == "math":
            data = sample_data.math_problems
            texts = [item['question'] for item in data]
            targets = [item['answer'] for item in data]
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # Create embeddings
        X_train, vocab = create_embeddings(texts, vocab_size=self.model.input_size)
        
        # Create binary labels
        y_train = np.random.randint(0, 2, (len(texts), 1)).astype(float)
        
        print_colored(f"Data prepared: {X_train.shape[0]} samples", Colors.GREEN)
        return X_train, y_train, texts
    
    def fine_tune(self, X_train: np.ndarray, y_train: np.ndarray,
                  epochs: int = 50, learning_rate: float = 0.01) -> Dict[str, Any]:
        """Perform LoRA fine-tuning."""
        print_subsection_header("LoRA Fine-tuning Process")
        
        losses = []
        accuracies = []
        adaptation_ratios = []
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self.model.forward(X_train)
            
            # Compute loss
            loss = self.model.compute_loss(y_train, predictions)
            
            # Backward pass (only updates LoRA parameters)
            self.model.backward(X_train, y_train, predictions)
            
            # Update weights
            self.model.update_weights(learning_rate)
            
            # Calculate accuracy
            predicted_labels = (predictions > 0.5).astype(int)
            true_labels = y_train.astype(int)
            accuracy = np.mean(predicted_labels == true_labels)
            
            # Track adaptation magnitude
            adaptation_info = self.model.get_adaptation_magnitude()
            avg_adaptation_ratio = (adaptation_info['adaptation_ratio_1'] + 
                                  adaptation_info['adaptation_ratio_2']) / 2
            
            # Store metrics
            losses.append(loss)
            accuracies.append(accuracy)
            adaptation_ratios.append(avg_adaptation_ratio)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print_colored(
                    f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, "
                    f"Accuracy: {accuracy:.4f}, Adaptation Ratio: {avg_adaptation_ratio:.4f}",
                    Colors.BLUE
                )
        
        # Store training history
        self.training_history['losses'] = losses
        self.training_history['accuracies'] = accuracies
        self.training_history['adaptation_ratios'] = adaptation_ratios
        
        final_accuracy = accuracies[-1]
        print_colored(f"LoRA fine-tuning completed! Final accuracy: {final_accuracy:.4f}", Colors.GREEN)
        
        return {
            'final_accuracy': final_accuracy,
            'final_loss': losses[-1],
            'final_adaptation_ratio': adaptation_ratios[-1],
            'training_history': self.training_history
        }
    
    def analyze_adaptations(self):
        """Analyze the learned LoRA adaptations."""
        print_subsection_header("LoRA Adaptation Analysis")
        
        adaptation_info = self.model.get_adaptation_magnitude()
        
        print_colored("Adaptation Magnitudes:", Colors.BLUE, bold=True)
        print(f"  Layer 1 - Base: {adaptation_info['base1_magnitude']:.4f}, "
              f"LoRA: {adaptation_info['lora1_magnitude']:.4f}")
        print(f"  Layer 2 - Base: {adaptation_info['base2_magnitude']:.4f}, "
              f"LoRA: {adaptation_info['lora2_magnitude']:.4f}")
        
        print_colored("Adaptation Ratios (LoRA/Base):", Colors.GREEN, bold=True)
        print(f"  Layer 1: {adaptation_info['adaptation_ratio_1']:.4f}")
        print(f"  Layer 2: {adaptation_info['adaptation_ratio_2']:.4f}")
        
        # Visualize LoRA matrices
        self.visualize_lora_matrices()
        
        return adaptation_info
    
    def visualize_lora_matrices(self):
        """Visualize the learned LoRA matrices."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # LoRA 1 matrices
        im1 = axes[0, 0].imshow(self.model.lora1.A.T, aspect='auto', cmap='RdBu')
        axes[0, 0].set_title('LoRA1 Matrix A')
        axes[0, 0].set_xlabel('Input Dimension')
        axes[0, 0].set_ylabel('Rank Dimension')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(self.model.lora1.B, aspect='auto', cmap='RdBu')
        axes[0, 1].set_title('LoRA1 Matrix B')
        axes[0, 1].set_xlabel('Output Dimension')
        axes[0, 1].set_ylabel('Rank Dimension')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # LoRA 2 matrices
        im3 = axes[1, 0].imshow(self.model.lora2.A.T, aspect='auto', cmap='RdBu')
        axes[1, 0].set_title('LoRA2 Matrix A')
        axes[1, 0].set_xlabel('Input Dimension')
        axes[1, 0].set_ylabel('Rank Dimension')
        plt.colorbar(im3, ax=axes[1, 0])
        
        im4 = axes[1, 1].imshow(self.model.lora2.B, aspect='auto', cmap='RdBu')
        axes[1, 1].set_title('LoRA2 Matrix B')
        axes[1, 1].set_xlabel('Output Dimension')
        axes[1, 1].set_ylabel('Rank Dimension')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_progress(self):
        """Plot training curves including adaptation ratios."""
        if not self.training_history['losses']:
            print_colored("No training history to plot", Colors.RED)
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        epochs = range(1, len(self.training_history['losses']) + 1)
        
        # Plot loss
        axes[0].plot(epochs, self.training_history['losses'], 'r-', label='Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        axes[1].plot(epochs, self.training_history['accuracies'], 'g-', label='Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot adaptation ratios
        axes[2].plot(epochs, self.training_history['adaptation_ratios'], 'b-', label='Adaptation Ratio')
        axes[2].set_title('LoRA Adaptation Ratio')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Adaptation Ratio')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()

def demonstrate_lora_adaptation():
    """Main demonstration function for LoRA adaptation."""
    
    print_section_header("LoRA (Low-Rank Adaptation) Demonstration")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Test different ranks
    ranks_to_test = [2, 4, 8]
    results = {}
    
    for rank in ranks_to_test:
        print_subsection_header(f"Testing LoRA with Rank {rank}")
        
        # Initialize LoRA fine-tuner
        model_config = {
            'input_size': 40,
            'hidden_size': 20,
            'output_size': 1,
            'rank': rank
        }
        
        lora_tuner = LoRAFineTuner(model_config)
        
        # Prepare training data
        X_train, y_train, texts = lora_tuner.prepare_data("instruction")
        
        # Perform fine-tuning
        with Timer(f"LoRA fine-tuning (rank={rank})"):
            training_results = lora_tuner.fine_tune(
                X_train, y_train,
                epochs=30,
                learning_rate=0.01
            )
        
        # Analyze adaptations
        adaptation_info = lora_tuner.analyze_adaptations()
        
        # Store results
        results[f"LoRA_rank_{rank}"] = {
            'method': f'LoRA (rank={rank})',
            'final_accuracy': training_results['final_accuracy'],
            'adaptation_ratio': training_results['final_adaptation_ratio'],
            'trainable_params': (lora_tuner.model.lora1.get_parameter_count() + 
                               lora_tuner.model.lora2.get_parameter_count()),
            'metrics': {
                'accuracy': training_results['final_accuracy'],
                'exact_matches': int(training_results['final_accuracy'] * len(X_train)),
                'total_samples': len(X_train),
                'avg_token_overlap': 0.8  # Simulated
            }
        }
    
    # Plot comparison of different ranks
    plot_rank_comparison(results)
    
    # Display final summary for the best rank
    best_rank = max(results.keys(), key=lambda k: results[k]['final_accuracy'])
    best_result = results[best_rank]
    
    print_subsection_header("Summary")
    print_colored("Key Benefits of LoRA:", Colors.BLUE, bold=True)
    print("• Extremely parameter-efficient (typically <1% of original parameters)")
    print("• Preserves original model capabilities")
    print("• Fast training and minimal memory overhead")
    print("• Easy to merge/swap different adaptations")
    print("• No additional inference latency")
    
    print_colored("\nKey Features:", Colors.GREEN, bold=True)
    print("• Low-rank matrix decomposition")
    print("• Frozen backbone with trainable adaptation matrices")
    print("• Configurable rank for efficiency/performance trade-off")
    print("• Suitable for multiple task adaptation")
    
    print_colored("\nUse Cases:", Colors.YELLOW, bold=True)
    print("• Resource-constrained environments")
    print("• Multiple task adaptation")
    print("• Quick domain adaptation")
    print("• Personalization without full fine-tuning")
    
    return results

def plot_rank_comparison(results: Dict[str, Dict[str, Any]]):
    """Plot comparison of different LoRA ranks."""
    ranks = []
    accuracies = []
    param_counts = []
    
    for key, result in results.items():
        rank = int(key.split('_')[-1])
        ranks.append(rank)
        accuracies.append(result['final_accuracy'])
        param_counts.append(result['trainable_params'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy vs Rank
    ax1.plot(ranks, accuracies, 'bo-', linewidth=2, markersize=8)
    ax1.set_title('LoRA Performance vs Rank')
    ax1.set_xlabel('LoRA Rank')
    ax1.set_ylabel('Final Accuracy')
    ax1.grid(True)
    
    # Parameter Count vs Rank
    ax2.plot(ranks, param_counts, 'ro-', linewidth=2, markersize=8)
    ax2.set_title('Trainable Parameters vs Rank')
    ax2.set_xlabel('LoRA Rank')
    ax2.set_ylabel('Trainable Parameters')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run the demonstration
    results = demonstrate_lora_adaptation()
    print_colored(f"\nLoRA demonstration completed successfully!", Colors.GREEN, bold=True) 