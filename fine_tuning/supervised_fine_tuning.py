"""
Supervised Fine-tuning (SFT) Implementation

This module demonstrates basic supervised fine-tuning where a pre-trained model
is adapted to specific tasks using curated datasets with input-output pairs.

Key Concepts:
- Task-specific adaptation
- Gradient-based optimization
- Overfitting prevention
- Performance evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import random
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import (
    print_section_header, print_subsection_header, print_colored, Colors,
    Timer, SimpleModel, calculate_metrics, plot_training_curves,
    demonstrate_method, set_random_seed, create_embeddings
)
from shared_data.sample_data import get_sample_data

class SupervisedFineTuner:
    """
    Implements supervised fine-tuning for language models.
    
    This is a simplified demonstration that shows the core concepts:
    1. Loading pre-trained model weights
    2. Adapting to task-specific data
    3. Preventing catastrophic forgetting
    4. Evaluating performance
    """
    
    def __init__(self, model_config: Dict[str, int]):
        """
        Initialize the fine-tuner.
        
        Args:
            model_config: Dictionary with model architecture parameters
        """
        self.input_size = model_config.get('input_size', 100)
        self.hidden_size = model_config.get('hidden_size', 50)
        self.output_size = model_config.get('output_size', 1)
        
        # Initialize model (simulating pre-trained weights)
        self.model = SimpleModel(self.input_size, self.hidden_size, self.output_size)
        self.original_params = self.model.get_parameters()  # Store original weights
        
        # Training history
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'epochs': []
        }
        
        print_colored("✓ Supervised Fine-tuner initialized", Colors.GREEN)
    
    def prepare_data(self, data_type: str = "instruction") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data for fine-tuning.
        
        Args:
            data_type: Type of data to use ('instruction', 'math', 'code')
            
        Returns:
            Tuple of (X_train, y_train, texts)
        """
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
        elif data_type == "code":
            data = sample_data.code_problems
            texts = [item['problem'] for item in data]
            targets = [item['solution'] for item in data]
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # Create embeddings
        X_train, vocab = create_embeddings(texts, vocab_size=self.input_size)
        
        # Create binary labels (simplified for demonstration)
        # In practice, this would be more sophisticated
        y_train = np.random.randint(0, 2, (len(texts), 1)).astype(float)
        
        print_colored(f"Data prepared: {X_train.shape[0]} samples", Colors.GREEN)
        print_colored(f"Vocabulary size: {len(vocab)}", Colors.BLUE)
        
        return X_train, y_train, texts
    
    def fine_tune(self, X_train: np.ndarray, y_train: np.ndarray, 
                  epochs: int = 50, learning_rate: float = 0.01,
                  regularization_strength: float = 0.001) -> Dict[str, Any]:
        """
        Perform supervised fine-tuning.
        
        Args:
            X_train: Training input features
            y_train: Training target labels
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            regularization_strength: L2 regularization strength
            
        Returns:
            Dictionary containing training results
        """
        print_subsection_header("Fine-tuning Process")
        
        n_samples = X_train.shape[0]
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self.model.forward(X_train)
            
            # Compute loss with L2 regularization
            base_loss = self.model.compute_loss(y_train, predictions)
            
            # Add L2 regularization to prevent catastrophic forgetting
            l2_penalty = 0
            current_params = self.model.get_parameters()
            for key in current_params:
                param_diff = current_params[key] - self.original_params[key]
                l2_penalty += np.sum(param_diff ** 2)
            
            total_loss = base_loss + regularization_strength * l2_penalty
            
            # Backward pass
            self.model.backward(X_train, y_train, predictions)
            
            # Update weights
            self.model.update_weights(learning_rate)
            
            # Calculate accuracy
            predicted_labels = (predictions > 0.5).astype(int)
            true_labels = y_train.astype(int)
            accuracy = np.mean(predicted_labels == true_labels)
            
            # Store metrics
            losses.append(total_loss)
            accuracies.append(accuracy)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print_colored(
                    f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}, "
                    f"Accuracy: {accuracy:.4f}, L2 Penalty: {l2_penalty:.4f}",
                    Colors.BLUE
                )
        
        # Store training history
        self.training_history['losses'] = losses
        self.training_history['accuracies'] = accuracies
        self.training_history['epochs'] = list(range(1, epochs + 1))
        
        final_accuracy = accuracies[-1]
        print_colored(f"Fine-tuning completed! Final accuracy: {final_accuracy:.4f}", Colors.GREEN)
        
        return {
            'final_accuracy': final_accuracy,
            'final_loss': losses[-1],
            'training_history': self.training_history
        }
    
    def evaluate_catastrophic_forgetting(self, original_data: np.ndarray) -> float:
        """
        Evaluate how much the model has forgotten original capabilities.
        
        Args:
            original_data: Original data to test retention
            
        Returns:
            Forgetting score (lower is better)
        """
        print_subsection_header("Evaluating Catastrophic Forgetting")
        
        # Test on original data with current weights
        current_output = self.model.forward(original_data)
        
        # Test with original weights
        self.model.set_parameters(self.original_params)
        original_output = self.model.forward(original_data)
        
        # Restore current weights
        current_params = self.model.get_parameters()
        self.model.set_parameters(current_params)
        
        # Calculate forgetting as difference in outputs
        forgetting_score = np.mean(np.abs(current_output - original_output))
        
        print_colored(f"Catastrophic forgetting score: {forgetting_score:.4f}", Colors.YELLOW)
        print_colored("(Lower scores indicate better retention of original knowledge)", Colors.BLUE)
        
        return forgetting_score
    
    def compare_with_baseline(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Compare fine-tuned model with baseline (original) model.
        
        Args:
            X_test: Test input features
            y_test: Test target labels
            
        Returns:
            Comparison metrics
        """
        print_subsection_header("Baseline Comparison")
        
        # Test fine-tuned model
        ft_predictions = self.model.forward(X_test)
        ft_accuracy = np.mean((ft_predictions > 0.5).astype(int) == y_test.astype(int))
        
        # Test original model
        current_params = self.model.get_parameters()
        self.model.set_parameters(self.original_params)
        orig_predictions = self.model.forward(X_test)
        orig_accuracy = np.mean((orig_predictions > 0.5).astype(int) == y_test.astype(int))
        
        # Restore fine-tuned weights
        self.model.set_parameters(current_params)
        
        improvement = ft_accuracy - orig_accuracy
        
        print_colored(f"Original model accuracy: {orig_accuracy:.4f}", Colors.BLUE)
        print_colored(f"Fine-tuned model accuracy: {ft_accuracy:.4f}", Colors.GREEN)
        print_colored(f"Improvement: {improvement:.4f}", Colors.YELLOW)
        
        return {
            'original_accuracy': orig_accuracy,
            'finetuned_accuracy': ft_accuracy,
            'improvement': improvement
        }
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training curves."""
        if not self.training_history['losses']:
            print_colored("No training history to plot", Colors.RED)
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        ax1.plot(self.training_history['epochs'], self.training_history['losses'], 'r-', label='Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.training_history['epochs'], self.training_history['accuracies'], 'g-', label='Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print_colored(f"Training plots saved to {save_path}", Colors.GREEN)
        
        plt.show()

def demonstrate_supervised_fine_tuning():
    """Main demonstration function for supervised fine-tuning."""
    
    print_section_header("Supervised Fine-tuning Demonstration")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Initialize fine-tuner
    model_config = {
        'input_size': 50,  # Smaller for demonstration
        'hidden_size': 25,
        'output_size': 1
    }
    
    fine_tuner = SupervisedFineTuner(model_config)
    
    # Prepare training data
    X_train, y_train, texts = fine_tuner.prepare_data("instruction")
    
    # Create test data (in practice, this would be held-out data)
    X_test = X_train[:3]  # Simple split for demonstration
    y_test = y_train[:3]
    
    # Perform fine-tuning
    with Timer("Fine-tuning"):
        results = fine_tuner.fine_tune(
            X_train, y_train,
            epochs=30,
            learning_rate=0.01,
            regularization_strength=0.001
        )
    
    # Evaluate catastrophic forgetting
    original_data = np.random.randn(5, model_config['input_size'])
    forgetting_score = fine_tuner.evaluate_catastrophic_forgetting(original_data)
    
    # Compare with baseline
    comparison = fine_tuner.compare_with_baseline(X_test, y_test)
    
    # Plot training progress
    fine_tuner.plot_training_progress()
    
    # Prepare final results
    final_results = {
        'method': 'Supervised Fine-tuning',
        'final_accuracy': results['final_accuracy'],
        'improvement_over_baseline': comparison['improvement'],
        'catastrophic_forgetting_score': forgetting_score,
        'training_epochs': len(results['training_history']['losses']),
        'metrics': {
            'accuracy': results['final_accuracy'],
            'exact_matches': int(results['final_accuracy'] * len(X_test)),
            'total_samples': len(X_test),
            'avg_token_overlap': 0.85  # Simulated
        }
    }
    
    print_subsection_header("Summary")
    print_colored("Key Benefits of Supervised Fine-tuning:", Colors.BLUE, bold=True)
    print("• Simple and effective for task-specific adaptation")
    print("• Well-understood optimization process")
    print("• Can leverage existing labeled datasets")
    print("• Relatively fast training compared to other methods")
    
    print_colored("\nKey Challenges:", Colors.YELLOW, bold=True)
    print("• Risk of catastrophic forgetting")
    print("• Requires high-quality labeled data")
    print("• May overfit to specific domains")
    print("• Limited ability to handle new task types")
    
    print_colored("\nUse Cases:", Colors.GREEN, bold=True)
    print("• Domain-specific applications (medical, legal, etc.)")
    print("• Task-specific optimization (classification, QA)")
    print("• Quick adaptation to new datasets")
    print("• When labeled data is readily available")
    
    return final_results

if __name__ == "__main__":
    # Run the demonstration
    results = demonstrate_supervised_fine_tuning()
    print_colored(f"\nDemonstration completed successfully!", Colors.GREEN, bold=True) 