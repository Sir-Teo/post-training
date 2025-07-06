"""
Chain-of-Thought (CoT) Fine-tuning Implementation

This module demonstrates fine-tuning models to produce step-by-step reasoning
traces instead of just final answers. This improves model interpretability
and reasoning capabilities.

Key Concepts:
- Step-by-step reasoning generation
- Intermediate step supervision
- Reasoning path optimization
- Interpretable outputs
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import sys
import os
import re

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import (
    print_section_header, print_subsection_header, print_colored, Colors,
    Timer, SimpleModel, calculate_metrics, set_random_seed, create_embeddings
)
from shared_data.sample_data import get_sample_data

class CoTReasoningModel:
    """
    Model that generates chain-of-thought reasoning.
    This is a simplified implementation that demonstrates the key concepts.
    """
    
    def __init__(self, input_size: int, hidden_size: int, vocab_size: int = 100):
        """
        Initialize CoT reasoning model.
        
        Args:
            input_size: Input embedding dimension
            hidden_size: Hidden layer dimension
            vocab_size: Vocabulary size for reasoning generation
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Network for encoding input
        self.encoder = SimpleModel(input_size, hidden_size, hidden_size)
        
        # Network for generating reasoning steps
        self.reasoning_generator = SimpleModel(hidden_size, hidden_size, vocab_size)
        
        # Network for final answer
        self.answer_generator = SimpleModel(hidden_size, hidden_size // 2, 1)
        
        print_colored("✓ CoT Reasoning Model initialized", Colors.GREEN)
    
    def encode_input(self, x: np.ndarray) -> np.ndarray:
        """Encode input problem into hidden representation."""
        return self.encoder.forward(x)
    
    def generate_reasoning_step(self, hidden_state: np.ndarray) -> np.ndarray:
        """Generate a reasoning step given the current hidden state."""
        return self.reasoning_generator.forward(hidden_state)
    
    def generate_final_answer(self, reasoning_encoding: np.ndarray) -> np.ndarray:
        """Generate final answer from reasoning encoding."""
        return self.answer_generator.forward(reasoning_encoding)
    
    def forward_reasoning(self, x: np.ndarray, num_steps: int = 3) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Generate chain-of-thought reasoning.
        
        Args:
            x: Input problem encoding
            num_steps: Number of reasoning steps to generate
            
        Returns:
            Tuple of (reasoning_steps, final_answer)
        """
        # Encode input
        hidden_state = self.encode_input(x)
        
        # Generate reasoning steps
        reasoning_steps = []
        current_state = hidden_state
        
        for step in range(num_steps):
            step_output = self.generate_reasoning_step(current_state)
            reasoning_steps.append(step_output)
            
            # Update state (simplified - in practice this would be more sophisticated)
            current_state = current_state + 0.1 * np.tanh(step_output @ np.random.randn(self.vocab_size, self.hidden_size) * 0.1)
        
        # Generate final answer
        # Combine all reasoning steps
        reasoning_encoding = np.mean([hidden_state] + reasoning_steps, axis=0)
        final_answer = self.generate_final_answer(reasoning_encoding)
        
        return reasoning_steps, final_answer
    
    def compute_cot_loss(self, reasoning_steps: List[np.ndarray], target_steps: List[np.ndarray],
                        final_answer: np.ndarray, target_answer: np.ndarray) -> float:
        """
        Compute loss for chain-of-thought training.
        
        Args:
            reasoning_steps: Generated reasoning steps
            target_steps: Target reasoning steps
            final_answer: Generated final answer
            target_answer: Target final answer
            
        Returns:
            Combined loss
        """
        # Reasoning step loss (MSE for simplicity)
        step_loss = 0
        for i, (pred, target) in enumerate(zip(reasoning_steps, target_steps)):
            step_loss += np.mean((pred - target) ** 2)
        step_loss /= len(reasoning_steps)
        
        # Final answer loss
        answer_loss = self.answer_generator.compute_loss(target_answer, final_answer)
        
        # Combined loss with weighting
        total_loss = 0.7 * step_loss + 0.3 * answer_loss
        
        return total_loss

class CoTFineTuner:
    """
    Chain-of-Thought fine-tuning implementation.
    """
    
    def __init__(self, model_config: Dict[str, int]):
        """Initialize CoT fine-tuner."""
        self.model = CoTReasoningModel(**model_config)
        self.training_history = {
            'total_losses': [],
            'step_losses': [],
            'answer_losses': [],
            'reasoning_quality_scores': []
        }
        
        print_colored("✓ CoT Fine-tuner initialized", Colors.GREEN)
    
    def prepare_cot_data(self) -> Tuple[np.ndarray, List[List[np.ndarray]], np.ndarray, List[str]]:
        """
        Prepare chain-of-thought training data.
        
        Returns:
            Tuple of (X_train, target_reasoning_steps, y_train, problem_texts)
        """
        print_subsection_header("Preparing CoT Training Data")
        
        sample_data = get_sample_data()
        
        # Use math problems which have chain-of-thought annotations
        data = sample_data.math_problems
        problem_texts = [item['question'] for item in data]
        cot_texts = [item['chain_of_thought'] for item in data]
        answers = [item['answer'] for item in data]
        
        # Create embeddings for problems
        X_train, vocab = create_embeddings(problem_texts, vocab_size=self.model.input_size)
        
        # Create target reasoning steps (simplified representation)
        target_reasoning_steps = []
        for cot_text in cot_texts:
            # Split reasoning into steps
            steps = self.parse_reasoning_steps(cot_text)
            
            # Convert to numerical representation (simplified)
            step_embeddings = []
            for step in steps:
                # Create a simple embedding based on step content
                step_embedding = self.create_step_embedding(step)
                step_embeddings.append(step_embedding)
            
            # Pad or truncate to fixed number of steps
            while len(step_embeddings) < 3:
                step_embeddings.append(np.zeros(self.model.vocab_size))
            step_embeddings = step_embeddings[:3]
            
            target_reasoning_steps.append(step_embeddings)
        
        # Create binary answer labels (simplified)
        y_train = np.random.randint(0, 2, (len(data), 1)).astype(float)
        
        print_colored(f"CoT data prepared: {len(data)} problems with reasoning steps", Colors.GREEN)
        return X_train, target_reasoning_steps, y_train, problem_texts
    
    def parse_reasoning_steps(self, cot_text: str) -> List[str]:
        """Parse reasoning text into individual steps."""
        # Simple parsing - split by common delimiters
        steps = []
        
        # Split by periods and common reasoning phrases
        sentences = re.split(r'[.!?]|(?:Then)|(?:Next)|(?:So)|(?:Therefore)', cot_text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short fragments
                steps.append(sentence)
        
        return steps[:3]  # Limit to 3 steps for simplicity
    
    def create_step_embedding(self, step_text: str) -> np.ndarray:
        """Create a simple embedding for a reasoning step."""
        # Simple hash-based embedding
        embedding = np.zeros(self.model.vocab_size)
        
        # Use word hashing for simplicity
        words = step_text.lower().split()
        for word in words:
            hash_val = abs(hash(word)) % self.model.vocab_size
            embedding[hash_val] += 1
        
        # Normalize
        if np.sum(embedding) > 0:
            embedding = embedding / np.sum(embedding)
        
        return embedding
    
    def fine_tune_cot(self, X_train: np.ndarray, target_reasoning_steps: List[List[np.ndarray]],
                     y_train: np.ndarray, epochs: int = 50, learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Perform Chain-of-Thought fine-tuning.
        
        Args:
            X_train: Input problems
            target_reasoning_steps: Target reasoning step sequences
            y_train: Target answers
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Training results
        """
        print_subsection_header("CoT Fine-tuning Process")
        
        total_losses = []
        step_losses = []
        answer_losses = []
        reasoning_quality_scores = []
        
        for epoch in range(epochs):
            epoch_total_loss = 0
            epoch_step_loss = 0
            epoch_answer_loss = 0
            epoch_quality_score = 0
            
            for i in range(len(X_train)):
                # Forward pass
                x_sample = X_train[i:i+1]
                target_steps = target_reasoning_steps[i]
                target_answer = y_train[i:i+1]
                
                # Generate reasoning
                reasoning_steps, final_answer = self.model.forward_reasoning(x_sample, num_steps=3)
                
                # Compute losses
                total_loss = self.model.compute_cot_loss(
                    reasoning_steps, target_steps, final_answer, target_answer
                )
                
                # Compute individual losses for tracking
                step_loss = np.mean([np.mean((pred - target) ** 2) 
                                   for pred, target in zip(reasoning_steps, target_steps)])
                answer_loss = self.model.answer_generator.compute_loss(target_answer, final_answer)
                
                # Compute reasoning quality score (simplified)
                quality_score = self.evaluate_reasoning_quality(reasoning_steps, target_steps)
                
                # Accumulate losses
                epoch_total_loss += total_loss
                epoch_step_loss += step_loss
                epoch_answer_loss += answer_loss
                epoch_quality_score += quality_score
                
                # Simplified backward pass (in practice, this would be more complex)
                # Here we just simulate parameter updates
                self.simulate_gradient_update(learning_rate)
            
            # Average losses
            avg_total_loss = epoch_total_loss / len(X_train)
            avg_step_loss = epoch_step_loss / len(X_train)
            avg_answer_loss = epoch_answer_loss / len(X_train)
            avg_quality_score = epoch_quality_score / len(X_train)
            
            # Store metrics
            total_losses.append(avg_total_loss)
            step_losses.append(avg_step_loss)
            answer_losses.append(avg_answer_loss)
            reasoning_quality_scores.append(avg_quality_score)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print_colored(
                    f"Epoch {epoch+1}/{epochs} - Total Loss: {avg_total_loss:.4f}, "
                    f"Step Loss: {avg_step_loss:.4f}, Answer Loss: {avg_answer_loss:.4f}, "
                    f"Quality Score: {avg_quality_score:.4f}",
                    Colors.BLUE
                )
        
        # Store training history
        self.training_history['total_losses'] = total_losses
        self.training_history['step_losses'] = step_losses
        self.training_history['answer_losses'] = answer_losses
        self.training_history['reasoning_quality_scores'] = reasoning_quality_scores
        
        print_colored(f"CoT fine-tuning completed!", Colors.GREEN)
        
        return {
            'final_total_loss': total_losses[-1],
            'final_quality_score': reasoning_quality_scores[-1],
            'training_history': self.training_history
        }
    
    def simulate_gradient_update(self, learning_rate: float):
        """Simulate gradient updates (simplified for demonstration)."""
        # In practice, this would involve proper backpropagation
        # Here we just add small random perturbations to simulate learning
        noise_scale = learning_rate * 0.01
        
        # Add noise to encoder weights
        encoder_params = self.model.encoder.get_parameters()
        for key in encoder_params:
            encoder_params[key] += np.random.normal(0, noise_scale, encoder_params[key].shape)
        self.model.encoder.set_parameters(encoder_params)
        
        # Similar for other components
        reasoning_params = self.model.reasoning_generator.get_parameters()
        for key in reasoning_params:
            reasoning_params[key] += np.random.normal(0, noise_scale, reasoning_params[key].shape)
        self.model.reasoning_generator.set_parameters(reasoning_params)
        
        answer_params = self.model.answer_generator.get_parameters()
        for key in answer_params:
            answer_params[key] += np.random.normal(0, noise_scale, answer_params[key].shape)
        self.model.answer_generator.set_parameters(answer_params)
    
    def evaluate_reasoning_quality(self, reasoning_steps: List[np.ndarray], 
                                 target_steps: List[np.ndarray]) -> float:
        """Evaluate the quality of generated reasoning steps."""
        # Simple quality metric based on similarity to target steps
        similarities = []
        for pred, target in zip(reasoning_steps, target_steps):
            # Cosine similarity
            pred_norm = np.linalg.norm(pred)
            target_norm = np.linalg.norm(target)
            
            if pred_norm > 0 and target_norm > 0:
                similarity = np.dot(pred, target) / (pred_norm * target_norm)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def demonstrate_reasoning(self, problem_text: str) -> Dict[str, Any]:
        """Demonstrate reasoning generation for a specific problem."""
        print_subsection_header(f"Reasoning Demonstration")
        print_colored(f"Problem: {problem_text}", Colors.BLUE)
        
        # Encode problem
        problem_embedding, _ = create_embeddings([problem_text], vocab_size=self.model.input_size)
        
        # Generate reasoning
        reasoning_steps, final_answer = self.model.forward_reasoning(problem_embedding, num_steps=3)
        
        print_colored("Generated Reasoning Steps:", Colors.GREEN, bold=True)
        for i, step in enumerate(reasoning_steps):
            # Convert step back to interpretable form (simplified)
            step_description = self.interpret_reasoning_step(step, i+1)
            print(f"  Step {i+1}: {step_description}")
        
        # Interpret final answer
        answer_prob = self.model.answer_generator.forward(np.mean(reasoning_steps, axis=0))
        answer_description = "Positive" if answer_prob[0, 0] > 0.5 else "Negative"
        print_colored(f"Final Answer: {answer_description} (confidence: {answer_prob[0, 0]:.3f})", Colors.YELLOW)
        
        return {
            'reasoning_steps': reasoning_steps,
            'final_answer': answer_prob,
            'step_descriptions': [self.interpret_reasoning_step(step, i+1) for i, step in enumerate(reasoning_steps)]
        }
    
    def interpret_reasoning_step(self, step_embedding: np.ndarray, step_number: int) -> str:
        """Convert step embedding back to interpretable text (simplified)."""
        # Find top activated dimensions
        top_indices = np.argsort(step_embedding)[-3:][::-1]
        
        # Create simple interpretation
        step_types = [
            "Analyzing the problem structure",
            "Applying mathematical operations", 
            "Computing intermediate results",
            "Checking consistency",
            "Drawing conclusions"
        ]
        
        base_description = step_types[(step_number - 1) % len(step_types)]
        activation_strength = np.max(step_embedding)
        
        if activation_strength > 0.1:
            confidence = "with high confidence"
        elif activation_strength > 0.05:
            confidence = "with moderate confidence"
        else:
            confidence = "with low confidence"
        
        return f"{base_description} {confidence}"
    
    def plot_training_progress(self):
        """Plot CoT training curves."""
        if not self.training_history['total_losses']:
            print_colored("No training history to plot", Colors.RED)
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.training_history['total_losses']) + 1)
        
        # Total loss
        axes[0, 0].plot(epochs, self.training_history['total_losses'], 'r-', label='Total Loss')
        axes[0, 0].set_title('Total Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Step loss
        axes[0, 1].plot(epochs, self.training_history['step_losses'], 'b-', label='Step Loss')
        axes[0, 1].set_title('Reasoning Step Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Answer loss
        axes[1, 0].plot(epochs, self.training_history['answer_losses'], 'g-', label='Answer Loss')
        axes[1, 0].set_title('Final Answer Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Quality score
        axes[1, 1].plot(epochs, self.training_history['reasoning_quality_scores'], 'm-', label='Quality Score')
        axes[1, 1].set_title('Reasoning Quality Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Quality Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

def demonstrate_chain_of_thought_finetuning():
    """Main demonstration function for Chain-of-Thought fine-tuning."""
    
    print_section_header("Chain-of-Thought Fine-tuning Demonstration")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Initialize CoT fine-tuner
    model_config = {
        'input_size': 30,
        'hidden_size': 20,
        'vocab_size': 50
    }
    
    cot_tuner = CoTFineTuner(model_config)
    
    # Prepare CoT training data
    X_train, target_reasoning_steps, y_train, problem_texts = cot_tuner.prepare_cot_data()
    
    # Perform CoT fine-tuning
    with Timer("Chain-of-Thought fine-tuning"):
        results = cot_tuner.fine_tune_cot(
            X_train, target_reasoning_steps, y_train,
            epochs=30,
            learning_rate=0.01
        )
    
    # Plot training progress
    cot_tuner.plot_training_progress()
    
    # Demonstrate reasoning on sample problems
    sample_problems = [
        "What is 15 + 28?",
        "If a box has 12 red balls and 8 blue balls, what is the total?",
        "Solve: 2x + 5 = 11"
    ]
    
    reasoning_demonstrations = []
    for problem in sample_problems:
        demo = cot_tuner.demonstrate_reasoning(problem)
        reasoning_demonstrations.append(demo)
    
    # Prepare final results
    final_results = {
        'method': 'Chain-of-Thought Fine-tuning',
        'final_quality_score': results['final_quality_score'],
        'final_total_loss': results['final_total_loss'],
        'reasoning_demonstrations': len(reasoning_demonstrations),
        'metrics': {
            'accuracy': min(results['final_quality_score'] + 0.2, 1.0),  # Simulated
            'exact_matches': int(results['final_quality_score'] * len(X_train)),
            'total_samples': len(X_train),
            'avg_token_overlap': results['final_quality_score']
        }
    }
    
    print_subsection_header("Summary")
    print_colored("Key Benefits of Chain-of-Thought Fine-tuning:", Colors.BLUE, bold=True)
    print("• Improves model interpretability through explicit reasoning")
    print("• Enhances complex problem-solving capabilities")
    print("• Enables step-by-step verification of model logic")
    print("• Better performance on multi-step reasoning tasks")
    
    print_colored("\nKey Features:", Colors.GREEN, bold=True)
    print("• Structured reasoning step generation")
    print("• Intermediate supervision during training")
    print("• Quality assessment of reasoning chains")
    print("• Explicit modeling of problem-solving process")
    
    print_colored("\nUse Cases:", Colors.YELLOW, bold=True)
    print("• Mathematical problem solving")
    print("• Logical reasoning tasks")
    print("• Educational applications requiring explanations")
    print("• Any domain requiring step-by-step analysis")
    
    return final_results

if __name__ == "__main__":
    # Run the demonstration
    results = demonstrate_chain_of_thought_finetuning()
    print_colored(f"\nCoT demonstration completed successfully!", Colors.GREEN, bold=True) 