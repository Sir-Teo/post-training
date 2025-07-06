"""
Chain-of-Thought Prompting Implementation

This module demonstrates Chain-of-Thought (CoT) prompting, a test-time scaling method
that improves reasoning by encouraging models to generate intermediate reasoning steps.
CoT prompting is particularly effective for complex reasoning tasks.

Key concepts:
- Step-by-step reasoning generation
- Few-shot prompting with reasoning examples
- Reasoning chain verification
- Multi-step problem decomposition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any
import random
from dataclasses import dataclass
import re
import time

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
class CoTExample:
    """Data structure for Chain-of-Thought examples"""
    question: str
    reasoning_steps: List[str]
    answer: str
    problem_type: str

@dataclass
class CoTResponse:
    """Data structure for CoT responses"""
    question: str
    reasoning_chain: List[str]
    final_answer: str
    confidence: float
    reasoning_quality: float
    generation_time: float

class CoTReasoningModel(nn.Module):
    """Model for Chain-of-Thought reasoning"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Reasoning encoder
        self.reasoning_encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=2)
        
        # Step generator
        self.step_generator = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=2)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, tokens: torch.Tensor, temperature: float = 1.0):
        """
        Forward pass for reasoning generation
        
        Args:
            tokens: Input tokens
            temperature: Sampling temperature
        """
        # Embed tokens
        embedded = self.embedding(tokens)
        embedded = self.dropout(embedded)
        
        # Encode reasoning context
        reasoning_context, _ = self.reasoning_encoder(embedded)
        
        # Generate next step
        step_output, _ = self.step_generator(embedded)
        
        # Get logits
        logits = self.output_layer(step_output)
        
        # Apply temperature
        logits = logits / temperature
        
        # Get confidence
        confidence = self.confidence_head(step_output[:, -1, :])
        
        return logits, confidence

class CoTPrompter:
    """Main class for Chain-of-Thought prompting"""
    
    def __init__(self, model: CoTReasoningModel, vocab_size: int, tokenizer_vocab_size: int = 1000):
        self.model = model
        self.vocab_size = vocab_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Simple tokenization for demonstration
        self.word_to_id = {}
        self.id_to_word = {}
        self._build_vocab(tokenizer_vocab_size)
        
        # CoT prompt templates
        self.templates = {
            'math': "Let's solve this step by step:",
            'reasoning': "Let me think through this carefully:",
            'general': "Let's break this down:"
        }
        
        # Few-shot examples
        self.few_shot_examples = self._create_few_shot_examples()
    
    def _build_vocab(self, vocab_size: int):
        """Build vocabulary for tokenization"""
        # Create a vocabulary with reasoning-related words
        words = ['<pad>', '<unk>', '<start>', '<end>', '<sep>', '<step>']
        reasoning_words = [
            'first', 'second', 'third', 'next', 'then', 'therefore', 'because',
            'since', 'so', 'thus', 'hence', 'step', 'let', 'we', 'need', 'to',
            'find', 'solve', 'calculate', 'determine', 'answer', 'result',
            'plus', 'minus', 'times', 'divided', 'equals', 'is', 'are',
            'what', 'how', 'why', 'when', 'where', 'which', 'who',
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then'
        ]
        
        words.extend(reasoning_words)
        words.extend([f'word_{i}' for i in range(vocab_size - len(words))])
        
        self.word_to_id = {word: i for i, word in enumerate(words)}
        self.id_to_word = {i: word for i, word in enumerate(words)}
    
    def _tokenize(self, text: str) -> List[int]:
        """Simple tokenization"""
        tokens = text.lower().split()
        return [self.word_to_id.get(token, 1) for token in tokens]
    
    def _detokenize(self, tokens: List[int]) -> str:
        """Convert tokens back to text"""
        words = [self.id_to_word.get(token, '<unk>') for token in tokens]
        return ' '.join(words).replace('<pad>', '').strip()
    
    def _create_few_shot_examples(self) -> Dict[str, List[CoTExample]]:
        """Create few-shot examples for different problem types"""
        examples = {
            'math': [
                CoTExample(
                    question="What is 25 + 17?",
                    reasoning_steps=[
                        "I need to add 25 and 17",
                        "First, I'll add the ones place: 5 + 7 = 12",
                        "That's 2 in the ones place and 1 to carry",
                        "Now the tens place: 2 + 1 + 1 = 4",
                        "So the answer is 42"
                    ],
                    answer="42",
                    problem_type="math"
                ),
                CoTExample(
                    question="If a rectangle has length 8 and width 5, what is its area?",
                    reasoning_steps=[
                        "To find the area of a rectangle, I use: Area = length × width",
                        "Given: length = 8, width = 5",
                        "Area = 8 × 5",
                        "Area = 40"
                    ],
                    answer="40",
                    problem_type="math"
                )
            ],
            'reasoning': [
                CoTExample(
                    question="If all birds can fly and a penguin is a bird, can a penguin fly?",
                    reasoning_steps=[
                        "Let me analyze this logical statement",
                        "Premise 1: All birds can fly",
                        "Premise 2: A penguin is a bird",
                        "Following the logic: If all birds can fly, and penguin is a bird, then penguin can fly",
                        "However, this conflicts with reality - penguins cannot fly",
                        "This means the first premise is false - not all birds can fly"
                    ],
                    answer="No, penguins cannot fly. The premise that 'all birds can fly' is incorrect.",
                    problem_type="reasoning"
                )
            ]
        }
        
        return examples
    
    def create_cot_prompt(self, question: str, problem_type: str = 'general') -> str:
        """Create a Chain-of-Thought prompt with few-shot examples"""
        # Start with few-shot examples
        prompt_parts = []
        
        # Add relevant few-shot examples
        if problem_type in self.few_shot_examples:
            for example in self.few_shot_examples[problem_type]:
                prompt_parts.append(f"Question: {example.question}")
                prompt_parts.append(f"Answer: {self.templates.get(problem_type, self.templates['general'])}")
                
                for i, step in enumerate(example.reasoning_steps, 1):
                    prompt_parts.append(f"Step {i}: {step}")
                
                prompt_parts.append(f"Final Answer: {example.answer}")
                prompt_parts.append("")  # Empty line
        
        # Add the current question
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append(f"Answer: {self.templates.get(problem_type, self.templates['general'])}")
        
        return "\n".join(prompt_parts)
    
    def generate_reasoning_step(self, prompt: str, temperature: float = 0.7) -> Tuple[str, float]:
        """Generate a single reasoning step"""
        # Tokenize prompt
        tokens = self._tokenize(prompt)
        
        # Generate step
        generated_tokens = []
        total_confidence = 0.0
        step_count = 0
        
        with torch.no_grad():
            current_tokens = tokens.copy()
            
            for _ in range(50):  # Max step length
                # Convert to tensor
                input_tensor = torch.tensor(current_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
                
                # Get logits and confidence
                logits, confidence = self.model(input_tensor, temperature=temperature)
                next_logits = logits[0, -1, :]
                
                # Sample next token
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Stop conditions
                if next_token == 0 or next_token == 3:  # <pad> or <end>
                    break
                
                # Add to sequence
                current_tokens.append(next_token)
                generated_tokens.append(next_token)
                
                total_confidence += confidence.item()
                step_count += 1
                
                # Break at natural stopping points
                if self.id_to_word.get(next_token, '') in ['.', '!', '?']:
                    break
        
        # Convert to text
        step_text = self._detokenize(generated_tokens)
        avg_confidence = total_confidence / max(step_count, 1)
        
        return step_text, avg_confidence
    
    def generate_cot_response(self, question: str, problem_type: str = 'general', 
                            max_steps: int = 6) -> CoTResponse:
        """Generate a complete Chain-of-Thought response"""
        start_time = time.time()
        
        # Create CoT prompt
        prompt = self.create_cot_prompt(question, problem_type)
        
        reasoning_chain = []
        confidences = []
        
        current_prompt = prompt
        
        # Generate reasoning steps
        for step_num in range(1, max_steps + 1):
            # Generate step
            step_text, confidence = self.generate_reasoning_step(current_prompt, temperature=0.7)
            
            if not step_text.strip():
                break
            
            # Clean up and format step
            step_text = step_text.strip()
            formatted_step = f"Step {step_num}: {step_text}"
            
            reasoning_chain.append(formatted_step)
            confidences.append(confidence)
            
            # Update prompt for next step
            current_prompt += f"\n{formatted_step}"
            
            # Check if we should stop (answer seems complete)
            if any(indicator in step_text.lower() for indicator in ['answer is', 'result is', 'solution is']):
                break
        
        # Generate final answer
        final_answer_prompt = current_prompt + "\nFinal Answer:"
        final_answer, final_confidence = self.generate_reasoning_step(final_answer_prompt, temperature=0.5)
        
        # Calculate overall metrics
        avg_confidence = np.mean(confidences + [final_confidence]) if confidences else final_confidence
        reasoning_quality = self._evaluate_reasoning_quality(reasoning_chain)
        generation_time = time.time() - start_time
        
        return CoTResponse(
            question=question,
            reasoning_chain=reasoning_chain,
            final_answer=final_answer.strip(),
            confidence=avg_confidence,
            reasoning_quality=reasoning_quality,
            generation_time=generation_time
        )
    
    def _evaluate_reasoning_quality(self, reasoning_chain: List[str]) -> float:
        """Evaluate the quality of reasoning chain"""
        if not reasoning_chain:
            return 0.0
        
        quality_score = 0.0
        
        # Check for logical connectors
        logical_connectors = ['because', 'since', 'therefore', 'thus', 'so', 'hence', 'if', 'then']
        connector_count = sum(1 for step in reasoning_chain 
                            for connector in logical_connectors 
                            if connector in step.lower())
        
        # Bonus for logical connectors
        quality_score += min(connector_count / len(reasoning_chain), 0.3)
        
        # Check for step progression
        step_words = ['first', 'second', 'third', 'next', 'then', 'finally']
        progression_count = sum(1 for step in reasoning_chain 
                              for word in step_words 
                              if word in step.lower())
        
        # Bonus for step progression
        quality_score += min(progression_count / len(reasoning_chain), 0.3)
        
        # Check for mathematical operations (for math problems)
        math_operations = ['+', '-', '*', '/', '=', 'plus', 'minus', 'times', 'divided', 'equals']
        math_count = sum(1 for step in reasoning_chain 
                        for op in math_operations 
                        if op in step.lower())
        
        # Bonus for mathematical reasoning
        quality_score += min(math_count / len(reasoning_chain), 0.2)
        
        # Base score for having multiple steps
        quality_score += min(len(reasoning_chain) / 5, 0.2)
        
        return min(quality_score, 1.0)
    
    def compare_with_direct_answer(self, question: str, problem_type: str = 'general') -> Dict[str, Any]:
        """Compare CoT approach with direct answer generation"""
        # Generate CoT response
        cot_response = self.generate_cot_response(question, problem_type)
        
        # Generate direct response
        direct_prompt = f"Question: {question}\nAnswer:"
        direct_answer, direct_confidence = self.generate_reasoning_step(direct_prompt, temperature=0.5)
        
        return {
            'question': question,
            'cot_response': cot_response,
            'direct_answer': direct_answer.strip(),
            'direct_confidence': direct_confidence,
            'cot_advantage': cot_response.confidence - direct_confidence,
            'reasoning_quality': cot_response.reasoning_quality
        }

def demonstrate_cot_prompting():
    """Demonstrate Chain-of-Thought prompting with examples"""
    print_colored("=== Chain-of-Thought Prompting Demonstration ===", "green")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Create model and prompter
    vocab_size = 1000
    model = CoTReasoningModel(vocab_size)
    prompter = CoTPrompter(model, vocab_size)
    
    # Test questions
    test_questions = [
        ("What is 15 + 27?", "math"),
        ("If a train travels 60 miles in 2 hours, what is its speed?", "math"),
        ("Why do leaves change color in fall?", "reasoning"),
        ("How does photosynthesis work?", "reasoning")
    ]
    
    print_colored("Testing Chain-of-Thought responses...", "cyan")
    
    results = []
    
    for question, problem_type in test_questions:
        print_colored(f"\nQuestion: {question}", "yellow")
        print_colored(f"Problem Type: {problem_type}", "magenta")
        
        # Generate CoT response
        cot_response = prompter.generate_cot_response(question, problem_type)
        
        # Display reasoning chain
        print_colored("Reasoning Chain:", "cyan")
        for step in cot_response.reasoning_chain:
            print(f"  {step}")
        
        print_colored(f"Final Answer: {cot_response.final_answer}", "green")
        print(f"Confidence: {cot_response.confidence:.3f}")
        print(f"Reasoning Quality: {cot_response.reasoning_quality:.3f}")
        print(f"Generation Time: {cot_response.generation_time:.3f}s")
        
        results.append(cot_response)
    
    # Compare with direct answers
    print_colored("\nComparing CoT vs Direct Answers...", "cyan")
    
    comparison_results = []
    
    for question, problem_type in test_questions[:2]:  # Limit for demo
        comparison = prompter.compare_with_direct_answer(question, problem_type)
        comparison_results.append(comparison)
        
        print_colored(f"\nQuestion: {question}", "yellow")
        print(f"CoT Answer: {comparison['cot_response'].final_answer}")
        print(f"Direct Answer: {comparison['direct_answer']}")
        print(f"CoT Confidence: {comparison['cot_response'].confidence:.3f}")
        print(f"Direct Confidence: {comparison['direct_confidence']:.3f}")
        print(f"CoT Advantage: {comparison['cot_advantage']:.3f}")
        print(f"Reasoning Quality: {comparison['reasoning_quality']:.3f}")
    
    # Analyze results
    print_colored("\nAnalyzing results...", "cyan")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Confidence comparison
    plt.subplot(2, 2, 1)
    cot_confidences = [r.confidence for r in results]
    direct_confidences = [c['direct_confidence'] for c in comparison_results]
    
    x = np.arange(len(comparison_results))
    width = 0.35
    
    plt.bar(x - width/2, [c['cot_response'].confidence for c in comparison_results], 
            width, label='CoT', alpha=0.7)
    plt.bar(x + width/2, direct_confidences, width, label='Direct', alpha=0.7)
    
    plt.title('Confidence Comparison')
    plt.xlabel('Question')
    plt.ylabel('Confidence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Reasoning quality
    plt.subplot(2, 2, 2)
    reasoning_qualities = [r.reasoning_quality for r in results]
    plt.bar(range(len(results)), reasoning_qualities, alpha=0.7)
    plt.title('Reasoning Quality Scores')
    plt.xlabel('Question')
    plt.ylabel('Quality Score')
    plt.grid(True, alpha=0.3)
    
    # Generation time
    plt.subplot(2, 2, 3)
    generation_times = [r.generation_time for r in results]
    plt.bar(range(len(results)), generation_times, alpha=0.7)
    plt.title('Generation Time')
    plt.xlabel('Question')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    
    # CoT advantage
    plt.subplot(2, 2, 4)
    cot_advantages = [c['cot_advantage'] for c in comparison_results]
    plt.bar(range(len(comparison_results)), cot_advantages, alpha=0.7)
    plt.title('CoT Confidence Advantage')
    plt.xlabel('Question')
    plt.ylabel('Confidence Difference')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cot_prompting_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print_colored("\nSummary Statistics:", "cyan")
    print(f"Average CoT Confidence: {np.mean(cot_confidences):.3f}")
    print(f"Average Direct Confidence: {np.mean(direct_confidences):.3f}")
    print(f"Average Reasoning Quality: {np.mean(reasoning_qualities):.3f}")
    print(f"Average Generation Time: {np.mean(generation_times):.3f}s")
    print(f"Average CoT Advantage: {np.mean(cot_advantages):.3f}")
    
    print_colored("\nChain-of-Thought prompting demonstration completed!", "green")
    print("Key insights:")
    print("1. CoT prompting breaks down complex problems into manageable steps")
    print("2. Step-by-step reasoning often leads to more reliable answers")
    print("3. CoT responses typically have higher confidence than direct answers")
    print("4. The reasoning quality can be evaluated using logical indicators")
    print("5. Few-shot examples help guide the reasoning process")
    print("6. CoT is particularly effective for mathematical and logical problems")

if __name__ == "__main__":
    demonstrate_cot_prompting() 