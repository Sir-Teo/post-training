"""
Shared sample data for post-training method demonstrations.
This module provides common datasets used across all implementations.
"""

import json
import random
from typing import List, Dict, Any, Tuple
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class SampleData:
    """Provides sample datasets for post-training demonstrations."""
    
    def __init__(self):
        self.math_problems = self._create_math_problems()
        self.reasoning_tasks = self._create_reasoning_tasks()
        self.instruction_data = self._create_instruction_data()
        self.preference_data = self._create_preference_data()
        self.dialogue_data = self._create_dialogue_data()
        self.code_problems = self._create_code_problems()
        
    def _create_math_problems(self) -> List[Dict[str, Any]]:
        """Create sample math problems for CoT and reasoning demonstrations."""
        problems = [
            {
                "question": "What is 23 + 47?",
                "answer": "70",
                "chain_of_thought": "Let me add these step by step: 23 + 47 = 20 + 40 + 3 + 7 = 60 + 10 = 70",
                "difficulty": "easy"
            },
            {
                "question": "A box contains 12 red balls and 8 blue balls. If I randomly pick 3 balls, what's the probability they're all red?",
                "answer": "11/57",
                "chain_of_thought": "Total balls = 12 + 8 = 20. Probability of first red = 12/20. Second red = 11/19. Third red = 10/18. Total probability = (12/20) × (11/19) × (10/18) = 1320/6840 = 11/57",
                "difficulty": "medium"
            },
            {
                "question": "If x + 2y = 10 and 2x - y = 5, what is x?",
                "answer": "4",
                "chain_of_thought": "From equation 1: x = 10 - 2y. Substituting into equation 2: 2(10 - 2y) - y = 5. This gives: 20 - 4y - y = 5, so 20 - 5y = 5, thus 5y = 15, so y = 3. Therefore x = 10 - 2(3) = 4",
                "difficulty": "medium"
            },
            {
                "question": "What is the sum of the first 10 prime numbers?",
                "answer": "129",
                "chain_of_thought": "The first 10 prime numbers are: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29. Sum = 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29 = 129",
                "difficulty": "medium"
            },
            {
                "question": "A train travels 300 miles in 4 hours. How long will it take to travel 750 miles at the same speed?",
                "answer": "10 hours",
                "chain_of_thought": "First, find the speed: 300 miles ÷ 4 hours = 75 mph. For 750 miles: 750 ÷ 75 = 10 hours",
                "difficulty": "easy"
            }
        ]
        return problems
    
    def _create_reasoning_tasks(self) -> List[Dict[str, Any]]:
        """Create sample reasoning tasks for various reasoning methods."""
        tasks = [
            {
                "prompt": "All birds can fly. Penguins are birds. Can penguins fly?",
                "correct_answer": "No, this is a logical fallacy. The premise 'All birds can fly' is incorrect.",
                "reasoning_type": "logical",
                "steps": [
                    "Analyze the premise: 'All birds can fly'",
                    "Identify the specific case: 'Penguins are birds'",
                    "Apply knowledge: Penguins cannot fly",
                    "Conclude: The initial premise is false"
                ]
            },
            {
                "prompt": "If it's raining, then the ground is wet. The ground is wet. Is it raining?",
                "correct_answer": "Cannot be determined. The ground being wet doesn't necessarily mean it's raining.",
                "reasoning_type": "logical",
                "steps": [
                    "Identify the conditional: Rain → Wet ground",
                    "Note what we know: Ground is wet",
                    "Check if this is affirming the consequent fallacy",
                    "Conclude: Other factors could make ground wet"
                ]
            },
            {
                "prompt": "A farmer has chickens and cows. There are 20 heads and 54 legs total. How many chickens and cows?",
                "correct_answer": "13 chickens and 7 cows",
                "reasoning_type": "mathematical",
                "steps": [
                    "Let c = chickens, w = cows",
                    "Heads equation: c + w = 20",
                    "Legs equation: 2c + 4w = 54",
                    "Solve: c = 13, w = 7"
                ]
            },
            {
                "prompt": "What comes next in the sequence: 2, 6, 12, 20, 30, ?",
                "correct_answer": "42",
                "reasoning_type": "pattern",
                "steps": [
                    "Find differences: 6-2=4, 12-6=6, 20-12=8, 30-20=10",
                    "Pattern in differences: 4, 6, 8, 10 (increasing by 2)",
                    "Next difference should be 12",
                    "Therefore: 30 + 12 = 42"
                ]
            }
        ]
        return tasks
    
    def _create_instruction_data(self) -> List[Dict[str, Any]]:
        """Create sample instruction-following data."""
        instructions = [
            {
                "instruction": "Write a short poem about artificial intelligence.",
                "input": "",
                "output": "Silicon dreams and digital minds,\nLearning patterns, data finds,\nNeural networks, layers deep,\nAI's promise we shall keep.",
                "category": "creative"
            },
            {
                "instruction": "Explain the concept of photosynthesis in simple terms.",
                "input": "",
                "output": "Photosynthesis is how plants make their own food. They use sunlight, water, and carbon dioxide from the air to create sugar and oxygen. It's like cooking with sunlight as the energy source!",
                "category": "educational"
            },
            {
                "instruction": "Translate the following English text to French: 'Hello, how are you?'",
                "input": "Hello, how are you?",
                "output": "Bonjour, comment allez-vous?",
                "category": "translation"
            },
            {
                "instruction": "Write a Python function to calculate the factorial of a number.",
                "input": "",
                "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)",
                "category": "coding"
            },
            {
                "instruction": "Summarize the key points of this text.",
                "input": "Climate change is one of the most pressing issues of our time. It refers to long-term shifts in global temperatures and weather patterns. While climate change is natural, human activities have accelerated these changes since the 1800s. The main cause is the burning of fossil fuels, which releases greenhouse gases.",
                "output": "Key points: 1) Climate change involves long-term temperature and weather shifts. 2) While natural, human activities have accelerated it since the 1800s. 3) Main cause is burning fossil fuels that release greenhouse gases.",
                "category": "summarization"
            }
        ]
        return instructions
    
    def _create_preference_data(self) -> List[Dict[str, Any]]:
        """Create sample preference data for RLHF and DPO."""
        preferences = [
            {
                "prompt": "What is the capital of France?",
                "chosen": "The capital of France is Paris. Paris is located in north-central France and is the country's largest city and economic center.",
                "rejected": "Paris is the capital of France.",
                "reason": "More informative and detailed response"
            },
            {
                "prompt": "How do you make a sandwich?",
                "chosen": "To make a basic sandwich: 1) Choose your bread, 2) Add your preferred protein (meat, cheese, etc.), 3) Include vegetables like lettuce or tomatoes, 4) Add condiments if desired, 5) Close with the second piece of bread. Enjoy!",
                "rejected": "Put stuff between bread.",
                "reason": "Provides clear steps and helpful details"
            },
            {
                "prompt": "What should I do if I'm feeling sad?",
                "chosen": "If you're feeling sad, try these gentle approaches: talk to someone you trust, engage in activities you enjoy, get some fresh air or exercise, practice self-care, or consider professional help if sadness persists. Remember, it's okay to feel sad sometimes.",
                "rejected": "Just get over it and think positive thoughts.",
                "reason": "More empathetic and provides actionable, helpful advice"
            },
            {
                "prompt": "Explain quantum computing.",
                "chosen": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in multiple states simultaneously, potentially solving certain problems exponentially faster than classical computers.",
                "rejected": "Quantum computers are really fast computers that use quantum stuff to compute things better.",
                "reason": "Accurate technical explanation with proper terminology"
            }
        ]
        return preferences
    
    def _create_dialogue_data(self) -> List[Dict[str, Any]]:
        """Create sample dialogue data for multi-turn conversations."""
        dialogues = [
            {
                "conversation": [
                    {"role": "user", "content": "I'm planning a trip to Japan. What should I know?"},
                    {"role": "assistant", "content": "Japan is a wonderful destination! What time of year are you planning to visit? This will help me give you more specific advice about weather, activities, and what to pack."},
                    {"role": "user", "content": "I'm thinking of going in spring, around March or April."},
                    {"role": "assistant", "content": "Perfect timing! March-April is cherry blossom season (sakura). Here are key things to know: 1) Book accommodations early as it's peak season, 2) Pack layers as weather can be unpredictable, 3) Consider getting a JR Pass for train travel, 4) Learn basic Japanese phrases - locals appreciate the effort, 5) Don't forget to try authentic sushi and ramen!"}
                ],
                "topic": "travel"
            },
            {
                "conversation": [
                    {"role": "user", "content": "I'm having trouble with my Python code. It keeps giving me an error."},
                    {"role": "assistant", "content": "I'd be happy to help you debug your Python code! Could you share the error message and the relevant code snippet? This will help me understand what's going wrong."},
                    {"role": "user", "content": "The error is 'list index out of range' and it happens when I try to access my_list[5] in a list with only 3 elements."},
                    {"role": "assistant", "content": "Ah, I see the issue! You're trying to access index 5 in a list that only has 3 elements (indices 0, 1, 2). To fix this, you can: 1) Check the list length first with len(my_list), 2) Use try/except to handle the error gracefully, or 3) Make sure your list has enough elements before accessing that index. Which approach would work best for your use case?"}
                ],
                "topic": "programming"
            }
        ]
        return dialogues
    
    def _create_code_problems(self) -> List[Dict[str, Any]]:
        """Create sample code problems for code generation tasks."""
        problems = [
            {
                "problem": "Write a function to check if a number is prime.",
                "solution": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
                "test_cases": [
                    {"input": 2, "expected": True},
                    {"input": 4, "expected": False},
                    {"input": 17, "expected": True},
                    {"input": 1, "expected": False}
                ]
            },
            {
                "problem": "Write a function to find the maximum element in a list.",
                "solution": "def find_max(lst):\n    if not lst:\n        return None\n    max_val = lst[0]\n    for item in lst:\n        if item > max_val:\n            max_val = item\n    return max_val",
                "test_cases": [
                    {"input": [1, 3, 2, 5, 4], "expected": 5},
                    {"input": [-1, -3, -2], "expected": -1},
                    {"input": [42], "expected": 42}
                ]
            }
        ]
        return problems
    
    def get_random_sample(self, data_type: str, n: int = 1) -> List[Dict[str, Any]]:
        """Get random samples from specified data type."""
        data_map = {
            "math": self.math_problems,
            "reasoning": self.reasoning_tasks,
            "instruction": self.instruction_data,
            "preference": self.preference_data,
            "dialogue": self.dialogue_data,
            "code": self.code_problems
        }
        
        if data_type not in data_map:
            raise ValueError(f"Unknown data type: {data_type}")
            
        return random.sample(data_map[data_type], min(n, len(data_map[data_type])))
    
    def get_all_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all sample data."""
        return {
            "math": self.math_problems,
            "reasoning": self.reasoning_tasks,
            "instruction": self.instruction_data,
            "preference": self.preference_data,
            "dialogue": self.dialogue_data,
            "code": self.code_problems
        }

# Create global instance
sample_data = SampleData()

def get_sample_data():
    """Get the global sample data instance."""
    return sample_data 