#!/usr/bin/env python3
"""
Comprehensive Post-Training Methods Demonstration

This script demonstrates all implemented post-training methods from the research paper:
"LLM Post-Training: A Deep Dive into Reasoning Large Language Models"

The script runs demonstrations for:
1. Fine-tuning Methods (4 methods)
2. Reinforcement Learning Methods (3 methods)
3. Test-time Scaling Methods (2 methods)

Usage: python run_all_demonstrations.py [--method METHOD] [--quick]
"""

import argparse
import time
import sys
import os
from typing import Dict, List, Any
import traceback

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.common import print_colored, set_random_seed

def run_fine_tuning_demos(quick_mode: bool = False):
    """Run all fine-tuning method demonstrations"""
    print_colored("=" * 80, "green")
    print_colored("FINE-TUNING METHODS DEMONSTRATION", "green")
    print_colored("=" * 80, "green")
    
    methods = [
        ("Supervised Fine-tuning", "fine_tuning.supervised_fine_tuning"),
        ("LoRA Adaptation", "fine_tuning.lora_adaptation"),
        ("Chain-of-Thought Fine-tuning", "fine_tuning.chain_of_thought_finetuning"),
        ("Instruction Tuning", "fine_tuning.instruction_tuning")
    ]
    
    results = {}
    
    for method_name, module_name in methods:
        print_colored(f"\n{'='*60}", "blue")
        print_colored(f"Running {method_name}", "blue")
        print_colored(f"{'='*60}", "blue")
        
        try:
            # Import and run the demonstration
            module = __import__(module_name, fromlist=[''])
            
            # Find the demonstration function
            demo_func = None
            for attr_name in dir(module):
                if attr_name.startswith('demonstrate_'):
                    demo_func = getattr(module, attr_name)
                    break
            
            if demo_func:
                start_time = time.time()
                demo_func()
                end_time = time.time()
                
                results[method_name] = {
                    'status': 'success',
                    'duration': end_time - start_time,
                    'module': module_name
                }
                
                print_colored(f"✓ {method_name} completed in {end_time - start_time:.2f}s", "green")
            else:
                print_colored(f"✗ No demonstration function found for {method_name}", "red")
                results[method_name] = {'status': 'error', 'error': 'No demo function'}
                
        except Exception as e:
            print_colored(f"✗ Error running {method_name}: {str(e)}", "red")
            if not quick_mode:
                traceback.print_exc()
            results[method_name] = {'status': 'error', 'error': str(e)}
    
    return results

def run_reinforcement_learning_demos(quick_mode: bool = False):
    """Run all reinforcement learning method demonstrations"""
    print_colored("=" * 80, "green")
    print_colored("REINFORCEMENT LEARNING METHODS DEMONSTRATION", "green")
    print_colored("=" * 80, "green")
    
    methods = [
        ("Reward Modeling", "reinforcement_learning.reward_modeling"),
        ("Direct Preference Optimization (DPO)", "reinforcement_learning.dpo"),
        ("Proximal Policy Optimization (PPO)", "reinforcement_learning.ppo")
    ]
    
    results = {}
    
    for method_name, module_name in methods:
        print_colored(f"\n{'='*60}", "blue")
        print_colored(f"Running {method_name}", "blue")
        print_colored(f"{'='*60}", "blue")
        
        try:
            # Import and run the demonstration
            module = __import__(module_name, fromlist=[''])
            
            # Find the demonstration function
            demo_func = None
            for attr_name in dir(module):
                if attr_name.startswith('demonstrate_'):
                    demo_func = getattr(module, attr_name)
                    break
            
            if demo_func:
                start_time = time.time()
                demo_func()
                end_time = time.time()
                
                results[method_name] = {
                    'status': 'success',
                    'duration': end_time - start_time,
                    'module': module_name
                }
                
                print_colored(f"✓ {method_name} completed in {end_time - start_time:.2f}s", "green")
            else:
                print_colored(f"✗ No demonstration function found for {method_name}", "red")
                results[method_name] = {'status': 'error', 'error': 'No demo function'}
                
        except Exception as e:
            print_colored(f"✗ Error running {method_name}: {str(e)}", "red")
            if not quick_mode:
                traceback.print_exc()
            results[method_name] = {'status': 'error', 'error': str(e)}
    
    return results

def run_test_time_scaling_demos(quick_mode: bool = False):
    """Run all test-time scaling method demonstrations"""
    print_colored("=" * 80, "green")
    print_colored("TEST-TIME SCALING METHODS DEMONSTRATION", "green")
    print_colored("=" * 80, "green")
    
    methods = [
        ("Best-of-N Search", "test_time_scaling.best_of_n_search"),
        ("Chain-of-Thought Prompting", "test_time_scaling.chain_of_thought_prompting")
    ]
    
    results = {}
    
    for method_name, module_name in methods:
        print_colored(f"\n{'='*60}", "blue")
        print_colored(f"Running {method_name}", "blue")
        print_colored(f"{'='*60}", "blue")
        
        try:
            # Import and run the demonstration
            module = __import__(module_name, fromlist=[''])
            
            # Find the demonstration function
            demo_func = None
            for attr_name in dir(module):
                if attr_name.startswith('demonstrate_'):
                    demo_func = getattr(module, attr_name)
                    break
            
            if demo_func:
                start_time = time.time()
                demo_func()
                end_time = time.time()
                
                results[method_name] = {
                    'status': 'success',
                    'duration': end_time - start_time,
                    'module': module_name
                }
                
                print_colored(f"✓ {method_name} completed in {end_time - start_time:.2f}s", "green")
            else:
                print_colored(f"✗ No demonstration function found for {method_name}", "red")
                results[method_name] = {'status': 'error', 'error': 'No demo function'}
                
        except Exception as e:
            print_colored(f"✗ Error running {method_name}: {str(e)}", "red")
            if not quick_mode:
                traceback.print_exc()
            results[method_name] = {'status': 'error', 'error': str(e)}
    
    return results

def print_summary_report(all_results: Dict[str, Dict[str, Any]]):
    """Print a comprehensive summary report"""
    print_colored("\n" + "=" * 80, "green")
    print_colored("COMPREHENSIVE SUMMARY REPORT", "green")
    print_colored("=" * 80, "green")
    
    # Overall statistics
    total_methods = 0
    successful_methods = 0
    failed_methods = 0
    total_duration = 0
    
    for category, results in all_results.items():
        print_colored(f"\n{category}:", "cyan")
        print_colored("-" * len(category), "cyan")
        
        for method_name, result in results.items():
            total_methods += 1
            
            if result['status'] == 'success':
                successful_methods += 1
                duration = result['duration']
                total_duration += duration
                print_colored(f"  ✓ {method_name:<40} {duration:>8.2f}s", "green")
            else:
                failed_methods += 1
                error = result.get('error', 'Unknown error')
                print_colored(f"  ✗ {method_name:<40} {error}", "red")
    
    # Summary statistics
    print_colored(f"\nOVERALL STATISTICS:", "yellow")
    print_colored(f"  Total Methods: {total_methods}", "white")
    print_colored(f"  Successful: {successful_methods}", "green")
    print_colored(f"  Failed: {failed_methods}", "red")
    print_colored(f"  Success Rate: {successful_methods/total_methods*100:.1f}%", "cyan")
    print_colored(f"  Total Duration: {total_duration:.2f}s", "white")
    
    if successful_methods > 0:
        print_colored(f"  Average Duration: {total_duration/successful_methods:.2f}s", "white")
    
    # Method category breakdown
    print_colored(f"\nMETHOD CATEGORY BREAKDOWN:", "yellow")
    for category, results in all_results.items():
        category_success = sum(1 for r in results.values() if r['status'] == 'success')
        category_total = len(results)
        category_duration = sum(r.get('duration', 0) for r in results.values() if r['status'] == 'success')
        
        print_colored(f"  {category}:", "cyan")
        print_colored(f"    Success: {category_success}/{category_total} ({category_success/category_total*100:.1f}%)", "white")
        print_colored(f"    Duration: {category_duration:.2f}s", "white")
    
    # Recommendations
    print_colored(f"\nRECOMMENDATIONS:", "yellow")
    
    if failed_methods > 0:
        print_colored("  • Some methods failed - check error messages above", "red")
        print_colored("  • Consider running individual methods for debugging", "white")
    
    if successful_methods > 0:
        print_colored("  • Check generated plots and outputs for each method", "white")
        print_colored("  • Compare performance across different approaches", "white")
        print_colored("  • Consider experimenting with different parameters", "white")
    
    print_colored("\n" + "=" * 80, "green")

def run_individual_method(method_name: str, quick_mode: bool = False):
    """Run a specific method demonstration"""
    method_mapping = {
        'sft': 'fine_tuning.supervised_fine_tuning',
        'lora': 'fine_tuning.lora_adaptation',
        'cot-ft': 'fine_tuning.chain_of_thought_finetuning',
        'instruction': 'fine_tuning.instruction_tuning',
        'reward': 'reinforcement_learning.reward_modeling',
        'dpo': 'reinforcement_learning.dpo',
        'ppo': 'reinforcement_learning.ppo',
        'best-of-n': 'test_time_scaling.best_of_n_search',
        'cot-prompt': 'test_time_scaling.chain_of_thought_prompting'
    }
    
    if method_name not in method_mapping:
        print_colored(f"Unknown method: {method_name}", "red")
        print_colored("Available methods:", "yellow")
        for key, value in method_mapping.items():
            print_colored(f"  {key}: {value}", "white")
        return
    
    module_name = method_mapping[method_name]
    
    print_colored(f"Running {method_name} ({module_name})", "blue")
    
    try:
        module = __import__(module_name, fromlist=[''])
        
        # Find the demonstration function
        demo_func = None
        for attr_name in dir(module):
            if attr_name.startswith('demonstrate_'):
                demo_func = getattr(module, attr_name)
                break
        
        if demo_func:
            start_time = time.time()
            demo_func()
            end_time = time.time()
            
            print_colored(f"✓ {method_name} completed in {end_time - start_time:.2f}s", "green")
        else:
            print_colored(f"✗ No demonstration function found for {method_name}", "red")
            
    except Exception as e:
        print_colored(f"✗ Error running {method_name}: {str(e)}", "red")
        if not quick_mode:
            traceback.print_exc()

def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description='Post-Training Methods Demonstration')
    parser.add_argument('--method', type=str, help='Run specific method only')
    parser.add_argument('--quick', action='store_true', help='Quick mode (less detailed error reporting)')
    parser.add_argument('--category', type=str, choices=['fine-tuning', 'rl', 'test-time'], 
                       help='Run specific category only')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    print_colored("🚀 LLM Post-Training Methods Demonstration", "green")
    print_colored("=" * 80, "green")
    print_colored("Educational implementations of post-training methods", "white")
    print_colored("Based on: 'LLM Post-Training: A Deep Dive into Reasoning Large Language Models'", "white")
    print_colored("=" * 80, "green")
    
    start_time = time.time()
    
    # Run individual method if specified
    if args.method:
        run_individual_method(args.method, args.quick)
        return
    
    # Run specific category if specified
    if args.category:
        if args.category == 'fine-tuning':
            results = run_fine_tuning_demos(args.quick)
            all_results = {'Fine-tuning Methods': results}
        elif args.category == 'rl':
            results = run_reinforcement_learning_demos(args.quick)
            all_results = {'Reinforcement Learning Methods': results}
        elif args.category == 'test-time':
            results = run_test_time_scaling_demos(args.quick)
            all_results = {'Test-time Scaling Methods': results}
    else:
        # Run all demonstrations
        all_results = {}
        
        # Fine-tuning methods
        print_colored("\n🔧 Starting Fine-tuning Methods...", "cyan")
        all_results['Fine-tuning Methods'] = run_fine_tuning_demos(args.quick)
        
        # Reinforcement learning methods
        print_colored("\n🎯 Starting Reinforcement Learning Methods...", "cyan")
        all_results['Reinforcement Learning Methods'] = run_reinforcement_learning_demos(args.quick)
        
        # Test-time scaling methods
        print_colored("\n⚡ Starting Test-time Scaling Methods...", "cyan")
        all_results['Test-time Scaling Methods'] = run_test_time_scaling_demos(args.quick)
    
    # Print summary report
    end_time = time.time()
    print_summary_report(all_results)
    
    print_colored(f"\n🎉 All demonstrations completed in {end_time - start_time:.2f}s", "green")
    print_colored("Check the generated plots and outputs for detailed results!", "white")

if __name__ == "__main__":
    main() 