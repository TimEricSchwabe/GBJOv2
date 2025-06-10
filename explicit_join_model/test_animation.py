#!/usr/bin/env python3
"""
Test script for the optimization animation functionality.
This creates animations for a small number of queries to verify the implementation.
"""

import os
import sys
import json
from datetime import datetime

# Add the parent directory to the path to import the main module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimization_evaluation_leftlinear import (
    load_sparql_queries, 
    evaluate_optimization, 
    optimize_query_gumbel
)

def main():
    """Test the animation functionality with a small configuration."""
    
    # Test configuration with animation enabled
    config = {
        # General parameters
        'queries_file': "/home/tim/query_optimization/datasets/sparql_queries_4_tp/queries.pkl",
        'model_path': "/home/tim/query_optimization/explicit_join_model/models/join_plus_tp_prediction_all_sizes.pt",
        'num_queries': 3,  # Small number for testing
        'optimization_steps': 100,  # Shorter optimization for testing
        'verbose': True,
        'save_path': "animation_test_results",
        
        # Query optimization hyperparameters
        'optimization_params': {
            # Optimization procedure selection
            'optimization_procedure': 'gumbel',
            
            # Optimizer parameters
            'learning_rate': 0.25,
            
            # Penalty weights (using simplified values for testing)
            'lambda_acyclic': 1000.0,
            'lambda_triple_in': 1000.0,
            'lambda_triple_out': 1000.0,
            'lambda_join_in': 1000.0,
            'lambda_join_out': 1000.0,
            'lambda_entropy': 0.0,
            'lambda_total_penalty': 1.0,
            'lambda_left_linear': 1000.0,
            
            # Gumbel-Sigmoid specific parameters
            'init_tau': 10.0,
            'min_tau': 1.0,
            'tau_decay': 0.95,
            'use_temperature_annealing': True,
            
            # Solution selection and penalty ramping
            'return_best': True,
            'min_penalty_threshold': 30.0,
            'use_lambda_ramping': True,
            
            # Sampling method selection
            'logit_sampling': 'dual-softmax',
            
            # Animation parameters - ENABLED FOR TESTING
            'save_animation_data': True,
            'animation_save_interval': 5,  # Save every 5 steps for more frames
        }
    }
    
    # Create unique save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join(config['save_path'], f"animation_test_{timestamp}")
    os.makedirs(save_directory, exist_ok=True)
    
    print(f"Testing animation functionality")
    print(f"Results will be saved to: {save_directory}")
    
    # Save configuration
    config_copy = config.copy()
    config_copy['save_directory'] = save_directory
    config_copy['timestamp'] = timestamp
    with open(os.path.join(save_directory, "config.json"), 'w') as f:
        json.dump(config_copy, f, indent=2)
    
    # Print configuration
    print("\nTest configuration:")
    print(f"Number of queries: {config['num_queries']}")
    print(f"Optimization steps: {config['optimization_steps']}")
    print(f"Animation enabled: {config['optimization_params']['save_animation_data']}")
    print(f"Animation save interval: {config['optimization_params']['animation_save_interval']}")
    
    # Load queries
    sparql_queries = load_sparql_queries(config['queries_file'], config['num_queries'])
    
    # Select optimization function
    optimization_procedure = config['optimization_params'].pop('optimization_procedure')
    if optimization_procedure == 'gumbel':
        optimization_function = optimize_query_gumbel
    else:
        raise ValueError(f"Unknown optimization procedure: {optimization_procedure}")
    
    # Run evaluation with animation
    try:
        stats = evaluate_optimization(
            sparql_queries, 
            config['model_path'],
            num_queries=config['num_queries'],
            optimization_steps=config['optimization_steps'],
            verbose=config['verbose'],
            optimization_params=config['optimization_params'],
            optimization_function=optimization_function,
            save_directory=save_directory
        )
        
        print(f"\nAnimation test completed successfully!")
        print(f"Check {save_directory} for:")
        print("- MP4 animation files (optimization_animation_query_*.mp4)")
        print("- Static visualizations")
        print("- Configuration and statistics")
        
    except Exception as e:
        print(f"Error during animation test: {e}")
        raise

if __name__ == "__main__":
    main() 