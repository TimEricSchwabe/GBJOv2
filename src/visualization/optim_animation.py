import os
import sys
import json
import pickle
import argparse
from pathlib import Path


def create_optimization_animation(animation_data, visualization_dir, query_idx, fps=10, 
                                use_tree_layout=False, max_edge_weight=1.0, last_n_steps=None):
    """
    Create an MP4 animation showing how the graph changes during optimization.
    
    Args:
        animation_data: Dictionary containing edge weights history and metadata
        visualization_dir: Directory to save the animation
        query_idx: Index of the current query (for filename)
        fps: Frames per second for the animation
        use_tree_layout: If True, use a tree layout; otherwise use force-directed
        max_edge_weight: Maximum edge weight for normalization
        last_n_steps: If provided, only animate the last N steps of optimization
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    
    if animation_data is None or not animation_data['edge_weights_history']:
        print("No animation data available")
        return
    
    print(f"Creating optimization animation for query {query_idx}")
    print(f"Total available frames: {len(animation_data['edge_weights_history'])}")
    
    # Extract data
    edge_weights_history = animation_data['edge_weights_history']
    step_numbers = animation_data['step_numbers']
    edge_index = animation_data['edge_index']
    n_nodes = animation_data['n_nodes']
    triples_num = animation_data['triples_num']
    cost_history = [np.exp(cost) for cost in animation_data.get('cost_history', [])]
    penalty_history = animation_data.get('penalty_history', [])
    
    if last_n_steps is not None and last_n_steps > 0:
        if last_n_steps < len(edge_weights_history):
            edge_weights_history = edge_weights_history[-last_n_steps:]
            step_numbers = step_numbers[-last_n_steps:]
            if cost_history:
                cost_history = cost_history[-last_n_steps:]
            if penalty_history:
                penalty_history = penalty_history[-last_n_steps:]
            print(f"Using only the last {last_n_steps} steps for animation")
        else:
            print(f"Requested {last_n_steps} steps, but only {len(edge_weights_history)} available. Using all steps.")
    
    print(f"Animation frames: {len(edge_weights_history)}")
    
    # Use FINAL weights to determine layout (what we're converging to)
    final_weights = edge_weights_history[-1]
    
    # Create a NetworkX graph for layout based on final adjacency
    G_layout = nx.DiGraph()
    G_layout.add_nodes_from(range(n_nodes))
    
    # Add all possible edges to track for animation
    all_edges = []
    for i in range(len(edge_index[0])):
        src, dst = edge_index[0][i].item(), edge_index[1][i].item()
        all_edges.append((src, dst))
    
    # For layout, only add strong edges from the final weights (like visualize_adjacency_matrix does)
    strong_edge_threshold = 0.5
    for i, (src, dst) in enumerate(all_edges):
        final_weight = final_weights[i]
        if hasattr(final_weight, 'item'):
            final_weight = final_weight.item()
        final_weight = float(final_weight)
        
        if final_weight > strong_edge_threshold:
            G_layout.add_edge(src, dst, weight=final_weight)
    
    # Determine layout using final adjacency structure
    if use_tree_layout:
        # Find the root node from final weights - join node with no outgoing edges
        root = n_nodes - 1  # Default fallback
        try:
            for node_idx in range(triples_num, n_nodes):
                out_weights = [final_weights[i] for i, (src, dst) in enumerate(zip(edge_index[0], edge_index[1])) 
                              if src.item() == node_idx]
                if sum(out_weights) < 0.01:
                    root = node_idx
                    break
            
            # Try tree layout with the identified root
            pos = nx.drawing.nx_agraph.graphviz_layout(G_layout, prog='dot', root=root)
            print(f"Using tree layout with root = {root}")
        except Exception as e:
            print(f"Tree layout failed: {e}, falling back to circular layout")
            pos = nx.circular_layout(G_layout)
    else:
        pos = nx.circular_layout(G_layout)
        print("Using circular layout")
    
    # Create the figure and subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Main graph subplot
    ax_graph = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    
    # Cost subplot
    ax_cost = plt.subplot2grid((2, 3), (0, 2))
    
    # Penalty subplot  
    ax_penalty = plt.subplot2grid((2, 3), (1, 2))
    
    # Create node colors and labels
    node_colors = []
    labels = {}
    for i in range(n_nodes):
        if i < triples_num:
            node_colors.append('lightblue')
            labels[i] = f"T{i}"
        else:
            node_colors.append('lightcoral')
            labels[i] = "⋈"  # Bowtie symbol for join nodes
    
    def update(frame):
        # Clear all axes
        for ax in [ax_graph, ax_cost, ax_penalty]:
            ax.clear()
        
        # Get current edge weights
        current_weights = edge_weights_history[frame]
        current_step = step_numbers[frame]
        
        # Create edge weight mapping - ensure all weights are scalars
        edge_weights = {}
        for i in range(len(all_edges)):
            weight = current_weights[i]
            # Convert to scalar if it's an array
            if hasattr(weight, 'item'):
                weight = weight.item()
            edge_weights[all_edges[i]] = float(weight)
        
        # Normalize weights for visualization - ensure scalar values
        weight_values = list(edge_weights.values())
        max_current_weight = max(weight_values) if weight_values else 1.0
        max_current_weight = float(max_current_weight)
        
        # Draw the graph
        ax_graph.set_title(f"Query Optimization - Step {current_step}", fontsize=14, fontweight='bold')
        
        # Draw nodes
        nx.draw_networkx_nodes(G_layout, pos, node_color=node_colors, node_size=800, 
                              edgecolors='black', linewidths=2, ax=ax_graph)
        
        # Filter significant edges and draw them
        significant_edges = []
        edge_colors = []
        edge_widths = []
        
        edge_threshold = 0.005  # Lower threshold to show more edges
        
        for edge, weight in edge_weights.items():
            weight = float(weight)  # Ensure scalar
            if weight > edge_threshold:
                significant_edges.append(edge)
                # Color intensity based on weight
                color_intensity = weight / max_current_weight
                edge_colors.append(plt.cm.Reds(color_intensity))
                # Width based on weight (1 to 6 pixels)
                edge_widths.append(1 + weight * 5)
        
        # Draw significant edges
        if significant_edges:
            nx.draw_networkx_edges(G_layout, pos, edgelist=significant_edges, 
                                  edge_color=edge_colors, width=edge_widths,
                                  arrows=True, arrowstyle='-|>', arrowsize=20,
                                  alpha=0.8, ax=ax_graph)
            
            # Add weight labels on strong edges
            strong_edges = [(edge, edge_weights[edge]) for edge in significant_edges if float(edge_weights[edge]) > 0.3]
            for (src, dst), weight in strong_edges:
                x = (pos[src][0] + pos[dst][0]) / 2
                y = (pos[src][1] + pos[dst][1]) / 2
                ax_graph.text(x, y, f'{weight:.2f}', fontsize=8, ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Draw labels
        nx.draw_networkx_labels(G_layout, pos, labels=labels, font_size=12, 
                               font_color='black', font_weight='bold', ax=ax_graph)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='black', label='Triple Patterns'),
            Patch(facecolor='lightcoral', edgecolor='black', label='Join Nodes')
        ]
        ax_graph.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        # Add info text
        num_active_edges = len(significant_edges)
        max_weight_current = max(weight_values) if weight_values else 0
        max_weight_current = float(max_weight_current)

        
        ax_graph.axis('off')
        
        # Cost plot
        if cost_history and frame < len(cost_history):
            costs_so_far = cost_history[:frame+1]
            steps_so_far = step_numbers[:frame+1]
            
            # Ensure all values are scalars
            costs_so_far = [float(c) for c in costs_so_far]
            steps_so_far = [int(s) for s in steps_so_far]
            
            ax_cost.plot(steps_so_far, costs_so_far, 'b-', linewidth=2)
            ax_cost.scatter([current_step], [costs_so_far[-1]], color='red', s=50, zorder=5)
            ax_cost.set_title('Cost', fontsize=12, fontweight='bold')
            ax_cost.set_ylabel('Cost')
            ax_cost.set_yscale('log')  # Make y-axis logarithmic
            ax_cost.grid(True, alpha=0.3)
            if len(costs_so_far) > 1:
                cost_range = float(max(costs_so_far) - min(costs_so_far))
                if cost_range > 0:
                    min_cost = float(min(costs_so_far))
                    max_cost = float(max(costs_so_far))
                    ax_cost.set_ylim(min_cost - 0.1*cost_range, max_cost + 0.1*cost_range)
        
        # Penalty plot
        if penalty_history and frame < len(penalty_history):
            penalties_so_far = penalty_history[:frame+1]
            steps_so_far = step_numbers[:frame+1]
            
            # Ensure all values are scalars
            penalties_so_far = [float(p) for p in penalties_so_far]
            steps_so_far = [int(s) for s in steps_so_far]
            
            ax_penalty.plot(steps_so_far, penalties_so_far, 'r-', linewidth=2)
            ax_penalty.scatter([current_step], [penalties_so_far[-1]], color='red', s=50, zorder=5)
            ax_penalty.set_title('Constraint Violation', fontsize=12, fontweight='bold')
            ax_penalty.set_ylabel('Penalty')
            ax_penalty.set_yscale('log')  # Make y-axis logarithmic
            ax_penalty.grid(True, alpha=0.3)
            if len(penalties_so_far) > 1:
                penalty_range = float(max(penalties_so_far) - min(penalties_so_far))
                if penalty_range > 0:
                    min_penalty = float(min(penalties_so_far))
                    max_penalty = float(max(penalties_so_far))
                    ax_penalty.set_ylim(min_penalty - 0.1*penalty_range, max_penalty + 0.1*penalty_range)
        
        plt.tight_layout()
    
    # Create animation
    num_frames = len(edge_weights_history)
    anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000//fps, 
                                  blit=False, repeat=True)
    
    # Save animation
    layout_type = "tree" if use_tree_layout else "circular"
    steps_suffix = f"_last{last_n_steps}" if last_n_steps is not None else ""
    animation_filename = f"{visualization_dir}/optimization_animation_query_{query_idx}_{layout_type}{steps_suffix}.mp4"
    
    try:
        print(f"Saving animation to {animation_filename}")
        anim.save(animation_filename, writer='ffmpeg', fps=fps, bitrate=1800)
        print(f"Animation saved successfully")
    except Exception as e:
        print(f"Failed to save animation: {e}")
        print("Trying alternative writer...")
        try:
            anim.save(animation_filename, writer='pillow', fps=fps)
            print(f"Animation saved with pillow writer")
        except Exception as e2:
            print(f"Failed to save with pillow: {e2}")
    
    plt.close(fig)
    return animation_filename





def load_animation_data(animation_data_file):
    """
    Load animation data from a pickle file.
    
    Args:
        animation_data_file: Path to the pickle file containing animation data
        
    Returns:
        Dictionary containing animation data or None if loading fails
    """
    try:
        with open(animation_data_file, 'rb') as f:
            animation_data = pickle.load(f)
        return animation_data
    except Exception as e:
        print(f"Error loading animation data from {animation_data_file}: {e}")
        return None


def generate_animations_for_run(run_directory, fps=10, use_tree_layout=True, max_edge_weight=2.0, last_n_steps=None):
    """
    Generate all optimization animations for a given run directory.
    
    Args:
        run_directory: Path to the directory containing saved animation data
        fps: Frames per second for the animations
        use_tree_layout: Whether to use tree layout for the graphs
        max_edge_weight: Maximum edge weight for normalization
        last_n_steps: If provided, only animate the last N steps of optimization
    """
    run_path = Path(run_directory)
    
    # Check if the run directory exists
    if not run_path.exists():
        print(f"Error: Run directory {run_directory} does not exist")
        return False
    
    # Load animation metadata
    metadata_file = run_path / "animation_metadata.json"
    if not metadata_file.exists():
        print(f"Error: Animation metadata file {metadata_file} not found")
        return False
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading animation metadata: {e}")
        return False
    
    animation_data_dir = Path(metadata['animation_data_dir'])
    visualization_dir = Path(metadata['visualization_dir'])
    num_queries = metadata['num_queries']
    
    # Ensure visualization directory exists
    os.makedirs(visualization_dir, exist_ok=True)
    
    print(f"Generating animations for {num_queries} queries...")
    print(f"Animation data directory: {animation_data_dir}")
    print(f"Visualization directory: {visualization_dir}")
    print(f"Animation parameters: fps={fps}, tree_layout={use_tree_layout}, max_edge_weight={max_edge_weight}")
    if last_n_steps is not None:
        print(f"Using only the last {last_n_steps} steps for each animation")
    
    # Track success/failure statistics
    successful_animations = 0
    failed_animations = 0
    
    # Generate animation for each query
    for query_idx in range(num_queries):
        animation_data_file = animation_data_dir / f"query_{query_idx}_animation_data.pkl"
        
        if not animation_data_file.exists():
            print(f"Warning: Animation data file for query {query_idx} not found: {animation_data_file}")
            failed_animations += 1
            continue
        
        print(f"Generating animation for query {query_idx}...")
        
        # Load animation data
        animation_data = load_animation_data(animation_data_file)
        if animation_data is None:
            print(f"Failed to load animation data for query {query_idx}")
            failed_animations += 1
            continue
        
        # Generate the animation
        try:
            animation_filename = create_optimization_animation(
                animation_data=animation_data,
                visualization_dir=str(visualization_dir),
                query_idx=query_idx,
                fps=fps,
                use_tree_layout=use_tree_layout,
                max_edge_weight=max_edge_weight,
                last_n_steps=last_n_steps
            )
            
            if animation_filename:
                print(f"Successfully created animation: {animation_filename}")
                successful_animations += 1
            else:
                print(f"Failed to create animation for query {query_idx}")
                failed_animations += 1
                
        except Exception as e:
            print(f"Error creating animation for query {query_idx}: {e}")
            failed_animations += 1
    
    # Print summary
    print(f"\nAnimation generation complete!")
    print(f"Successful animations: {successful_animations}")
    print(f"Failed animations: {failed_animations}")
    print(f"Total queries: {num_queries}")
    
    if successful_animations > 0:
        print(f"Animations saved to: {visualization_dir}")
    
    return successful_animations > 0

def main():
    """Main function to generate animations."""
    # Configuration parameters
    run_directory = "optimization_results/run_20250606_123336"
    fps = 10
    use_tree_layout = True
    max_edge_weight = 2.0
    last_n_steps = None 

    # Generate animations
    success = generate_animations_for_run(
        run_directory=run_directory,
        fps=fps,
        use_tree_layout=use_tree_layout,
        max_edge_weight=max_edge_weight,
        last_n_steps=last_n_steps
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
