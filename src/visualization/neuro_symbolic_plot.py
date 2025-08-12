import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_wiggly_landscape(x, y):
    """
    Create a smoother mathematical landscape with:
    - A very sharp global minimum in the center
    - Several visible but shallow local minima around it
    - Mostly flat surface with gentle wiggly texture
    """
    # Base level - mostly flat
    base_level = 0.5
    
    # Very sharp central minimum (much narrower Gaussian well, reduced depth)
    central_well = -2.0 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.003)
    
    # Gentle wiggly texture on the flat surface
    wiggles = (
        0.1 * np.sin(4 * np.pi * x) * np.sin(4 * np.pi * y) +
        0.08 * np.sin(6 * np.pi * x) * np.cos(5 * np.pi * y) +
        0.06 * np.cos(8 * np.pi * x) * np.sin(7 * np.pi * y)
    )
    
    # More prominent local minima - visible but not too deep
    local_minima = (
        # Ring of local minima around the center
        -0.8 * np.exp(-((x - 0.25)**2 + (y - 0.25)**2) / 0.008) +
        -0.7 * np.exp(-((x - 0.75)**2 + (y - 0.25)**2) / 0.008) +
        -0.8 * np.exp(-((x - 0.75)**2 + (y - 0.75)**2) / 0.008) +
        -0.7 * np.exp(-((x - 0.25)**2 + (y - 0.75)**2) / 0.008) +
        
        # Additional scattered minima
        -0.6 * np.exp(-((x - 0.15)**2 + (y - 0.5)**2) / 0.007) +
        -0.5 * np.exp(-((x - 0.85)**2 + (y - 0.5)**2) / 0.007) +
        -0.6 * np.exp(-((x - 0.5)**2 + (y - 0.15)**2) / 0.007) +
        -0.5 * np.exp(-((x - 0.5)**2 + (y - 0.85)**2) / 0.007) +
        
        # Some edge minima
        -0.4 * np.exp(-((x - 0.1)**2 + (y - 0.3)**2) / 0.006) +
        -0.4 * np.exp(-((x - 0.9)**2 + (y - 0.7)**2) / 0.006) +
        -0.3 * np.exp(-((x - 0.3)**2 + (y - 0.1)**2) / 0.006) +
        -0.3 * np.exp(-((x - 0.7)**2 + (y - 0.9)**2) / 0.006)
    )
    
    # Combine all components
    landscape = base_level + central_well + wiggles + local_minima
    
    return landscape


def plot_neuro_symbolic_landscape(resolution=100, clean_plot=False):
    """
    Create and display a 3D plot of the wiggly landscape.
    
    Args:
        resolution: Number of grid points along each axis (default increased to 100)
        clean_plot: If True, remove axes and labels for minimal visualization
    """
    # Create coordinate grids
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute the landscape
    Z = create_wiggly_landscape(X, Y)
    
    # Create 3D plot in horizontal landscape format
    fig = plt.figure(figsize=(16, 9))  # Changed to horizontal landscape format
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface with nice colormap and increased transparency
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.7, 
                          linewidth=0, antialiased=True, edgecolor='none')
    
    if clean_plot:
        # Remove all axes, grids, and labels for minimal visualization
        ax.set_axis_off()
        filename = 'neuro_symbolic_landscape_clean.png'
    else:
        # Normal plot with labels and annotations
        ax.set_xlabel('Dimension X', fontsize=12)
        ax.set_ylabel('Dimension Y', fontsize=12)
        ax.set_zlabel('Cost/Energy', fontsize=12)
        ax.set_title('Neuro-Symbolic Optimization Landscape', fontsize=14)
        
        # Add colorbar with transparency
        fig.colorbar(surf, shrink=0.6, aspect=15, alpha=0.8)
        
        filename = 'neuro_symbolic_landscape.png'
    
    # Set viewing angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Find global minimum for reference
    min_idx = np.unravel_index(Z.argmin(), Z.shape)
    global_min_z = Z[min_idx]
    
    print(f"Global minimum depth: {global_min_z:.3f}")
    print(f"Landscape plot saved as: {filename}")


def plot_contour_view(resolution=150):
    """
    Create a 2D contour plot of the same landscape for additional visualization.
    """
    # Create coordinate grids
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute the landscape
    Z = create_wiggly_landscape(X, Y)
    
    # Create contour plot in horizontal format
    fig, ax = plt.subplots(figsize=(14, 8))  # Changed to horizontal format
    
    # Plot filled contours with transparency
    contour_filled = ax.contourf(X, Y, Z, levels=30, cmap='plasma', alpha=0.7)
    
    # Plot contour lines
    contour_lines = ax.contour(X, Y, Z, levels=30, colors='black', alpha=0.3, linewidths=0.5)
    
    # Add colorbar
    cbar = fig.colorbar(contour_filled, ax=ax)
    cbar.set_label('Cost/Energy', fontsize=12)
    
    ax.set_xlabel('Dimension X', fontsize=12)
    ax.set_ylabel('Dimension Y', fontsize=12)
    ax.set_title('Neuro-Symbolic Landscape - Contour View', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('neuro_symbolic_contour.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Create the 3D landscape plot with fine mesh
    print("Creating 3D neuro-symbolic landscape with fine mesh...")
    plot_neuro_symbolic_landscape(resolution=120, clean_plot=False)
    
    # Also create a clean version without labels
    print("\nCreating clean version...")
    plot_neuro_symbolic_landscape(resolution=120, clean_plot=True)
    
    # Create contour plot for additional perspective
    print("\nCreating contour view...")
    plot_contour_view(resolution=150)
