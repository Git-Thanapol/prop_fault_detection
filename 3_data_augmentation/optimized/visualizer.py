import matplotlib
# Fix for Tcl/Tk errors: Use 'Agg' backend for non-interactive image saving
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import config

def save_visualization(spec_tensor, output_path, cmap='viridis'):
    """
    Saves the tensor as a PNG image without axes.
    """
    # Move to CPU numpy
    # Ensure tensor is detached if it tracks gradients (unlikely here but safe)
    data = spec_tensor.detach().cpu().squeeze().numpy()
    
    # Origin is normally bottom-left for spec, but imshow expects image data convention.
    # Librosa/Matplotlib specs usually need origin='lower' in imshow.
    
    plt.figure(figsize=config.FIG_SIZE, frameon=False)
    ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
    ax.set_axis_off()
    plt.gcf().add_axes(ax)

    # origin='lower' puts low frequencies at the bottom
    ax.imshow(data, aspect='auto', cmap=cmap, origin='lower')
    
    # Ensure directory exists (redundant if caller handles it, but safe)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
