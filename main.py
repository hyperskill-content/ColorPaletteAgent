import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from agent import get_agent


def visualize_and_save(result, image_name, output_dir):
    """
    Visualize palette with numbered points and save to file.
    
    Args:
        result: Dictionary with 'image', 'colors', 'metadata'
        image_name: Original image filename
        output_dir: Directory to save outputs
    """
    img = result['image']
    colors = result['colors']
    
    # Create figure with image on left and palette on right
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.15)
    
    # Left: Original image with numbered points
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img)
    ax1.set_title("Original Image with Color Sampling Points", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Add numbered circles at sampling points
    for i, color_data in enumerate(colors):
        x, y = color_data['coordinates']
        # Draw white circle with black border
        circle = plt.Circle((x, y), 15, color='white', ec='black', linewidth=2, zorder=10)
        ax1.add_patch(circle)
        # Add number
        ax1.text(x, y, str(i+1), ha='center', va='center', 
                fontsize=12, fontweight='bold', color='black', zorder=11)
    
    # Right: Color palette as vertical blocks
    ax2 = fig.add_subplot(gs[1])
    
    # Create palette image with larger blocks
    block_height = 100
    palette_img = np.zeros((block_height * len(colors), 200, 3), dtype=np.uint8)
    
    for i, color_data in enumerate(colors):
        rgb = color_data['rgb']
        palette_img[i*block_height:(i+1)*block_height, :, :] = rgb
    
    ax2.imshow(palette_img)
    ax2.set_title(f"Extracted Palette ({len(colors)} Colors)", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Add color information as text below
    info_text = "\n".join([
        f"{i+1}. {c['hex']} @ ({c['coordinates'][0]}, {c['coordinates'][1]}) - {c['description']}"
        for i, c in enumerate(colors)
    ])
    
    fig.text(0.5, 0.02, info_text, ha='center', va='bottom', 
             fontsize=9, family='monospace', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    
    # Save figure
    output_path = os.path.join(output_dir, f"{image_name}_palette.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved visualization: {output_path}")
    
    return output_path


def save_metadata(result, image_name, output_dir, image_path):
    """Save palette metadata as JSON."""
    metadata = {
        'source_image': image_path,
        'timestamp': datetime.now().isoformat(),
        'agent': result['metadata']['agent'],
        'method': result['metadata']['method'],
        'n_colors': int(result['metadata']['n_colors']),
        'image_size': [int(x) for x in result['metadata']['image_size']],
        'colors': [
            {
                'index': i + 1,
                'rgb': [int(x) for x in c['rgb']],
                'hex': c['hex'],
                'percentage': float(c['percentage']),
                'coordinates': [int(x) for x in c['coordinates']],
                'description': c['description']
            }
            for i, c in enumerate(result['colors'])
        ]
    }
    
    json_path = os.path.join(output_dir, f"{image_name}_palette.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  [OK] Saved metadata: {json_path}")
    
    return json_path


def process_image(agent, image_path, output_base_dir, tag=None):
    """Process a single image and save results."""
    image_name = Path(image_path).stem
    
    print(f"\nProcessing: {image_path}")
    
    # Extract palette
    result = agent.extract_palette(image_path)
    
    # Print colors
    print(f"  Extracted {len(result['colors'])} colors:")
    for i, color in enumerate(result['colors']):
        print(f"    {i+1}. {color['hex']} - {color['description']}")
    
    # Create output directory
    agent_name = result['metadata']['agent']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if tag:
        dir_name = f"{agent_name}_{tag}_{timestamp}"
    else:
        dir_name = f"{agent_name}_{timestamp}"
    
    output_dir = os.path.join(output_base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save visualization and metadata
    visualize_and_save(result, image_name, output_dir)
    save_metadata(result, image_name, output_dir, image_path)
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Color Palette Generator')
    parser.add_argument('--image', type=str, help='Path to specific image')
    parser.add_argument('--all', action='store_true', help='Process all images in images/ folder')
    parser.add_argument('--agent', type=str, default='classic-ml', help='Agent version to use')
    parser.add_argument('--tag', type=str, help='Tag for experiment naming')
    parser.add_argument('--colors', type=int, default=5, help='Number of colors to extract')
    
    args = parser.parse_args()
    
    # Get agent
    agent = get_agent(version=args.agent, n_colors=args.colors)
    print(f"Using agent: {agent.get_agent_name()}")
    
    # Output directory
    output_base_dir = 'output'
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Determine which images to process
    if args.all:
        # Process all images in images/ folder
        images_dir = 'images'
        if not os.path.exists(images_dir):
            print(f"Error: {images_dir} directory not found.")
            return
        
        image_files = [
            os.path.join(images_dir, f) 
            for f in os.listdir(images_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]
        
        if not image_files:
            print(f"No images found in {images_dir}/")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        for image_path in image_files:
            process_image(agent, image_path, output_base_dir, args.tag)
        
        print(f"\nProcessed {len(image_files)} images!")
        
    elif args.image:
        # Process specific image
        if not os.path.exists(args.image):
            print(f"Error: {args.image} not found.")
            return
        
        output_dir = process_image(agent, args.image, output_base_dir, args.tag)
        print(f"\nDone! Results saved to: {output_dir}")
        
    else:
        # Default: process red-car.jpg
        default_image = os.path.join('images', 'red-car.jpg')
        if not os.path.exists(default_image):
            print(f"Error: {default_image} not found.")
            print("Use --image <path> or --all to specify images.")
            return
        
        output_dir = process_image(agent, default_image, output_base_dir, args.tag)
        print(f"\nDone! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

