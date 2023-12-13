"""Auxiliar functions for image disturbance"""

import numpy as np

def add_black_pixels(img, n_black_pixels):
    """Adds random black pixels to an image."""
    
    # Get image size
    img_size = img.shape[0]
    
    # Create random black pixels
    for _ in range(n_black_pixels):
        
        # Random size of black pixels
        size = np.random.randint(1, 10)
        
        # Random location of black pixels
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)
        
        # Add black pixels to image
        img[x:x+size, y:y+size, :] = 0
        
    return img