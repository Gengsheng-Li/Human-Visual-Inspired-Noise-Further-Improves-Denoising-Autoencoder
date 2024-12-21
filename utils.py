import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from datetime import datetime

def create_run_name(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"LP_bs{args.batch_size}_lr{args.learning_rate}_e{args.num_epochs}_{timestamp}"

def apply_non_uniform_noise(image, center_noise_level=0.0, edge_noise_level=1.0):
    """
    Apply non-uniform noise to the image, simulating the varying sensitivity of the human eye.
    
    Args:
    image (numpy.ndarray): Input image (H x W x C).
    center_noise_level (float): Noise level at the center of the image (0 to 1).
    edge_noise_level (float): Noise level at the edges of the image (0 to 1).
    
    Returns:
    numpy.ndarray: Image with non-uniform noise applied (H x W x C).
    """
    height, width = image.shape[:2]
    center_x, center_y = (width-1) / 2, (height-1) / 2
    
    # Create a distance map from the center of the image
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    # Normalize distances and create noise level map
    normalized_dist = dist_from_center / max_dist
    noise_level_map = normalized_dist * (edge_noise_level - center_noise_level) + center_noise_level
    
    # Expand noise_level_map to match the number of channels in the image
    noise_level_map = np.expand_dims(noise_level_map, axis=-1)
    noise_level_map = np.repeat(noise_level_map, image.shape[2], axis=-1)
    
    # Generate and apply noise
    noise = np.random.normal(0, noise_level_map * 255, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def apply_motion_blurring(image, kernel_size_factor=0.1, angle=45):
    """
    Apply motion blur to the image, simulating the blur effect during rapid eye movement.
    
    Args:
    image (numpy.ndarray): Input image (H x W x C).
    kernel_size_factor (float): Factor to determine kernel size based on image dimensions.
    angle (float): Angle of motion blur in degrees.
    
    Returns:
    numpy.ndarray: Motion blurred image (H x W x C).
    """
    height, width = image.shape[:2]
    kernel_size = max(3, int(min(height, width) * kernel_size_factor))
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((kernel_size/2-0.5, kernel_size/2-0.5), angle, 1.0), (kernel_size, kernel_size))
    kernel = kernel / np.sum(kernel)
    
    # Apply kernel to image
    return cv2.filter2D(image, -1, kernel)

def apply_color_fading(image, color_fade_factor=0.7):
    """
    Apply a color perception model to the image, simulating the varying color sensitivity across the human retina.
    
    Args:
    image (numpy.ndarray): Input image (H x W x C).
    color_fade_factor (float): Factor controlling the rate of color fading from center to edge (0 to 1).
    
    Returns:
    numpy.ndarray: Image with applied color perception model (H x W x C).
    """
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Create a distance map from the center of the image
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    # Create color fade map
    normalized_dist = dist_from_center / max_dist
    color_fade_map = 1 - (normalized_dist * color_fade_factor)
    
    # Apply color fading to each channel
    b, g, r = cv2.split(image)
    channels = []
    for channel in [b, g, r]:
        faded_channel = channel.astype(np.float32) * color_fade_map + (255 * (1 - color_fade_map))
        channels.append(np.clip(faded_channel, 0, 255).astype(np.uint8))
    
    return cv2.merge(channels)

def apply_mae_noise_pixel_level(image, mask_ratio=0.75, mask_value=0):
    """
    Apply MAE masking noise to the image.
    
    Args:
    image (numpy.ndarray): Input image array of shape (H, W, C).
    mask_ratio (float): Ratio of pixels to mask (default: 0.75).
    mask_value (int or float): Value to fill in masked areas (default: 0).
    
    Returns:
    numpy.ndarray: Masked image array of shape (H, W, C).
    """
    H, W, C = image.shape
    N = H * W
    num_masked = int(mask_ratio * N)

    # Create a random permutation of pixel indices
    ids_shuffle = np.random.permutation(N)
    ids_mask = ids_shuffle[:num_masked]

    # Create the binary mask
    mask = np.zeros(N, dtype=bool)
    mask[ids_mask] = True
    mask = mask.reshape(H, W)

    # Apply masking
    masked_image = image.copy()
    masked_image[mask] = mask_value

    return masked_image

def apply_mae_noise_patch_level(image, patch_size=16, mask_ratio=0.75, mask_value=0):
    """
    Apply MAE-inspired masking noise to the image using patch-based approach.
    
    Args:
    image (numpy.ndarray): Input image array of shape (H, W, C).
    patch_size (int): Size of each patch (default: 16).
    mask_ratio (float): Ratio of patches to mask (default: 0.75).
    mask_value (int or float): Value to fill in masked patches (default: 0).
    
    Returns:
    numpy.ndarray: Masked image array of shape (H, W, C).
    """
    H, W, C = image.shape
    
    # Ensure the image dimensions are divisible by patch_size
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch_size"
    
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    
    # Create a random permutation of patch indices
    num_masked = int(mask_ratio * num_patches)
    patch_indices = np.random.permutation(num_patches)
    mask_indices = patch_indices[:num_masked]
    
    # Create the mask
    mask = np.ones((num_patches_h, num_patches_w), dtype=bool)
    mask.flat[mask_indices] = False
    
    # Apply masking
    masked_image = image.copy()
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            if not mask[i, j]:
                masked_image[i*patch_size:(i+1)*patch_size, 
                             j*patch_size:(j+1)*patch_size, :] = mask_value
    
    return masked_image

def simulate_human_vision(image, noise_params=None, color_params=None, motion_params=None, mae_params_pixel=None, mae_params_patch=None):
    """
    Apply multiple visual effects to simulate human vision, including MAE noise.
    
    Args:
    image (numpy.ndarray): Input image (H x W x C).
    noise_params (dict): Parameters for non-uniform noise (center_noise_level, edge_noise_level).
    color_params (dict): Parameters for color perception model (color_fade_factor).
    motion_params (dict): Parameters for motion blur (kernel_size_factor, angle).
    mae_params (dict): Parameters for MAE noise (mask_ratio, mask_value).
    
    Returns:
    numpy.ndarray: Image with combined visual effects applied (H x W x C).
    """
    if noise_params is not None:
        image = apply_non_uniform_noise(image, **noise_params)
    if color_params is not None:
        image = apply_color_fading(image, **color_params)        
    if motion_params is not None:
        image = apply_motion_blurring(image, **motion_params)
    if mae_params_pixel is not None:
        image = apply_mae_noise_pixel_level(image, **mae_params_pixel)
    if mae_params_patch is not None:    
        image = apply_mae_noise_patch_level(image, **mae_params_patch)
    return image

def demonstrate_visual_effects(prepro_args, image_tensor=None, image_path=None, save_fig=True):
    """
    Demonstrate various visual effects on an input image.
    
    Args:
    prepro_args (dic): Arguments needed for preprocessing.
    image_tensor (torch.Tensor, optional): Input image tensor (C x H x W).
    image_path (str, optional): Path to the input image file.
    save_fig (boolean): Wether or not save the fig.
    
    Returns:
    None: Displays a matplotlib figure with the original and processed images.
    """
    if image_path:
        # Read the image from file
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Unable to read the image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image_tensor is not None:
        # Convert torch tensor to numpy array and change from CxHxW to HxWxC
        image = image_tensor.permute(1, 2, 0).numpy()
        
        # Ensure the image is in the correct range (0-255) and data type
        image = (image * 255).astype(np.uint8)
    else:
        raise ValueError("Either image_tensor or image_path must be provided")
    
    # Apply individual effects
    uniform_noise = simulate_human_vision(
        image,
        noise_params={'center_noise_level': 0.1, 'edge_noise_level': 0.1},
    )
    noise_image = simulate_human_vision(
        image,
        noise_params={'center_noise_level': prepro_args["center_noise_level"], 'edge_noise_level': prepro_args["edge_noise_level"]},
    )
    color_image = simulate_human_vision(
        image,
        color_params={'color_fade_factor': prepro_args["color_fade_factor"]},
    )
    motion_image = simulate_human_vision(
        image,
        motion_params={'kernel_size_factor': prepro_args["kernel_size_factor"], 'angle': prepro_args["angle"]},
    )
    mae_image_pixel = simulate_human_vision(
        image,
        mae_params_pixel={'mask_ratio': 0.75, 'mask_value': 0},
    )
    mae_image_patch = simulate_human_vision(
        image,
        mae_params_patch={'patch_size': 6, 'mask_ratio': 0.75, 'mask_value': 0},
    )
    
    # Combine various noises
    noise_color_img = simulate_human_vision(
        image,
        noise_params={'center_noise_level': prepro_args["center_noise_level"], 'edge_noise_level': prepro_args["edge_noise_level"]},
        color_params={'color_fade_factor': prepro_args["color_fade_factor"]},
    )

    noise_motion_img = simulate_human_vision(
        image,
        noise_params={'center_noise_level': prepro_args["center_noise_level"], 'edge_noise_level': prepro_args["edge_noise_level"]},
        motion_params={'kernel_size_factor': prepro_args["kernel_size_factor"], 'angle': prepro_args["angle"]},
    )
    
    color_motion_img = simulate_human_vision(
        image,
        color_params={'color_fade_factor': prepro_args["color_fade_factor"]},
        motion_params={'kernel_size_factor': prepro_args["kernel_size_factor"], 'angle': prepro_args["angle"]},
    )
  
    noise_color_motion_img = simulate_human_vision(
        image,
        noise_params={'center_noise_level': prepro_args["center_noise_level"], 'edge_noise_level': prepro_args["edge_noise_level"]},
        color_params={'color_fade_factor': prepro_args["color_fade_factor"]},
        motion_params={'kernel_size_factor': prepro_args["kernel_size_factor"], 'angle': prepro_args["angle"]},
    )
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 6, figsize=(15, 10))
    fig.suptitle('Human Visual-Inspired Noise Demonstration', fontsize=16)
    
    # Display images
    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Original')

    axs[0, 1].imshow(uniform_noise)
    axs[0, 1].set_title('Uniform Noise')
    axs[0, 2].imshow(mae_image_pixel)
    axs[0, 2].set_title('MAE Noise Pixel Level')
    axs[0, 3].imshow(mae_image_patch)
    axs[0, 3].set_title('MAE Noise Patch Level')
 
    axs[0, 4].imshow(noise_image)
    axs[0, 4].set_title('Non-uniform Noise')
    axs[0, 5].imshow(color_image)
    axs[0, 5].set_title('Color Attenuation')
    axs[1, 0].imshow(motion_image)
    axs[1, 0].set_title('Eye Movement Blur')

    axs[1, 1].imshow(noise_color_img)
    axs[1, 1].set_title('Non-uniform Noise + Color Attenuation')
    axs[1, 2].imshow(noise_motion_img)
    axs[1, 2].set_title('Non-uniform Noise + Eye Movement Blur')
    axs[1, 3].imshow(color_motion_img)
    axs[1, 3].set_title('Color Attenuation + Eye Movement Blur')
    axs[1, 4].imshow(noise_color_motion_img)
    axs[1, 4].set_title('All in one')
    
    # Remove axes for better visualization
    for ax in axs.flat:
        ax.axis('off')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    # Save figure in specified directory if save_figg is True
    if save_fig:
        save_dir = "./preprocess_results"
        fig_name = (f"img{prepro_args['img_idx']}_"
                    f"noise{prepro_args['center_noise_level']}_{prepro_args['edge_noise_level']}_"
                    f"color{prepro_args['color_fade_factor']}_"
                    f"motion{prepro_args['kernel_size_factor']}_{prepro_args['angle']}.png")
        save_pth = os.path.join(save_dir, fig_name)
        plt.savefig(save_pth)
    
    
# Usage example
if __name__ == "__main__":
    # Set argumentations for preprocessing
    prepro_args = {"center_noise_level": 0.1,
                   "edge_noise_level": 0.5,
                   "color_fade_factor": 0.7,
                   "kernel_size_factor": 0.1,
                   "angle": 45,
                   "stl10_dir": 'E:\Dataset\STL10', 
                   "img_idx": None,
                   "img_pth": None,
                   }
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load STL-10 dataset
    stl10_train = torchvision.datasets.STL10(root=prepro_args['stl10_dir'], split='train', download=True, transform=transform)

    if prepro_args["img_idx"]:
        image_tensor, _ = stl10_train[prepro_args["img_idx"]]
    else:
        # Choose a random image from the dataset
        random_idx = torch.randint(0, len(stl10_train), (1,)).item()
        image_tensor, _ = stl10_train[random_idx]
        prepro_args["img_idx"] = random_idx

    if prepro_args["img_pth"]:
        # Optionally, you can also test with an image file
        demonstrate_visual_effects(prepro_args=prepro_args, image_path=prepro_args["img_pth"], save_fig=True)
    else:
        # Demonstrate visual effects on the selected image
        demonstrate_visual_effects(prepro_args=prepro_args, image_tensor=image_tensor, save_fig=True)