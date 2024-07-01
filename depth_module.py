import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Transformation for input image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384), antialias=True)
])

# Initialize the plot outside the function
fig = plt.figure('dAIanna Visualizer')  # Create a new figure
ax = fig.add_subplot(111, projection='3d')  # Add 3D subplot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Points Visualization')
plt.ion()
plt.show()

def correct_perspective(points, depth_map, horizon_y, height):
    """
    Correct the perspective distortion in 3D points based on a depth map, using inverse perspective transformation.
    Adjusts z-coordinate based on distance from the horizon line and removes points beyond a certain threshold.
    
    :param points: 3D points as a numpy array of [x, y, z] coordinates
    :param depth_map: Original depth map used for the depth estimation
    :param horizon_y: y-coordinate of the horizon line
    :param height: height of the original depth map (not directly used in this function)
    :return: Corrected 3D points
    """
    horizon_y = int(horizon_y)
    
    # Adjust z-coordinate based on distance from the horizon line and remove points with excessive depth
    adjusted_points = []
    for i in range(points.shape[0]):
        x = points[i, 0]
        y = points[i, 1]
        z = points[i, 2]
        
        # Calculate distance from horizon
        distance_from_horizon = abs(y - horizon_y)
        
        # Inverse perspective correction with aggressive reduction
        if distance_from_horizon > 0:
            # Adjust logarithmic reduction factor based on distance from the horizon
            reduction_factor = np.exp(-distance_from_horizon / 30.0)
            z_corrected = z * reduction_factor
            
            # Remove points that exceed a depth threshold (e.g., 230)
            if y:
                # Add corrected point to the list
                adjusted_points.append([x, y, z_corrected])
    
    # Convert adjusted points to numpy array
    adjusted_points = np.array(adjusted_points)
    
    return adjusted_points

def depth_map_to_3d(depth_map, fx=640, fy=480, cx=None, cy=None):
    """
    Convert a depth map to 3D points.
    
    :param depth_map: 2D numpy array of depth values
    :param fx: focal length in x direction (default: 640)
    :param fy: focal length in y direction (default: 480)
    :param cx: principal point x-coordinate (default: width / 2)
    :param cy: principal point y-coordinate (default: height / 2)
    :return: 3D points as a numpy array of [x, y, z] coordinates
    """
    height, width = depth_map.shape
    
    if cx is None:
        cx = width / 2
    if cy is None:
        cy = height / 2
    
    # Create meshgrid of image coordinates
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y, indexing='xy')
    
    # Normalize coordinates based on camera intrinsics
    x_normalized = (x_grid - cx) / fx
    y_normalized = (y_grid - cy) / fy
    
    # Reshape to create points in [x, y, z] format
    points = np.dstack((x_normalized, -depth_map, -y_normalized)).reshape(-1, 3)
    
    return points

def find_horizon_line(depth_map, depth_threshold=97):
    """
    Attempt to identify the horizon line in the depth map and visualize the mask.

    :param depth_map: 2D numpy array of depth values
    :param depth_threshold: Threshold value to identify the horizon
    :return: y-coordinate of the horizon line
    """
    height, width = depth_map.shape
    
    # Create a binary mask based on the depth threshold
    mask = depth_map > depth_threshold
    
    # Visualize the mask using OpenCV
    mask_visualization = np.uint8(mask * 255)  # Convert boolean mask to uint8 for visualization
    mask_visualization = cv2.cvtColor(mask_visualization, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color image display
    
    # Calculate the mean y-coordinates for rows with significant depth
    y_coords = np.arange(height).reshape(-1, 1)
    significant_y_coords = y_coords[mask.any(axis=1)]
    
    # Calculate the median of significant y-coordinates
    median_y = np.median(significant_y_coords)
    
    # Filter out values that deviate significantly from the median
    filtered_y_coords = significant_y_coords[np.abs(significant_y_coords - median_y) < height * 0.20]
    
    # Calculate weighted mean along the y-axis with heavier weighting towards lower y coordinates
    weights = np.exp(filtered_y_coords)  # Exponential scaling
    weighted_mean_y = np.sum(filtered_y_coords * weights) / np.sum(weights)
    
    # If no significant depth pixels are found, return the bottom of the image
    if np.isnan(weighted_mean_y) or len(filtered_y_coords) == 0:
        horizon_y = height - 1
    else:
        horizon_y = int(weighted_mean_y)
    
    # Draw a red line indicating the horizon on the mask visualization
    
    #mask_preview = mask_visualization
    #cv2.line(mask_preview, (0, horizon_y), (width - 1, horizon_y), (0, 0, 255), thickness=2)
    
    # Display the mask with the horizon line
    #cv2.imshow('Foreground Mask / Horizon Estimation', mask_preview)
    #cv2.waitKey(1)  # Wait for any key press to close (blocking)
    
    return horizon_y

def visualize_3d_points(points, subsample=None, horizon_y = None):
    """
    Visualize 3D points in a matplotlib 3D scatter plot.

    :param points: 3D points as a numpy array of [x, y, z] coordinates
    :param subsample: Optional integer to subsample points for visualization
    """
    ax.clear()  # Clear the previous plot
    
    # Plot a subsample of points if specified
    if subsample:
        points = points[::subsample]

    # Use a single 's' parameter to specify marker size (s=1 in this case)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='inferno', s=1)

    # Compute processing time and update title
    formatted_title = f'dAIanna Visualizer - Time to Process: {processing_time:.3f} seconds'
    if horizon_y is not None:
        formatted_title += f"\nHorizon: {horizon_y}px"
        
    plt.title(formatted_title)

    # Redraw the plot
    plt.draw()
    plt.pause(0.1)

def depth_estimation(image):
    """
    Perform depth estimation using the MiDaS model.

    :param image: Input image as a numpy array in BGR format
    :return: Depth map as a 2D numpy array
    """
    # Convert image to RGB (MiDaS expects RGB input)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess the image
    input_batch = transform(image_rgb).unsqueeze(0).to(device)

    # Perform depth estimation
    with torch.no_grad():
        start_time = time.time()
        prediction = midas(input_batch)
        end_time = time.time()

    # Calculate processing time
    global processing_time
    processing_time = end_time - start_time
    
    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    # Convert to numpy array
    depth_map = prediction.cpu().numpy() 
    
    # Normalize the depth map (optional, depending on visualization needs)
    depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    return depth_map

def process_depth_mapping(image_for_depth_mapping, palette_image = None, horizon_y=190, height=240):
    """
    Process depth mapping, perspective correction, and visualization for a given image.

    :param image_for_depth_mapping: Image data or path to the input image
    :param horizon_y: y-coordinate of the horizon line
    :param height: height of the original depth map
    """
    # Load image if image_for_depth_mapping is a path
    if isinstance(image_for_depth_mapping, str):
        screen_image = cv2.imread(image_for_depth_mapping, cv2.IMREAD_COLOR)
    else:
        screen_image = image_for_depth_mapping

    if isinstance(palette_image, str):
        palette_image = cv2.imread(palette_image, cv2.IMREAD_COLOR)
    
    if screen_image is None:
        raise ValueError(f"Error: Unable to load image from {image_for_depth_mapping}")

    # Perform depth estimation
    depth_map = depth_estimation(screen_image)
    
    # Find horizon line in the depth map
    horizon_y = find_horizon_line(depth_map)
    
    # Convert depth map to 3D points
    points = depth_map_to_3d(depth_map)
    
    # Correct perspective distortion in 3D points
    points_corrected = correct_perspective(points, depth_map, horizon_y, height)
    
    # Visualize 3D points
    visualize_3d_points(points_corrected, subsample=50, horizon_y = horizon_y)

# Example usage if needed
if __name__ == "__main__":
    pass  # This script is intended to be imported and used within another script
