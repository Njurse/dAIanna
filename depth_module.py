#This encompasses the depth estimator system (depth map creation, interpretation to point cloud, perspective correction, collision mesh creation, and a generic form obstruction detection)
#Future plan is identify objects to respond appropriately (drive toward peds or opponents, drive away from walls and pitfalls)

import cv2
print("Importing dependencies for the depth module...")
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from skimage import measure
midas = []
device = []

# Load MiDaS model - neural network for estimating depth map information from a static image (in this case, the game)
def initialize_midas():
    global midas, device
    print("Initializing MiDaS for depth map estimation...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()
    print("MiDaS loaded!")

# Transformation for input image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384), antialias=True)
])

# Initialize the plot outside the function
fig = plt.figure('dAIanna Visualizer',figsize=(15, 8))
ax = fig.add_subplot(121, projection='3d')
plt.ion()
#plt.show()

import numpy as np

def rotate_points(points, angle_x=0.0, angle_y=0.0):
    """
    Rotate the 3D points around the X and Y axes.
    
    :param points: 3D points array [x, y, z]
    :param angle_x: rotation angle around the X-axis in radians
    :param angle_y: rotation angle around the Y-axis in radians
    :return: rotated 3D points
    """
    # Rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    
    R_y = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    
    # Combined rotation matrix
    R = R_y @ R_x
    
    # Apply rotation
    rotated_points = points @ R.T
    
    return rotated_points

def depth_map_to_3d(depth_map, fx=80.0, fy=90.0, cx=None, cy=None, scale=np.array([1, 1, 1])):
    """
    Convert a depth map to 3D points.

    :param depth_map: 2D numpy array of depth values
    :param fx: focal length in x direction (default: 640/2)
    :param fy: focal length in y direction (default: 480)
    :param cx: principal point x-coordinate (default: width / 2)
    :param cy: principal point y-coordinate (default: height / 2)
    :param scale: scaling factor for [x, y, z] coordinates (default: [1, 1, 1])
    :return: 3D points as a numpy array of [x, y, z] coordinates
    """
    depth_map = cv2.resize(depth_map, (160, 120))
    height, width = depth_map.shape

    # Set the principal point to the image center if not provided
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    # Create meshgrid of image coordinates
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y, indexing='xy')

    # Normalize coordinates based on camera intrinsics
    x_normalized = (x_grid - cx) / fx
    y_normalized = (y_grid - cy) / fy

    # Convert to 3D coordinates [x, y, z]
    points = np.dstack((x_normalized * depth_map, y_normalized * depth_map, depth_map)).reshape(-1, 3)

    # Apply scaling factor if needed
    points_scaled = points * scale
    points_scaled = rotate_points(points_scaled, 180, 0)
    #points_scaled = rotate_points(points_scaled, 0, 180)
    # Ensure the output is a 3D numpy array with correct shape
    if points_scaled.ndim != 2 or points_scaled.shape[1] != 3:
        raise ValueError("Output points array is not in the expected 2D shape (N, 3).")

    return points_scaled
    
    
#Convert the depth map to 3D by plotting pixels and estimating distance based on known resolution the game runs at (640x480)
def depth_map_to_3d_pregpt(depth_map, fx=640/2, fy=480, cx=None, cy=None):
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
    
    # Convert to 3D coordinates [x, y, z]
    points = np.dstack((x_normalized, y_normalized, depth_map)).reshape(-1, 3)
    
    # Apply scaling factor
    scale = np.array([1, 1, 1])  # Adjust this scale as per your requirement
    points_scaled = points * scale
    
    # Ensure the output is a 3D numpy array
    if points_scaled.ndim != 2 or points_scaled.shape[1] != 3:
        raise ValueError("Output points array is not in the expected 2D shape (N, 3).")
    
    return points_scaled

def find_horizon_line(depth_map, depth_threshold=85):
    """
    Attempt to identify the horizon line in the depth map and visualize the mask.

    :param depth_map: 2D numpy array of depth values
    :param depth_threshold: Threshold value to identify the horizon
    :return: y-coordinate of the horizon line
    """

    start_time = time.time()

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
    filtered_y_coords = significant_y_coords[np.abs(significant_y_coords - median_y) < height * 0.1]    
    # Calculate weighted mean along the y-axis with heavier weighting towards lower y coordinates
    weights = filtered_y_coords / np.max(filtered_y_coords)
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
    end_time = time.time()    
    processing_time = end_time - start_time
    print(f"Time to process horizon estimation: {processing_time} (Horizon estimated at {horizon_y}px)")    
    return horizon_y
    
def correct_perspective(points, depth_map, horizon_y, min_height=1.0):
    horizon_y = int(horizon_y)
    
    start_time = time.time()
    height, width = depth_map.shape
    distance_from_horizon = np.abs(points[:, 1] - horizon_y)
    reduction_factor = 1 / (1 + np.exp((distance_from_horizon + horizon_y) / 15.0))
    
    # Apply correction to z-coordinate with minimum height
    corrected_z = np.maximum(points[:, 2] * reduction_factor, min_height)
    
    # Create a new array with corrected points
    corrected_points = np.column_stack((points[:, 1], points[:, 0], corrected_z))
    
    end_time = time.time()    
    processing_time = end_time - start_time
    print(f"Time to process perspective correction: {processing_time}")
    
    return corrected_points
    
#To create the mesh we would have used Delaunay triangulation but shit sucked
def downsample_point_cloud_kmeans(points, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(points)
    downsampled_points = kmeans.cluster_centers_
    return downsampled_points



    
#temporarily here because i want to export these frames to review while working - this will be removed in a future version
frame = 0
def visualize_3d_points(points, subsample=None, horizon_y = None):
    global frame
    """
    Visualize 3D points in a matplotlib 3D scatter plot.

    :param points: 3D points as a numpy array of [x, y, z] coordinates
    :param subsample: Optional integer to subsample points for visualization
    """
    plt.ion()
    ax.clear()  # Clear the previous plot
    
    # Plot a subsample of points if specified
    if subsample:
        points = points[::subsample]

    # Use a single 's' parameter to specify marker size (s=1 in this case)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='inferno', s=1)

    # Compute processing time and update title
    formatted_title = f'3D Environment Estimation ({processing_time:.3f} seconds)'
    if horizon_y is not None:
        formatted_title += f"\nHorizon: {horizon_y}px"
        
    plt.title(formatted_title)

    # Redraw the plot
    plt.draw()
    plt.pause(2)
    frame = frame + 1
    


def depth_estimation(image):
    """
    Perform depth estimation using the MiDaS model.

    :param image: Input image as a numpy array in BGR format
    :return: Depth map as a 2D numpy array
    """
    start_time = time.time()    
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
    # Bias towards further distances faster
    epsilon = 1e-6  # Small positive offset
    depth_map = np.maximum(depth_map, epsilon)
    depth_map_transformed = np.power(depth_map, 0.9)
    # Normalize the depth map (optional, depending on visualization needs)
    depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    end_time = time.time()    
    processing_time = end_time - start_time
    print(f"Time to process MiDaS depth map estimation: {processing_time}")
    return depth_map
    

frame_number = 0


#main loop
def process_depth_mapping(image_for_depth_mapping, palette_image=None, horizon_y=None, height=120):
    """
    Process depth mapping, perspective correction, and visualization for a given image.

    :param image_for_depth_mapping: Image data or path to the input image
    :param horizon_y: y-coordinate of the horizon line
    :param height: height of the original depth map
    """
    image_for_depth_mapping = cv2.resize(image_for_depth_mapping, (160, 120))
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

    correct_perspective(points, depth_map, horizon_y, min_height=1.0)
    
    # Perform surface reconstruction using Poisson surface reconstruction
    #vertices, faces = perform_poisson_surface_reconstruction(depth_map)
    
    
    # Visualize the reconstructed mesh
    visualize_3d_points(points,5,None)



#Debug functionality for testing with a video file, main application shall use opencv2 capture of the game window
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Unable to open video.")
        os.exit(1)
        return
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Perform your calculations on each frame here
        # Example: depth estimation, perspective correction, etc.
        # For simplicity, let's just display the frame
        process_depth_mapping(frame, palette_image = None, horizon_y=190, height=240)
        # Press 'q' on keyboard to exit
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage if needed
if __name__ == "__main__":
    #image_for_depth_mapping = cv2.imread("screencap_noborder.png", cv2.IMREAD_COLOR)
    #process_depth_mapping(image_for_depth_mapping, palette_image = None, horizon_y=190, height=240)
    pass  # This script is intended to be imported and used within another script
