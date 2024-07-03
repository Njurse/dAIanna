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


#Convert the depth map to 3D by plotting pixels and estimating distance based on known resolution the game runs at (640x480)
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
    reduction_factor = np.exp(-distance_from_horizon / 15.0)
    
    # Apply correction to z-coordinate with minimum height
    corrected_z = np.maximum(points[:, 2] * reduction_factor, min_height)
    
    # Create a new array with corrected points
    corrected_points = np.column_stack((points[:, 0], points[:, 1], corrected_z))
    
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
    depth_map_transformed = np.power(depth_map, 0.7)
    # Normalize the depth map (optional, depending on visualization needs)
    depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    end_time = time.time()    
    processing_time = end_time - start_time
    print(f"Time to process MiDaS depth map estimation: {processing_time}")
    return depth_map
    

def depth_map_to_voxel_grid(depth_map, fx=320, fy=240, cx=None, cy=None, voxel_size=40.0):
    """
    Convert a depth map to a voxel grid and render it as a 3D texture.

    :param depth_map: 2D numpy array of depth values
    :param fx: focal length in x direction (default: 320)
    :param fy: focal length in y direction (default: 240)
    :param cx: principal point x-coordinate (default: width / 2)
    :param cy: principal point y-coordinate (default: height / 2)
    :param voxel_size: size of each voxel in millimeters (default: 40.0)
    :return: 3D voxel grid representing the volumetric data suitable for Poisson reconstruction
    """
    height, width = depth_map.shape
    start_time = time.time()    

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
    
    # Calculate voxel grid dimensions
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    z_range = int(np.ceil((max_depth - min_depth) / voxel_size))
    
    # Create voxel grid
    voxel_grid = np.zeros((height, z_range, width), dtype=bool)
    
    # Assign points to voxel grid
    for point in points:
        x_idx = int(round(point[0] * fx + cx))
        y_idx = int(round(point[1] * fy + cy))
        z_idx = int(round((point[2] - min_depth) / voxel_size))
        
        if 0 <= x_idx < width and 0 <= y_idx < height and 0 <= z_idx < z_range:
            voxel_grid[y_idx, z_idx, x_idx] = True

    end_time = time.time()    
    processing_time = end_time - start_time
    print(f"Time to process voxel grid construction: {processing_time}")

    # Render the voxel grid and save it as a 3D texture to a file
    render_voxel_grid(voxel_grid)
 
    return voxel_grid

def create_voxel_grid_from_bool(voxel_grid):
    """
    Create a vtkImageData object representing a voxel grid from a boolean array.

    :param voxel_grid: 3D numpy boolean array representing the voxel grid
    :return: vtk.vtkImageData object representing the voxel grid
    """
    dimensions = voxel_grid.shape
    grid = vtk.vtkImageData()
    grid.SetDimensions(dimensions[2], dimensions[0], dimensions[1])  # Note the correct order
    grid.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    for z in range(dimensions[1]):
        for y in range(dimensions[0]):
            for x in range(dimensions[2]):
                grid.SetScalarComponentFromDouble(x, y, z, 0, 255 if voxel_grid[y, z, x] else 0)

    return grid
import vtk
def render_voxel_grid(voxel_grid):
    """
    Render the voxel grid and save it as a 3D texture to a file.

    :param voxel_grid: 3D numpy boolean array representing the voxel grid
    """
    # Create vtkImageData from voxel grid
    grid = create_voxel_grid_from_bool(voxel_grid)

    # Create the mapper and volume
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(grid)

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()
    volume.SetProperty(volumeProperty)

    # Create the renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    renderer.AddVolume(volume)
    renderer.SetBackground(1, 1, 1)  # Set background to white
    renderWindow.SetSize(600, 600)
    renderWindow.Render()

    # Capture the image and save it
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(renderWindow)
    window_to_image_filter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName("voxel_grid.png")
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.Write()

    # Initialize and start the interactor
    interactor.Initialize()
    interactor.CreateRepeatingTimer(1)  # Create a repeating timer to keep the window responsive
    interactor.Start()
def perform_poisson_surface_reconstruction(depth_map, fx=320, fy=480, cx=None, cy=None):
    """
    Perform Poisson surface reconstruction.

    :param depth_map: 2D numpy array of depth values
    :param fx: focal length in x direction (default: 640)
    :param fy: focal length in y direction (default: 480)
    :param cx: principal point x-coordinate (default: width / 2)
    :param cy: principal point y-coordinate (default: height / 2)
    :return: vertices and faces of the reconstructed mesh
    """
    height, width = depth_map.shape
    voxel_size = 50
    start_time = time.time()
    if cx is None:
        cx = width / 2
    if cy is None:
        cy = height / 2
    fy = find_horizon_line(depth_map)/2
    
    # Convert depth map to voxel grid
    voxel_grid = depth_map_to_voxel_grid(depth_map, fx, fy, cx, cy, voxel_size)
    
    # Use marching cubes to extract mesh
    try:
        # Adjust the spacing based on your voxel_size
        spacing = (voxel_size, voxel_size, voxel_size)
        print(f"Marching cubes with voxel spacing of {spacing}")
        mesh = measure.marching_cubes(voxel_grid, spacing=spacing)
    except RuntimeError as e:
        print(f"RuntimeError occurred: {e}")
        # Optionally, visualize or log voxel_grid and other parameters for debugging
        raise e
    
    # Extract vertices and faces
    vertices = mesh[0]
    faces = mesh[1]
    
    # Logging and checks
    print(f"Number of vertices extracted: {len(vertices)}")
    print(f"Number of faces extracted: {len(faces)}")
    end_time = time.time()    
    processing_time = end_time - start_time
    print(f"Time to process Poisson surface construction: {processing_time}")        
    # Display the mesh
    visualize_mesh(vertices, faces, depth_map)
    
    return vertices, faces

frame_number = 0

#Here's the adjusted version of the function that incorporates the depth map display when it's provided: (lifted straight from anthropic claude tyvm)
def visualize_mesh(vertices, faces, depth_map=None):
    global frame_number #for export to png
    """
    Visualize the mesh using matplotlib, and optionally display the depth map.
    :param vertices: Vertices of the mesh
    :param faces: Faces of the mesh
    :param depth_map: Optional depth map to display alongside the mesh
    """
    if depth_map is None:
        #fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        #fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(121, projection='3d')
    
    ax.view_init(elev=30, azim=0)
    # Plot the mesh
    mesh = Poly3DCollection(vertices[faces], alpha=0.3)
    ax.add_collection3d(mesh)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Environment Reconstruction')
    
    # Annotate number of vertices and faces
    num_vertices = len(vertices)
    num_faces = len(faces)
    ax.text2D(0.05, 0.95, f'Vertices: {num_vertices}', transform=ax.transAxes)
    ax.text2D(0.05, 0.90, f'Faces: {num_faces}', transform=ax.transAxes)
    
    if depth_map is not None:
        # Add depth map to the right
        ax_depth = fig.add_subplot(122)
        im = ax_depth.imshow(depth_map, cmap='plasma')
        ax_depth.set_title('Depth Map')
        plt.colorbar(im, ax=ax_depth, label='Depth')
    
    #plt.tight_layout()
    plt.show()
    # Create 'frames' directory if it doesn't exist
    os.makedirs('frames', exist_ok=True)
    
    # Save the figure as a PNG file
    filename = f'frames/frame{frame_number:04d}.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    frame_number += 1
    plt.pause(0.2)  # Pause for 2 seconds before closing
    
def process_depth_mapping(image_for_depth_mapping, palette_image=None, horizon_y=None, height=480):
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
    #horizon_y = find_horizon_line(depth_map)
    
    # Convert depth map to 3D points
    #points = depth_map_to_3d(depth_map)
    
    # Correct perspective distortion in 3D points

    
    # Perform surface reconstruction using Poisson surface reconstruction
    vertices, faces = perform_poisson_surface_reconstruction(depth_map)
    
    # Visualize the reconstructed mesh
    #visualize_mesh(vertices, faces)



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
