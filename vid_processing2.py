import cv2
import numpy as np

def preprocess_video(video_path, interval_seconds=0.1, max_frames=1000, target_size=(224, 224), create_collage=True):
    """
    Preprocess a video by extracting frames at specified time intervals.

    Args:
        video_path (str): Path to the input video file.
        interval_seconds (float): Extract one frame every `interval_seconds` seconds.
        max_frames (int): Maximum number of frames to extract.
        target_size (tuple): Target size for resizing frames (width, height).
        create_collage (bool): Whether to create and save a collage of the extracted frames.

    Returns:
        np.ndarray: Array of normalized frames.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
    frame_interval = int(fps * interval_seconds)  # Calculate frame interval based on FPS
    frame_count = 0  # Counter for extracted frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Check if the current frame corresponds to the desired interval
        if current_frame_idx % frame_interval == 0 and len(frames) < max_frames:
            frame = cv2.resize(frame, target_size)  # Resize frame
            frames.append(frame)
            frame_count += 1
        
        if frame_count >= max_frames:
            break  # Stop if we've reached the maximum number of frames
    
    cap.release()
    
    # Normalize all frames after the loop
    frames = np.array(frames).astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Create a collage of the frames
    if create_collage and len(frames) > 0:
        collage = create_frame_collage(frames, target_size)
        collage_path = "/home/nakama6000/Documents/git/Qwen2.5-VL/frame_collage.jpg"
        cv2.imwrite(collage_path, (collage * 255).astype(np.uint8))  # Save as an image file
        print(f"Collage saved to {collage_path}")
    
    return frames


def create_frame_collage(frames, target_size):
    """
    Create a grid-based collage from a list of frames.

    Args:
        frames (list): List of frames (numpy arrays).
        target_size (tuple): Size of each frame in the collage.

    Returns:
        np.ndarray: Collage as a single numpy array.
    """
    num_frames = len(frames)
    rows = int(np.ceil(np.sqrt(num_frames)))  # Number of rows
    cols = int(np.ceil(num_frames / rows))    # Number of columns
    
    # Create an empty canvas for the collage
    collage_height = rows * target_size[1]
    collage_width = cols * target_size[0]
    collage = np.zeros((collage_height, collage_width, 3), dtype=np.float32)
    
    # Place each frame in the grid
    for i, frame in enumerate(frames):
        row = i // cols
        col = i % cols
        y_start = row * target_size[1]
        y_end = y_start + target_size[1]
        x_start = col * target_size[0]
        x_end = x_start + target_size[0]
        collage[y_start:y_end, x_start:x_end] = frame
    
    return collage