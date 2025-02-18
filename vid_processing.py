import cv2
import numpy as np

def preprocess_video(video_path, max_frames=10, target_size=(224, 224), create_collage=True):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(max_frames, total_frames)  # Ensure max_frames doesn't exceed total_frames
    
    # Calculate evenly spaced frame indices
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Seek to the desired frame
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, target_size)  # Resize (keeping BGR format)
        frames.append(frame)
    
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