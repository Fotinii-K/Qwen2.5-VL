import os

# Function to determine input type based on file extension
def get_input_type(file_paths):
    if isinstance(file_paths, list):  # Check if it's a list of image paths
        exts = [os.path.splitext(path.lower())[1] for path in file_paths]
        if all(ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif'] for ext in exts):
            return "image" if len(file_paths) == 1 else "multi_image"
        else:
            print(f"Unsupported file types: {exts}. Please provide valid image files.")
            exit()
    elif isinstance(file_paths, str):  # Check if it's a single video path
        _, ext = os.path.splitext(file_paths.lower())
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            return "video"
        else:
            print(f"Unsupported file type: {ext}. Please provide a valid video file.")
            exit()
    else:
        print("Invalid input format. Please provide either a list of image paths or a single video path.")
        exit()
