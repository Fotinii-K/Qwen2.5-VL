# import torch
# print(torch.__version__)

# import torch
# print(torch.cuda.is_available())

# import torch

# if torch.cuda.is_available():
#     print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
# else:
#     print("CUDA is not available.")

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from vid_processing2 import preprocess_video
from spatial_understanding import decode_xml_points, plot_bounding_boxes, plot_points, parse_json, inference
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor

# Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# Load the processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

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

# Function to extract bounding box information from the response
def extract_bounding_box(response):
    try:
        # Attempt to parse JSON from the response
        bbox_data = parse_json(response)
        if isinstance(bbox_data, dict) and "bbox" in bbox_data:
            return bbox_data["bbox"]
        return None
    except Exception as e:
        print(f"Error parsing bounding box data: {e}")
        return None

# Function to handle spatial understanding tasks
def handle_spatial_understanding(image_path, response):
    try:
        bbox = extract_bounding_box(response)
        if bbox:
            image = Image.open(image_path)
            image.thumbnail([640, 640], Image.Resampling.LANCZOS)
            plot_bounding_boxes(image, bbox, image.width, image.height)
            image.show()  # Optionally display the image with bounding boxes
    except Exception as e:
        print(f"Error handling spatial understanding: {e}")

# Example file paths
file_paths = ["/home/nakama6000/Documents/git/Qwen2.5-VL/prfmc/demo.png"]

# Determine input type
input_type = get_input_type(file_paths)

# Define General Context Message
message_general_context = """You are working as the 'decision-making core' of a single armed household robot, capable of working on 
                            different tasks. You will receive a series of frames or a collage of sequence of images in chronological order, illustrating the 
                            progression of a task currently executed by the robot, with the last image representing your current observation. 
                            Your purpose is to infer whether the task has been successfully achieved and address next steps in case of 
                            a failure, based on the defined goal. 
                            Do not make assumption of your own. """

# Construct messages based on input type
if input_type in ["image", "multi_image"]:
    messages = [
        {"role": "system", "content": message_general_context},  
        {
            "role": "user",
            "content": [
                {"type": "image", "image": path} for path in file_paths
            ] + [
                {"type": "text", "text": 
                 """What is enclosed by the red box"""
                }
            ],
        }
    ]
elif input_type == "video":
    messages = [
        {"role": "system", "content": message_general_context },  
        {
            "role": "user",
            "content": [
                {"type": "video", "video": file_paths},
                {"type": "text", "text": """Has the robot successfully placed ALL the fruits inside the basket?"""}  
            ],
        }
    ]

# Process vision inputs
image_inputs, video_inputs = process_vision_info([messages])

# Preprocess video if the input type is video
if input_type == "video":
    video_inputs = preprocess_video(file_paths, interval_seconds=0.2, max_frames=1000)

# Prepare inputs for the model
inputs = processor(
    text=[processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    do_rescale=False if input_type == "video" else True,  # Disable rescaling for videos
)

# Move inputs to CUDA
inputs = inputs.to("cuda")

# Generate output
generated_ids = model.generate(**inputs, max_new_tokens=10000)

# Decode the generated output
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# Extract only the assistant's response
def extract_assistant_response(output_text):
    assistant_start = output_text[0].find("assistant\n") + len("assistant\n")
    assistant_response = output_text[0][assistant_start:].strip()
    return assistant_response

# Print the response
assistant_response = extract_assistant_response(output_text)
print(f"Response for {input_type}:")
print(assistant_response)





# Handle spatial understanding if bounding boxes are mentioned
if input_type in ["image", "multi_image"]:
    handle_spatial_understanding(file_paths[-1], assistant_response)