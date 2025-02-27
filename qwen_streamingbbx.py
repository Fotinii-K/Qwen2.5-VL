import torch
import numpy as np
import cv2
import json
import time

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from spatial_understanding import decode_xml_points, plot_bounding_boxes, plot_points, parse_json, inference

from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
from pyset_wrapper.pyzed_wrapper_v2 import *
from omegaconf import OmegaConf

# Load configuration and initialize ZED camera
conf = OmegaConf.load('pyset_wrapper/sam2_zed_small.yaml')
pw = Wrapper(conf)
try:
    pw.open_input_source()
except Exception as e:
    print(f"Failed to initialize ZED camera: {e}")
    exit(1)

# Start streaming
pw.start_stream()

# Initialize variables
frame_array = []
bounding_box_history = []  # List to store bounding box history
index = 0
max_frames_to_capture = 10

# Capture frames
while index < max_frames_to_capture:
    if pw.retrieve(is_image=True, is_measure=True):
        left_image = pw.output_image
        if left_image is not None:  # Ensure the frame is valid
            frame_array.append(left_image)
            index += 1
    time.sleep(0.5)

# Stop streaming and close camera
pw.stop_stream()
pw.close_input_source()

# Adjust color balance for ZED frames
def adjust_color_balance(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

frame_array = [adjust_color_balance(np.array(frame)) for frame in frame_array]

# Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", device_map="auto",
)

if torch.cuda.is_available():
    model = model.to("cuda")

# Load the processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Function to process bounding box requests
# def bounding_box_request(last_frame, prompt, system_prompt):
#     response, input_height, input_width = inference(last_frame, prompt, system_prompt)
#     try:
#         cleaned_response = parse_json(response)
#         json_data = json.loads(cleaned_response)
#         is_valid_json = True
#     except ValueError:
#         return None, []

#     if is_valid_json:
#         last_frame.thumbnail([640, 640], Image.Resampling.LANCZOS)
#         plot_bounding_boxes(last_frame, cleaned_response, input_width, input_height)
#         return cleaned_response, json_data.get("boxes", [])
#     else:
#         return None, []

def bounding_box_request(last_frame, prompt, system_prompt):
    response, input_height, input_width = inference(last_frame, prompt, system_prompt)
    
    try:
        # Parse the JSON response
        cleaned_response = parse_json(response)
        json_data = json.loads(cleaned_response)
        is_valid_json = True
    except ValueError:
        print("Invalid JSON response.")
        return None, []

    if is_valid_json:
        # Handle different JSON structures
        if isinstance(json_data, dict):  # Check if json_data is a dictionary
            boxes = json_data.get("boxes", [])
        elif isinstance(json_data, list):  # Check if json_data is a list of objects
            boxes = [item.get("coordinates") for item in json_data if "coordinates" in item]
        else:
            print("Unexpected JSON structure.")
            boxes = []

        # Resize the frame for thumbnail display
        last_frame.thumbnail([640, 640], Image.Resampling.LANCZOS)

        # Plot bounding boxes on the frame
        plot_bounding_boxes(last_frame, cleaned_response, input_width, input_height)

        return cleaned_response, boxes
    else:
        return None, []

# Define General Context Message
message_general_context = """You are a helpful assistant."""
prompt = """If any humans are identified in the scene, create a bounding box around them and output all the coordinates in JSON format."""

# Process each frame and maintain bounding box history
for frame_index, frame in enumerate(frame_array):
    frame = Image.fromarray(np.array(frame).astype(np.uint8))
    
    if "bounding box" in prompt.lower() or "outline" in prompt.lower():
        bbox_response, current_boxes = bounding_box_request(frame, prompt, message_general_context)
        
        if bbox_response is not None:
            # Add current bounding boxes to history with frame index
            for box in current_boxes:
                box["frame_index"] = frame_index
            bounding_box_history.extend(current_boxes)
            
            # Plot all bounding boxes from history on the current frame
            # response, input_height, input_width = inference(frame, prompt, system_prompt)
            _, input_height, input_width = inference(frame, prompt, message_general_context)  # Reuse inference to get dimensions
            plot_bounding_boxes(frame, bbox_response, input_width, input_height)
            
            # Save or display the annotated frame
            frame.save(f"annotated_frame_{frame_index}.png")
            print(f"Processed frame {frame_index} with bounding boxes.")
    else:
        messages = [
            {"role": "system", "content": message_general_context},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame}
                ] + [
                    {"type": "text", "text": prompt}
                ],
            }
        ]
        
        inputs = processor(
            text=[processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
            images=[frame],
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=10000)
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"Response for frame {frame_index}: {output_text[0]}")

# Extract only the assistant's response
def extract_assistant_response(output_text):
    assistant_start = output_text[0].find("assistant\n") + len("assistant\n")
    assistant_response = output_text[0][assistant_start:].strip()
    return assistant_response

# Print the final response
print("Final Response:")
print(extract_assistant_response(output_text))
