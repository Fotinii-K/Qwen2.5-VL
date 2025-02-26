import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from vid_processing2 import preprocess_video
from spatial_understanding import decode_xml_points, plot_bounding_boxes, plot_points, parse_json, inference
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import ast
import json

# Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto")
model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")

# Load the processor
processor = AutoProcessor.from_pretrained(model_path)

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

# Function to process bounding box requests
# def bounding_box_request(image_path, prompt, system_prompt):
#     image = Image.open(image_path)
#     response, input_height, input_width = inference(image, prompt, system_prompt)  # Get bounding box coordinates
#     image = Image.open(image_path)
#     image.thumbnail([640, 640], Image.Resampling.LANCZOS)  # Resize image for visualization
#     plot_bounding_boxes(image, response, input_width, input_height)  # Plot bounding boxes
#     # return response  # Return the JSON response with coordinates


def bounding_box_request(image_path, prompt, system_prompt):
    # Load the image
    image = Image.open(image_path)

    # Perform inference to get the response and input dimensions
    response, input_height, input_width = inference(image, prompt, system_prompt)

    try:
        # Clean the response using the parse_json utility function
        cleaned_response = parse_json(response)

        # Validate the cleaned JSON
        json_data = json.loads(cleaned_response)
        is_valid_json = True
    except ValueError:
        return None

    if is_valid_json:
        # Resize the image for visualization
        image.thumbnail([640, 640], Image.Resampling.LANCZOS)

        # Plot bounding boxes on the image
        plot_bounding_boxes(image, cleaned_response, input_width, input_height)

        # Return the JSON response with coordinates
        return cleaned_response
    else:
        return None  # No bounding boxes to return
    


# Extract only the assistant's response
def extract_assistant_response(output_text):
    assistant_start = output_text[0].find("assistant\n") + len("assistant\n")
    assistant_response = output_text[0][assistant_start:].strip()
    return assistant_response


# Example file paths
file_paths = ["/home/nakama6000/Documents/git/Qwen2.5-VL/demo.jpeg"]

# Determine input type
input_type = get_input_type(file_paths)

# Define General Context Message
message_general_context = """You are a helpful assistant. Describe everything you can see in the picture."""
prompt = """If any animals are identified in the scene create a bounding box around them and output all the coordinates in JSON format."""

# Construct messages based on input type
if input_type in ["image", "multi_image"]:
    for image_path in file_paths:
        if "bounding box" in prompt.lower() or "outline" in prompt.lower():
            # Process bounding box request
            bbox_response = bounding_box_request(image_path, prompt, system_prompt = message_general_context)
           
            
            if bbox_response is None:
                # No bounding boxes detected, continue to the next image
                continue
            # else:
            #     # Bounding boxes were detected, process the response
            #     print("Bounding Box Coordinates (JSON):")
            #     print(bbox_response)
        
# Define messages for non-bounding-box cases
            messages = [
                {"role": "system", "content": message_general_context},  
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": path} for path in file_paths
                    ] + [
                        {"type": "text", "text": prompt
                        #  """If any animals are identified in the scene create a bounding box around them."""
                        }
                    ],
                }
            ]
        # Process vision inputs
            image_inputs, _ = process_vision_info([messages])

            inputs = processor(
                text=[processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
                images=[Image.open(image_path)],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            generated_ids = model.generate(**inputs, max_new_tokens=10000)
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print(f"Response for image: {image_path}")
            print(output_text[0])


        # Print the response
        assistant_response = extract_assistant_response(output_text)
        print(f"Response for {input_type}:")
        print(assistant_response)


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
    _, video_inputs = process_vision_info([messages])

    video_inputs = preprocess_video(file_paths, interval_seconds=0.2, max_frames=1000)
    
    inputs = processor(
        text=[processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        do_rescale=False 
    )
    # Move inputs to CUDA
    inputs = inputs.to("cuda")

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=10000)

    # Decode the generated output
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Print the response
    assistant_response = extract_assistant_response(output_text)
    print(f"Response for {input_type}:")
    print(assistant_response)
