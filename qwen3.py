from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import numpy as np
from spatial_understanding import decode_xml_points, plot_bounding_boxes, plot_points, parse_json, inference
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
from pyset_wrapper.pyzed_wrapper_v2 import *
from omegaconf import OmegaConf


conf = OmegaConf.load('pyset_wrapper/sam2_zed_small.yaml')
pw = Wrapper(conf)

try:
    pw.open_input_source()
except Exception as e:
    print(f" exited with error: {e}")

index = 0
frame_array = []
pw.start_stream()
while True:
    # try
    if pw.retrieve(is_image=True, is_measure=True):
            # Extract from ZED camera
        if index<50:
            left_image = pw.output_image
            frame_array.append(left_image)
            index+=1
        else: break

            # depth_map = pw.output_measure
    # except Exception as e:
    #     print(f" exited with error: {e}")

    # finally:
    #     pw.stop_stream()
    #     pw.close_input_source()


# Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    # device_map="auto",
)

# Move the model to CUDA if available
if torch.cuda.is_available():
    model = model.to("cuda")

# Load the processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

last_frame = frame_array[index-1]

# Function to process bounding box requests
def bounding_box_request(last_frame, prompt):
    response, input_height, input_width = inference(last_frame, prompt)  # Get bounding box coordinates
    # image = Image.open(last_frame)
    last_frame.thumbnail([640, 640], Image.Resampling.LANCZOS)  # Resize image for visualization
    plot_bounding_boxes(last_frame, response, input_width, input_height)  # Plot bounding boxes
    # return response  # Return the JSON response with coordinates

# Define General Context Message 
# message_general_context = """You are working as the 'decision-making core' of a single armed household robot, capable of working on 
                            # different tasks. You will receive a series of frames or a collage of sequence of images in chronological order, illustrating the 
                            # progression of a task currently executed by the robot, with the last image representing your current observation. 
                            # Your purpose is to infer whether the task has been successfully achieved and address next steps in case of 
                            # a failure, based on the defined goal. 
                            # Do not make assumption of your own. """
message_general_context = """You are working a helpful assistant. """
prompt = """If any humans are identified in the scene create a bounding box around them and output all the coordinates in JSON format."""

for frame in frame_array:
    if "bounding box" in prompt.lower() or "outline" in prompt.lower():
        # Process bounding box request
        print(f"Processing bounding box request for image: {frame}")
        frame = np.array(frame).astype(np.uint8)
        bbox_response = bounding_box_request(frame, prompt)
        print("Bounding Box Coordinates (JSON):")
        print(bbox_response)
    else:
        messages = [
            {"role": "system", "content": message_general_context},  
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": path} for path in frame_array
                ] + [
                    {"type": "text", "text": prompt
                    }  
                ],
            }
        ]
        inputs = processor(
            text=[processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
            images=[Image.open(frame)],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=10000)
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"Response for image: {frame}")
        print(output_text[0])


# Process vision inputs
image_inputs, video_inputs = process_vision_info([messages])

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
# print(f"Response for {input_type}:")
print(assistant_response)