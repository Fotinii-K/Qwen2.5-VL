# import torch
# print(torch.__version__)

# import torch
# print(torch.cuda.is_available())

# import torch

# if torch.cuda.is_available():
#     print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
# else:
#     print("CUDA is not available.")

###### !
# import torch
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from spatial_understanding import decode_xml_points, plot_bounding_boxes, plot_points, parse_json, inference
# from PIL import Image, ImageDraw, ImageFont
# from PIL import ImageColor
# import time

# model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
# processor = AutoProcessor.from_pretrained(model_path)


# image_path = "/home/nakama6000/Documents/git/Qwen2.5-VL/cookbooks/assets/spatial_understanding/cakes.png"
# prompt = "Outline the position of each small cake and output all the coordinates in JSON format."
# response, input_height, input_width = inference(image_path, prompt)

# image = Image.open(image_path)
# print(image.size)
# image.thumbnail([640,640], Image.Resampling.LANCZOS)
# plot_bounding_boxes(image,response,input_width,input_height)

########## !

# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
# import torch
# import os
# from vid_processing2 import preprocess_video
# # from spatial_understanding import decode_xml_points, plot_bounding_boxes, plot_points, parse_json, inference
# from PIL import Image, ImageDraw, ImageFont
# from PIL import ImageColor
# from pyset_wrapper.pyzed_wrapper_v2 import *
# import numpy as np

# # Load the model on the available device(s)
# # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
# #     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# # )
# model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
# processor = AutoProcessor.from_pretrained(model_path)

# # Load the processor
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# # Function to determine input type based on file extension
# def get_input_type(file_paths):
#     if isinstance(file_paths, list):  # Check if it's a list of image paths
#         exts = [os.path.splitext(path.lower())[1] for path in file_paths]
#         if all(ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif'] for ext in exts):
#             return "image" if len(file_paths) == 1 else "multi_image"
#         else:
#             print(f"Unsupported file types: {exts}. Please provide valid image files.")
#             exit()
#     elif isinstance(file_paths, str):  # Check if it's a single video path
#         _, ext = os.path.splitext(file_paths.lower())
#         if ext in ['.mp4', '.avi', '.mov', '.mkv']:
#             return "video"
#         else:
#             print(f"Unsupported file type: {ext}. Please provide a valid video file.")
#             exit()
#     else:
#         print("Invalid input format. Please provide either a list of image paths or a single video path.")
#         exit()

# # Example file paths
# file_paths = ["/home/nakama6000/Documents/git/Qwen2.5-VL/demo.jpeg"]

# # Determine input type
# input_type = get_input_type(file_paths)

# # Define General Context Message
# message_general_context = """You are working a helpful assistant. """
# prompt = """If any animals are identified in the scene create a bounding box around them."""

# # Construct messages based on input type
# if input_type in ["image", "multi_image"]:
#     messages = [
#         {"role": "system", "content": message_general_context},  
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": path} for path in file_paths
#             ] + [
#                 {"type": "text", "text": prompt
#                 #  """If any animals are identified in the scene create a bounding box around them."""
#                 }
#             ],
#         }
#     ]
# elif input_type == "video":
#     messages = [
#         {"role": "system", "content": message_general_context },  
#         {
#             "role": "user",
#             "content": [
#                 {"type": "video", "video": file_paths},
#                 {"type": "text", "text": """Has the robot successfully placed ALL the fruits inside the basket?"""}  
#             ],
#         }
#     ]

# # Process vision inputs
# image_inputs, video_inputs = process_vision_info([messages])

# # Preprocess video if the input type is video
# if input_type == "video":
#     video_inputs = preprocess_video(file_paths, interval_seconds=0.2, max_frames=1000)

# # Prepare inputs for the model
# inputs = processor(
#     text=[processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
#     do_rescale=False if input_type == "video" else True,  # Disable rescaling for videos
# )

# # Move inputs to CUDA
# inputs = inputs.to("cuda")

# # Generate output
# generated_ids = model.generate(**inputs, max_new_tokens=10000)

# # Decode the generated output
# output_text = processor.batch_decode(
#     generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )

# # Extract only the assistant's response
# def extract_assistant_response(output_text):
#     assistant_start = output_text[0].find("assistant\n") + len("assistant\n")
#     assistant_response = output_text[0][assistant_start:].strip()
#     return assistant_response

# # Print the response
# assistant_response = extract_assistant_response(output_text)
# print(f"Response for {input_type}:")
# print(assistant_response)


