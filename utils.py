import torch
import os
import hashlib
import requests
import ast
import json
import shutil
import cv2
import markdown
import numpy as np

from torchvision import transforms

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from bs4 import BeautifulSoup
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor

import xml.etree.ElementTree as ET
from decord import VideoReader, cpu

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

####### From spatial_understanding.py  #######

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def decode_xml_points(text):
    try:
        root = ET.fromstring(text)
        num_points = (len(root.attrib) - 1) // 2
        points = []
        for i in range(num_points):
            x = root.attrib.get(f'x{i+1}')
            y = root.attrib.get(f'y{i+1}')
            points.append([x, y])
        alt = root.attrib.get('alt')
        phrase = root.text.strip() if root.text else None
        return {
            "points": points,
            "alt": alt,
            "phrase": phrase
        }
    except Exception as e:
        print(e)
        return None

def plot_bounding_boxes(im, bounding_boxes, input_width, input_height):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    # print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json_su(bounding_boxes)

    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    try:
      json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
      end_idx = bounding_boxes.rfind('"}') + len('"}')
      truncated_text = bounding_boxes[:end_idx] + "]"
      json_output = ast.literal_eval(truncated_text)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
      # Select a color from the list
      color = colors[i % len(colors)]

      # Convert normalized coordinates to absolute coordinates
      abs_y1 = int(bounding_box["bbox_2d"][1]/input_height * height)
      abs_x1 = int(bounding_box["bbox_2d"][0]/input_width * width)
      abs_y2 = int(bounding_box["bbox_2d"][3]/input_height * height)
      abs_x2 = int(bounding_box["bbox_2d"][2]/input_width * width)

      if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

      if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

      # Draw the bounding box
      draw.rectangle(
          ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
      )

      # Draw the text
      if "label" in bounding_box:
        draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    # Display the image
    img.show()


def plot_points(im, text, input_width, input_height):
  img = im
  width, height = img.size
  draw = ImageDraw.Draw(img)
  colors = [
    'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
    'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
    'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
  ] + additional_colors
  xml_text = text.replace('```xml', '')
  xml_text = xml_text.replace('```', '')
  data = decode_xml_points(xml_text)
  if data is None:
    img.show()
    return
  points = data['points']
  description = data['phrase']

  font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

  for i, point in enumerate(points):
    color = colors[i % len(colors)]
    abs_x1 = int(point[0])/input_width * width
    abs_y1 = int(point[1])/input_height * height
    radius = 2
    draw.ellipse([(abs_x1 - radius, abs_y1 - radius), (abs_x1 + radius, abs_y1 + radius)], fill=color)
    draw.text((abs_x1 + 8, abs_y1 + 6), description, fill=color, font=font)
  
  img.show()

  
def parse_json_su(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])            # Remove everything before "```json"
            json_output = json_output.split("```")[0]       # Remove everything after the closing "```"
            break                                           # Exit the loop once "```json" is found
    return json_output


def inference_su(image, prompt, system_prompt="You are a helpful assistant", max_new_tokens=1024):

  messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"image": image}
      ]
    }
  ]
  
  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  print("input:\n",text)
  inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to('cuda')

  output_ids = model.generate(**inputs, max_new_tokens=1024)
  generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
  output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
  print("output:\n",output_text[0])

  input_height = inputs['image_grid_thw'][0][1]*14
  input_width = inputs['image_grid_thw'][0][2]*14

  return output_text[0], input_height, input_width


def inference_su_video_bbx(image, messages, max_new_tokens=1024):

  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#   print("input:\n",text)
  inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to('cuda')

  output_ids = model.generate(**inputs, max_new_tokens=1024)
  generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
  output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#   print("output:\n",output_text[0])

  input_height = inputs['image_grid_thw'][0][1]*14
  input_width = inputs['image_grid_thw'][0][2]*14

  return output_text[0], input_height, input_width



#######  From video_understanding.py  ########

def download_video(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")


def get_video_frames(video_path, num_frames=128, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
    if video_path.startswith('http://') or video_path.startswith('https://'):
        video_file_path = os.path.join(cache_dir, f'{video_hash}.mp4')
        if not os.path.exists(video_file_path):
            download_video(video_path, video_file_path)
    else:
        video_file_path = video_path

    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npy')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npy')

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return video_file_path, frames, timestamps

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)
    
    return video_file_path, frames, timestamps


def create_image_grid(images, num_columns=8):
    pil_images = [Image.fromarray(image) for image in images]
    num_rows = (len(images) + num_columns - 1) // num_columns

    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    return grid_image


def inference_vu(video_path, system_prompt, prompt, max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type":"video", "video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]


def parse_json_vu(response):
    html = markdown.markdown(response, extensions=['fenced_code'])
    soup = BeautifulSoup(html, 'html.parser')
    json_text = soup.find('code').text
    
    data = json.loads(json_text)

    soup.find('code').decompose()  # Remove the JSON code block from the soup
    answer = soup.get_text(separator='\n', strip=True)

    return data, answer


from datetime import datetime

def time_to_seconds(time_value):
    """
    Converts a timestamp into seconds.
    Handles both string formats ('HH:MM:SS.ff') and numeric values (seconds as floats).
    """
    if isinstance(time_value, str):  # Handle string timestamps
        try:
            # Try parsing with hours included
            time_obj = datetime.strptime(time_value, '%H:%M:%S.%f')
        except ValueError:
            try:
                # Try parsing without hours
                time_obj = datetime.strptime(time_value, '%M:%S.%f')
            except ValueError:
                raise ValueError(f"Invalid timestamp format: {time_value}")

        total_seconds = (
            time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6 )
        return total_seconds

    elif isinstance(time_value, (int, float)):  # Handle numeric timestamps
        return float(time_value)

    else:
        raise TypeError(f"Unsupported timestamp type: {type(time_value)}")



############  Additional functions  ################

def transform_video_inputs_to_frames(video_inputs, output_dir):
    """
    Transform video inputs into a list of frame file paths.
    
    Args:
        video_inputs (list): List of video inputs, where each input is either:
            - A torch.Tensor of shape [N, C, H, W], or
            - A list of PIL.Image objects.
        output_dir (str): Directory to save the extracted frames.
    
    Returns:
        list: List of file paths to the saved frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_paths, frame_idx = [], 0
    
    # Define a transformation to convert tensors to PIL images
    to_pil = transforms.ToPILImage()
    
    for frames in video_inputs:
        if isinstance(frames, torch.Tensor):  # Video as tensor [N, C, H, W]
            for frame_tensor in frames:
                # Ensure the tensor is in the range [0, 1] and has dtype float32
                if frame_tensor.max() > 1:
                    frame_tensor = frame_tensor / 255.0  # Scale values to [0, 1]

                frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
                to_pil(frame_tensor).save(frame_path)  # Convert to PIL and save
                frame_paths.append(os.path.abspath(frame_path))
                frame_idx += 1
        elif isinstance(frames, list) and all(isinstance(f, Image.Image) for f in frames):  # Video as list of PIL images
            for frame_image in frames:

                frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
                frame_image.save(frame_path)
                frame_paths.append(os.path.abspath(frame_path))
                frame_idx += 1
        else:
            raise ValueError("Unsupported video input format.")
    
    return frame_paths


def timestamp_to_frame_index(timestamp_str, fps):
    minutes, seconds = map(float, timestamp_str.split(":"))
    total_seconds = minutes * 60 + seconds
    return int(total_seconds * fps)


def clear_folders(folders=["__pycache__", ".cache"], verbose=True):
    """
    Deletes specified folders if they exist.
    Args:
        folders (list): List of folder names or paths to clear.
        verbose (bool): If True, prints status messages.
    """
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)