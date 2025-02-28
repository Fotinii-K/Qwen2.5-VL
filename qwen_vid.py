import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from video_understanding import download_video, get_video_frames, create_image_grid, inference
from IPython.display import Markdown, display

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)


local_video_path = "/home/nakama6000/Documents/git/Qwen2.5-VL/fruits_in_basket.mp4"

# Ensure the prompt remains the same
prompt = "Could you go into detail about the content of this long video?"

# Use the get_video_frames function with the local video path
video_path, frames, timestamps = get_video_frames(local_video_path, num_frames=64)

# Optionally, create and display an image grid of the video frames
# image_grid = create_image_grid(frames, num_columns=8)
# display(image_grid.resize((640, 640)))

# Run the inference function with the local video
response = inference(video_path, prompt)

# Display the response
display(Markdown(response))