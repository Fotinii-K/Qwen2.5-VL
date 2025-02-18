import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from spatial_understanding import decode_xml_points, plot_bounding_boxes, plot_points, parse_json, inference
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import time

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

t_load_start = time.time()

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)
t_load_end = time.time()

print(f"time taken to load = {t_load_end-t_load_start}, Associated FPS = {1/(t_load_end-t_load_start)}")

# #### 1. Detect certain object in the image
# Let's start with a simple scenario where we want to locate certain objects in an image. 
# Besides, we can further prompt the model to describe their unique characteristics or features by explicitly giving that order.

image_path = "/home/nakama6000/Documents/git/Qwen2.5-VL/cookbooks/assets/spatial_understanding/cakes.png"

## Use a local HuggingFace model to inference.
# prompt in english
prompt = "Outline the position of each small cake and output all the coordinates in JSON format."

t_inference_start = time.time()
response, input_height, input_width = inference(image_path, prompt)
t_inference_end = time.time()

print(f"time taken to load = {t_inference_end-t_inference_start}, Associated FPS = {1/(t_inference_end-t_inference_start)}")


image = Image.open(image_path)
print(image.size)
image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_bounding_boxes(image,response,input_width,input_height)





# # #### 2. Detect a specific object using descriptions
# # Further, you can search for a specific object by using a short phrase or sentence to describe it.

# ! hello

#? 
# * 
image_path = "./assets/spatial_understanding/cakes.png"

# # prompt in chinses
# prompt = "定位最右上角的棕色蛋糕，以JSON格式输出其bbox坐标"
# prompt in english
prompt = "Locate the top right brown cake, output its bbox coordinates using JSON format."

# ## Use a local HuggingFace model to inference.
# response, input_height, input_width = inference(image_path, prompt)

# image = Image.open(image_path)
# image.thumbnail([640,640], Image.Resampling.LANCZOS)
# plot_bounding_boxes(image,response,input_width,input_height)


# image_path = "./assets/spatial_understanding/cakes.png"

# # prompt in chinese
# prompt = "以点的形式定位图中桌子远处的擀面杖，以XML格式输出其坐标"
# # prompt in english
# prompt = "point to the rolling pin on the far side of the table, output its coordinates in XML format <points x y>object</points>"

# ## Use a local HuggingFace model to inference.
# response, input_height, input_width = inference(image_path, prompt)

# image = Image.open(image_path)
# image.thumbnail([640,640], Image.Resampling.LANCZOS)
# plot_points(image, response, input_width, input_height)


# image_path = "/home/nakama6000/Documents/git/Qwen2.5-VL/cookbooks/assets/spatial_understanding/Origamis.jpg"

# # prompt in chinese
# prompt = "框出图中纸狐狸的影子，以json格式输出其bbox坐标"
# # prompt in english
# prompt = "Locate the shadow of the paper fox, report the bbox coordinates in JSON format."

# ## Use a local HuggingFace model to inference.
# response, input_height, input_width = inference(image_path, prompt)

# image = Image.open(image_path)
# image.thumbnail([640,640], Image.Resampling.LANCZOS)
# plot_bounding_boxes(image, response, input_width, input_height)

# # #### 5. Understand relationships across different instances

# image_path = "./assets/spatial_understanding/cartoon_brave_person.jpeg"

# # prompt in chinese
# prompt = "框出图中见义勇为的人，以json格式输出其bbox坐标"
# # prompt in english
# prompt = "Locate the person who act bravely, report the bbox coordinates in JSON format."

# ## Use a local HuggingFace model to inference.
# response, input_height, input_width = inference(image_path, prompt)

# image = Image.open(image_path)
# image.thumbnail([640,640], Image.Resampling.LANCZOS)
# plot_bounding_boxes(image, response, input_width, input_height)


# # #### 6. Find a special instance with unique characteristic (color, location, utility, ...)

# url = "./assets/spatial_understanding/multiple_items.png"

# # prompt in chinese
# prompt = "如果太阳很刺眼，我应该用这张图中的什么物品，框出该物品在图中的bbox坐标，并以json格式输出"
# # prompt in english
# prompt = "If the sun is very glaring, which item in this image should I use? Please locate it in the image with its bbox coordinates and its name and output in JSON format."

# ## Use a local HuggingFace model to inference.
# response, input_height, input_width = inference(url, prompt)

# image = Image.open(url)
# image.thumbnail([640,640], Image.Resampling.LANCZOS)
# plot_bounding_boxes(image, response, input_width, input_height)


# # #### 7. Use Qwen2.5-VL grounding capabilities to help counting

# image_path = "./assets/spatial_understanding/multiple_items.png"

# # prompt in chinese
# prompt = "请以JSON格式输出图中所有物体bbox的坐标以及它们的名字，然后基于检测结果回答以下问题：图中物体的数目是多少？"
# # prompt in english
# prompt = "Please first output bbox coordinates and names of every item in this image in JSON format, and then answer how many items are there in the image."

# ## Use a local HuggingFace model to inference.
# response, input_height, input_width = inference(image_path, prompt)

# image = Image.open(image_path)
# image.thumbnail([640,640], Image.Resampling.LANCZOS)
# plot_bounding_boxes(image,response,input_width,input_height)

# # #### 8. spatial understanding with designed system prompt
# # The above usage is based on the default system prompt. You can also change the system prompt to obtain other output format like plain text.
# # Qwen2.5-VL now support these formats:
# # * bbox-format: JSON
# # 
# # `{"bbox_2d": [x1, y1, x2, y2], "label": "object name/description"}`
# # 
# # * bbox-format: plain text
# # 
# # `x1,y1,x2,y2 object_name/description`
# # 
# # * point-format: XML
# # 
# # `<points x y>object_name/description</points>`
# # 
# # * point-format: JSON
# # 
# # `{"point_2d": [x, y], "label": "object name/description"}`

# # Change your system prompt to use plain text as output format

# # In[56]:


# image_path = "./assets/spatial_understanding/cakes.png"
# image = Image.open(image_path)
# system_prompt = "As an AI assistant, you specialize in accurate image object detection, delivering coordinates in plain text format 'x1,y1,x2,y2 object'."
# prompt = "find all cakes"
# response, input_height, input_width = inference(image_path, prompt, system_prompt=system_prompt)





# ###############################


# # NVIDIA CUDA Support
# # Requirements:

# # CUDA 12.0 and above.
# # We recommend the Pytorch container from Nvidia, which has all the required tools to install FlashAttention.

# # FlashAttention-2 with CUDA currently supports:

# # Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100). Support for Turing GPUs (T4, RTX 2080) is coming soon, please use FlashAttention 1.x for Turing GPUs for now.
# # Datatype fp16 and bf16 (bf16 requires Ampere, Ada, or Hopper GPUs).
# # All head dimensions up to 256. Head dim > 192 backward requires A100/A800 or H100/H800. Head dim 256 backward now works on consumer GPUs (if there's no dropout) as of flash-attn 2.5.5.



# # CUDA toolkit or ROCm toolkit
# # PyTorch 2.2 and above. ---> done :2.2.0+cu118     OK
# # packaging Python package (pip install packaging) --> OK
# # ninja Python package (pip install ninja) * --> OK
# # Linux. Might work for Windows starting v2.3.2 (we've seen a few positive reports) but Windows compilation still requires more testing. If you have ideas on how to set up prebuilt CUDA wheels for Windows, please reach out via Github issue.
# # * Make sure that ninja is installed and that it works correctly (e.g. ninja --version then echo $? should return exit code 0). If not (sometimes ninja --version then echo $? returns a nonzero exit code), uninstall then reinstall ninja (pip uninstall -y ninja && pip install ninja). Without ninja, compiling can take a very long time (2h) since it does not use multiple CPU cores. With ninja compiling takes 3-5 minutes on a 64-core machine using CUDA toolkit.



# #Currently Installed CUDA Versions
# # find /usr/local -type d -name "cuda*"