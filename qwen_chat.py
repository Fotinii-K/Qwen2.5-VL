from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
from vid_processing import preprocess_video

# Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    # device_map="auto",
)

# Move the model to CUDA if available
if torch.cuda.is_available():
    model = model.to("cuda")

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



file_paths = ["/home/nakama6000/Documents/git/Qwen2.5-VL/prfmc/demo5.png"]
# file_paths = ["/home/nakama6000/Documents/git/Qwen2.5-VL/prfmc/demo1.png","/home/nakama6000/Documents/git/Qwen2.5-VL/prfmc/demo2.png", "/home/nakama6000/Documents/git/Qwen2.5-VL/prfmc/demo3.png","/home/nakama6000/Documents/git/Qwen2.5-VL/prfmc/demo4.png", "/home/nakama6000/Documents/git/Qwen2.5-VL/prfmc/demo5.png"]  
# file_paths = "/home/nakama6000/Documents/git/Qwen2.5-VL/fruits_in_basket.mp4"  


input_type = get_input_type(file_paths)

if input_type in ["image", "multi_image"]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": path} for path in file_paths
            ] + [
                {"type": "text", "text":"""What properties does the object in the red box have, according to the following list of properties?
                                            List of object properties:
                                            breakable - Mark if the object is brittle, that is, it can be broken into smaller pieces by a human dropping it on the floor (e.g. wine bottle, room light cleaning)
                                            Tool - Is a [object] designed to clean things? (e.g. scrub brush)
                                            cookable - Can a [object] be cooked? (e.g. biscuit, pizza) 
                                            grabbable - If an object has this attribute, it is usually lightweight and can be potentially grabbed and picked up by the robot. (e.g. apple, bottle, rag, plate)
                                            openable - Mark if the object is designed to be opened. (e.g. mixer, keg) 
                                            sliceable - Can a [object] be sliced easily by a human with a knife? (e.g. sweet corn, sandwich)
                                            slicingTool - Can a [object] slice an apple? (e.g. blade, razor) 
                                            toggleable - The object can be switched between a finite number of discrete states and is designed to do so. (e.g. hot tub, light bulb) 
                                            waterSource - where you can get water (e.g. sink)"""}
            ],
        }
    ]
elif input_type == "video":
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": file_paths},
                {
                    "type": "text",
                    "text": "You are working as the 'decision-making core' of a single armed household robot, capable of working on different tasks. You will receive a collage of sequence of images in chronological order, illustrating the progression of a task currently executed by the robot, with the last image representing your current observation. Your purpose is to infer whether the task has been successfully achieved and address next steps in case of a failure, based on the defined goal. Do not make assumption of your own. In this scenario the robot's task is to place all the fruits inside the basket.",
                },
            ],
        }
    ]

# Process vision inputs
image_inputs, video_inputs = process_vision_info(messages)

# Preprocess video if the input type is video
if input_type == "video":
    video_inputs = preprocess_video(file_paths, max_frames=20)

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