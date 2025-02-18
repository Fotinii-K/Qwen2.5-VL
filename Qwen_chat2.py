from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
from vid_processing2 import preprocess_video

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

# images
# file_paths = ["/home/nakama6000/Documents/git/Qwen2.5-VL/prfmc/demo5.png"]
file_paths = ["/home/nakama6000/Documents/git/Qwen2.5-VL/prfmc/tasks/navigate.png"]  
# file_paths = ["/home/nakama6000/Documents/git/Qwen2.5-VL/prfmc/tasks/pick.png"]  
# file_paths = ["/home/nakama6000/Documents/git/Qwen2.5-VL/prfmc/tasks/place.png"]  
# file_paths = ["/home/nakama6000/Documents/git/Qwen2.5-VL/prfmc/tasks/demo4.png"]

#videos
# file_paths = "/home/nakama6000/Documents/git/Qwen2.5-VL/fruits_in_basket.mp4"  

input_type = get_input_type(file_paths)

# Define General Context Message 
# task completion evaluation
# message_general_context = """You are working as the 'decision-making core' of a single armed household robot, capable of working on 
#                             different tasks. You will receive a series of frames or a collage of sequence of images in chronological order, illustrating the 
#                             progression of a task currently executed by the robot, with the last image representing your current observation. 
#                             Your purpose is to infer whether the task has been successfully achieved and address next steps in case of 
#                             a failure, based on the defined goal. 
#                             Do not make assumption of your own. """

#spatial-temporal understanding
message_general_context = """You are working as the 'decision-making core' of a household robot, capable of working on different tasks. 
                             Based on the progression and changes depicted in the images, carefully organize them to the correct chronological order,
                             according to the requirements of the task.
                             Please respond only with the order of images."""

# create an array
# message_general_context = """Guide the robot to achieve the desired new position by adjusting 
#                             the motion vector [x, y, z, theta_x, theta_y, theta_z]. Each component controls movement or 
#                             rotation: x, y, z values of 1 move in the positive direction, -1 in the negative. 
#                             Similarly, theta_x, theta_y, theta_z values of 1 rotate counterclockwise, while -1 rotates clockwise. 
#                             The model must determine and update this vector to direct the robot efficiently toward its goal.
#                             Consider that your are acting as an external observer, but the array needs to be adjusted to the point of view of the robot.
#                             Also include the final array, which can only take the values -1,0 and 1 as described above."""

# message_general_context = """ Create a plan for achieving the desired task based on the predifined actions that the robot is capable of doing. 
                            
#                             Actions and their corresponding annotations are defined in the following list:

#                             navigate to(arg1): Navigate to the arg1, which can be a object or a room. If it’s a object, 
#                             you should get to a place where arg1 is reachable for the robot. 
#                             grasp(arg1): Grasp arg1. Preconditions: arg1 is within reachable distance and no object is currently held. 
#                             Postconditions: arg1 is being held. 
#                             place onTop(arg1, arg2): Place arg1 on top of arg2. Preconditions: arg1 is currently being held, 
#                             and arg2 is reachable. Postconditions: arg1 is put on top of arg2.
#                             place inside(arg1, arg2): Place arg1 inside of arg2. Preconditions: arg1 is currently being held,
#                             and arg2 is reachable. Postconditions:arg1 is put inside of arg2. 
#                             place under(arg1, arg2): Place arg1 under arg2. Preconditions: arg1 is currently being held, and 
#                             arg2 is reachable. Postconditions: arg1 is put under arg2. 
#                             place onLeft(arg1, arg2): Place arg1 on left of arg2. Preconditions: arg1 is currently being held, 
#                             and arg2 is reachable . Postconditions: arg1 is put on left of arg2. 
#                             place onRight(arg1, arg2): Place arg1 on right of arg2. Preconditions: arg1 is currently being held, 
#                             and arg2 is reachable. Postconditions: arg1 is put on right of arg2. 
#                             open(arg1): Open arg1. Preconditions: Arg1 is closed, and arg1 is reachable. Postconditions: Arg1 is open. 
#                             close(arg1): Close arg1. Preconditions: Arg1 is open, and arg1 is reachable. Postconditions: Arg1 is closed. 
#                             slice(arg1): Slice arg1, the item needs to be placed on the countertop. Preconditions: Arg1 is not sliced, 
#                             and arg1 is reachable. Postconditions: Arg1 is sliced. 
#                             wipe(arg1, arg2): Wipe across the surface of arg2 with arg1. Preconditions: Arg1 is currently being held, 
#                             and arg2 is reachable. Postconditions: Arg1 continues to be held, arg2 holds state unchanged. 
#                             wait(arg1): Wait for arg1 seconds. Preconditions: None. Postconditions: arg1 second(s) has(have) passed. 
#                             toggle(arg1): Press the button of arg1 to turn it on or off, Preconditions: Arg1 is open or closed, and 
#                             arg1 is reachable. Postconditions: Arg1 is closed or open."""


if input_type in ["image", "multi_image"]:

    messages = [
        {"role": "system", "content": message_general_context},  
        {
            "role": "user",
            "content": [
                {"type": "image", "image": path} for path in file_paths
            ] + [
                {"type": "text", "text": 
                 """The task is to navigate towards the bin."""
                #  """The task is to pick up the glass from the table."""
                #  """The task is to place the trash inside the bin."""
                # """Clean the table"""
                # """Evaluate if the table is fully cleaned, otherwise locate where the robot should clean next."""
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