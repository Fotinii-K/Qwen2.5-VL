from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from vid_processing import preprocess_video  

# Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# Load the processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Define the input paths
image_path = "/home/nakama6000/Documents/git/Qwen2.5-VL/demo.jpeg"
video_path = "/home/nakama6000/Documents/git/Qwen2.5-VL/fruits_in_basket.mp4"

# Ask the user which type of input to process
input_type = "video"

if input_type == "image":
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
elif input_type == "video":
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {
                    "type": "text",
                    "text": "You are working as the 'decision-making core' of a single armed household robot, capable of working on different tasks. You will receive a collage of sequence of images in chronological order, illustrating the progression of a task currently executed by the robot, with the last image representing your current observation. Your purpose is to infer whether the task has been successfully achieved and address next steps in case of a failure, based on the defined goal. Do not make assumption of your own. In this scenario the robot's task is to place all the fruits inside the basket.",
                },
            ],
        }
    ]
else:
    print("Invalid input type. Please enter 'image' or 'video'.")
    exit()

# Process vision inputs
image_inputs, video_inputs = process_vision_info(messages)

# Preprocess video if the input type is video
if input_type == "video":
    video_inputs = preprocess_video(video_path, max_frames=20)

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
# generated_ids = model.generate(**inputs, max_new_tokens=300 if input_type == "video" else 128)
generated_ids = model.generate(**inputs, max_new_tokens=128)

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