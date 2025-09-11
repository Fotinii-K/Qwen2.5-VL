import torch
import numpy as np
import os
import re
import json
from datetime import datetime
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import utils


# Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
output_root = os.path.join(script_dir, "output_examples")

responses_output_dir = os.path.join(output_root, "model_responses_prompted")
os.makedirs(responses_output_dir, exist_ok=True)

selected_frame_output_dir = os.path.join(responses_output_dir, "selected_frame")
os.makedirs(selected_frame_output_dir, exist_ok=True)

task_recordings_dir = os.path.join(script_dir, "task_recordings")

# Create a unique output folder for this evaluation run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
session_folder = os.path.join(responses_output_dir, f"response_{timestamp}")
os.makedirs(session_folder, exist_ok=True)

# Clean up GPU memory and temporary cache folders
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
utils.clear_folders()       

# Load the model and processor
model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# Define inference function
def inference(messages, max_new_tokens=100):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,      # Control output length
        do_sample=False,                    # Disable sampling - greedy decoding
        temperature=1,                      # Set to 1 to avoid HF warning; ignored when do_sample=False 
        num_beams=1,                     
    )
    generated_ids_trimmed = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids,  generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]

# Define inference function for video processing
def process_video_inference(messages, max_new_tokens=1000, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(
        **inputs, 
        max_new_tokens = max_new_tokens,      # Control output length
        do_sample = False,                    # Disable sampling - greedy decoding
        temperature = 1,                      # Set to 1 to avoid HF warning; ignored when do_sample=False 
        num_beams = 1,                        
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

# Define the robotic task to be evaluated
task_description = """
Task: Matching Cups and Plates by Color
Description: Three cups and three plates are placed on a table in random order. The cups and plates are arranged in two parallel rows, with cups on one side and plates on the other. Each cup and plate is of a distinct color, and each color appears exactly once among cups and once among plates.
Objective: Place each cup onto the plate that matches its color. Each cup must be matched with exactly one plate, and each plate must have exactly one cup. All items must be matched.
Success Criteria: All cups are placed on the plate with the same color.
Failure Conditions (any of the following):
-	A cup is placed on a plate of the wrong color.
-	A cup is left unmoved or placed outside of a plate or not placed on any plate at all.
-	A cup is not placed properly on any plate (e.g. dropped or stacked on another cup/plate)
-	More than one cup is placed on the same plate.
Notes:
-	The initial positions or arrangement of the cups and plates do not matter. Only the final matching is evaluated.
"""

# task_description = """
# Task: Stack Numbered Cups in Sequential Order
# Description: Three cups labeled with numbers 1, 2, and 3 are placed in a row on a table. Another cup, referred to as the “target cup” is located across from them, inside a marked area that indicates the stacking position.
# Objective: Stack the three numbered cups (1, 2, and 3) on top of the target cup in the correct ascending numerical order, such that: cup 1 is at the bottom (on the target cup), cup 2 is in the middle and then cup 3 is on top.
# Success Criteria: Cups are stacked in the correct order.
# Failure Conditions (any of the following):
# -	A cup is stacked out of order.
# -	A cup is dropped.
# -	A cup is failed to be grasped.
# Notes:
# -	The initial placement and the color of the cups do not affect evaluation. Only the final stack configuration is evaluated.
# """

# Batch processing of videos
def process_video(video_path):
    video_name = os.path.basename(video_path)
    system_prompt = """
        You are a robotic task evaluator.
        Your role is to analyze videos showing a robotic arm equipped with a gripper performing tasks.
        You will evaluate whether the task was completed successfully, or (partially) failed.
        """
    
    success_prompt = f"""
        Is the task currently being executed successfully completed, according to {task_description}? Explicitly address it, with a yes or no and why.
        Please respond using the following structure:
        Task Evaluation: < > 
        Justification: < >
        """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", 
        "content": [
            {"type": "text", "text": task_description},
            {"type": "video", "video": video_path},
            {"type": "text", "text": success_prompt},
            ],
        }
    ]
   
    task_success = process_video_inference(messages)
    print(f"Task Success Evaluation for {video_name}:\n", task_success)
    messages.append({"role": "assistant", "content": task_success})


    # If task failed
    cause_text = None
    grounding_response = None
    localization_response = None
    response_bbox = None
    selected_frame_path = None
    frame_bbox_path = None
    planning_response = None

    if "Task Evaluation: No" in task_success:
        failure_prompt = """
            You previously concluded that the task was not successfully completed.
            Based on the video input, describe the **specific visual cause** of the failure with enough 
            detail to allow temporal localization.
            Include:
                - What object(s) are involved?
                - What unexpected visual behavior occurred? 
            Please respond using the following structure:
            Cause: < short but precise description > """
        
        messages.append(
            {"role": "user", "content": [
                {"type": "text", "text": failure_prompt},
                ], 
            }
        )
        cause_text = process_video_inference(messages)
        print(f"Failure Cause for {video_name}:\n", cause_text)
        messages.append({"role": "assistant", "content": cause_text})


        video_path, frames, timestamps = utils.get_video_frames(video_path, num_frames=128)
        prompt_vid_grounding = f"""
            Localize when the visual cause that explains the failure is visible in the video.
            Please provide:
            1. The start and end timestamps in the format: 'The failure described occurs between mm:ss.ff and mm:ss.ff' in the video. """
        
        messages.append(
            {"role": "user", "content": [
                {"type": "text", "text": prompt_vid_grounding},
                ],
            } 
        )
        grounding_response = process_video_inference(messages, max_new_tokens=1000, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28)
        print(f"Video Grounding for {video_name}:\n", grounding_response)
        messages.append({"role": "assistant", "content": grounding_response})

        timestamp_match = re.search(r"(\d{2}:\d{2}\.\d{2}) and (\d{2}:\d{2}\.\d{2})", grounding_response)
        if timestamp_match:
            timestamp_start, timestamp_end = timestamp_match.groups()  
            print(f"First timestamp extracted: {timestamp_start}")
            print(f"Second timestamp extracted: {timestamp_end}")

            timestamp_start = utils.time_to_seconds(timestamp_start)
            timestamp_end = utils.time_to_seconds(timestamp_end)

            current_frames = []
            for frame, timestamp in zip(frames, timestamps):
                if timestamp[0] > timestamp_start and timestamp[1] < timestamp_end:
                    current_frames.append(frame)

            current_frames = np.array(current_frames)
            current_image_grid = utils.create_image_grid(current_frames, num_columns=8)

            current_image_grid.resize((480, (int(len(current_frames) / 8) + 1) * 60))
            current_image_grid.show()

            saved_frames = [f"frame_{i:03d}.png" for i in range(len(current_frames))]


        localization_frame = f"""
                    As previously identified the failure occurred between {timestamp_start} and {timestamp_end}, and the cause was: "{cause_text}
                    Please analyze only this time window in the video, and:
                    Identify the **single most clear frame** where the failure is visually obvious."""

        messages.append(
            {"role": "user", "content": [
                    {"type": "text", "text": localization_frame},
                ]
            },
        )

        localization_response = process_video_inference(messages)
        print(f"Localization for {video_name}:\n", localization_response)
        messages.append({"role": "assistant", "content": localization_response})

        timestamp_match = re.search(r"(\d{2}:\d{2}\.\d{2})", localization_response)

        if timestamp_match:
            selected_timestamp = timestamp_match.group(1)  
            print(f"Timestamp extracted: {selected_timestamp}")

            selected_timestamp = utils.time_to_seconds(selected_timestamp)

            for frame, timestamp in zip(frames, timestamps):
                if timestamp[0] > timestamp_start and timestamp[1] < timestamp_end:

                    selected_frame_path = os.path.join(selected_frame_output_dir, f"{os.path.splitext(video_name)[0]}_selected_frame.png")
                    Image.fromarray(frame).save(selected_frame_path)

                    selected_frame = Image.open(selected_frame_path)
                    selected_frame.thumbnail([640, 640], Image.Resampling.LANCZOS)
            
            selected_frame.show()

        prompt_bbox = f""" Based on the identified failure frame, determine the next action the robot should take to correct the failure and successfully complete the task.

                    Please provide:
                    1. A bounding box on the failure frame to indicate **the object the robot should move towards or interact with** to correct the issue and output all the coordinates in JSON format.
                    2. The name of the object in the bounding box.
                    """
        
        messages.append(
            {"role": "user", "content": [
                    {"type": "text", "text": prompt_bbox},
                    {"type": "image", "image": selected_frame_path}
                    ],
            },
        )
   
        messages_bbox = [
            {"role": "system", "content": "As an AI assistant, you specialize in accurate image object detection, delivering coordinates in plain text format 'x1,y1,x2,y2 object'."},
            {"role": "user", "content": [
                {"type": "text", "text": messages},
                {"type": "text", "text": prompt_bbox},
                {"type": "image", "image": selected_frame_path}
            ]}
        ] 
        frame_bbox = Image.open(selected_frame_path)
        response_bbox, input_height, input_width = utils.inference_su_video_bbx(frame_bbox, messages_bbox)
        messages.append({"role": "assistant", "content": response_bbox})
        
        print(f"Bounding box output for {video_name}:\n", response_bbox)

        # Extract structured data from model response
        json_response = utils.parse_json_su(response_bbox)
        frame_bbox.thumbnail([640, 640], Image.Resampling.LANCZOS)
        utils.plot_bounding_boxes(frame_bbox, json_response, input_width, input_height)

        # Save bounding box visualization in session folder
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_bbox_path = os.path.join(session_folder, f"{base_name}.png")
        frame_bbox.save(frame_bbox_path)
        
        planning_prompt = f""" Explain the corrective action the robot should take, based on the failure and identified object, to successfully complete the task.
                    Please provide:
                    1. A clear and concise explanation of what the robot should do next.
                    """

        messages.append(
            {"role": "user", "content": [
                    {"type": "text", "text": planning_prompt},
                    ],
            },
        )

        planning_response = process_video_inference(messages)
        print(f"Corrective planning for {video_name}:\n", planning_response)
        messages.append({"role": "assistant", "content": planning_response})

    return video_name, task_success, cause_text, grounding_response, localization_response, response_bbox, selected_frame_path, frame_bbox_path, planning_response


# List of video paths
video_paths = [
    os.path.join(task_recordings_dir, "TASK1", "Z2_T13.avi"),
    # os.path.join(task_recordings_dir, "TASK2", "Z2_T24.avi"),
]

# Save evaluation results in the session folder.
def save_response_outputs(
    video_name,
    session_folder,
    task_success=None,
    cause_text=None,
    grounding_response=None,
    localization_response=None,
    response_bbox=None,
    selected_frame_path=None,
    frame_bbox_path=None,
    planning_response=None
):
    responses = {"video_name": f"Responses for {video_name}"}

    # Add non-empty list values
    def add_if_present(key, value):
        if isinstance(value, list) and value:
            responses[key] = value[0]
        elif value is not None:
            responses[key] = value

    add_if_present("output_text_task_success", task_success)
    add_if_present("output_text_cause", cause_text)
    add_if_present("response_vid_grounding", grounding_response)
    add_if_present("response_localization_frame", localization_response)
    add_if_present("response_bbox", response_bbox)
    add_if_present("response_planning", planning_response)

    base_name = os.path.splitext(video_name)[0]

    if frame_bbox_path is not None and os.path.exists(frame_bbox_path):
        responses["frame_bbox_path"] = os.path.basename(frame_bbox_path)  # e.g., "Z1_T11_frame_bbox.png"


    # Save JSON
    json_output_path = os.path.join(session_folder, f"{base_name}.json")
    with open(json_output_path, 'w') as f:
        json.dump(responses, f, indent=4)

    return session_folder

# Process each video
for path in video_paths:
    try:
        print(f"\n--- Processing {path} ---")
        video_name, task_success, cause_text, grounding_response, localization_response, response_bbox, selected_frame_path, frame_bbox_path, planning_response = process_video(path)

        save_response_outputs(
            video_name=video_name,
            session_folder=session_folder,
            task_success=task_success,
            cause_text=cause_text,
            grounding_response=grounding_response,
            localization_response=localization_response,
            response_bbox=response_bbox,
            selected_frame_path=selected_frame_path,
            frame_bbox_path=frame_bbox_path,
            planning_response=planning_response

        )

        selected_frame_path = None
        frame_bbox_path = None
    except Exception as e:
        print(f"Error processing {path}: {e}")