import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image  # Import PIL for image processing
import torch
from qwen_vl_utils import process_vision_info

# Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
# Load the processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Initialize the camera (0 is usually the default camera, change if needed)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

message_general_context = """Create a plan for achieving the desired task based on the predefined actions that the robot is capable of doing.
                             
                             Actions and their corresponding annotations are defined in the following list:
                             navigate to(arg1): Navigate to the arg1, which can be a object or a room. If it’s a object, 
                             you should get to a place where arg1 is reachable for the robot. 
                             grasp(arg1): Grasp arg1. Preconditions: arg1 is within reachable distance and no object is currently held. 
                             Postconditions: arg1 is being held. 
                             place onTop(arg1, arg2): Place arg1 on top of arg2. Preconditions: arg1 is currently being held, 
                             and arg2 is reachable. Postconditions: arg1 is put on top of arg2.
                             ... [rest of the context]"""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab a frame.")
        break

    # Convert the frame to RGB as many models expect RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(rgb_frame)

    # Prepare messages with the current frame
    messages = [
        {"role": "system", "content": message_general_context},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},  # Pass the PIL Image here
                {"type": "text", "text": "Clean the table"}
            ],
        }
    ]

    # Process vision inputs
    image_inputs, _ = process_vision_info([messages])

    # Prepare inputs for the model
    inputs = processor(
        text=[processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
        do_rescale=False  # Adjust according to your model's requirements
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
    print("Response:")
    print(assistant_response)

    # Display the frame
    cv2.imshow('Real-Time Stream', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()