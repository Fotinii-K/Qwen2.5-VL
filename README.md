# Dynamic Sequential Prompting: Autonomous Evaluation of Robotic Task Execution

This repository implements **Dynamic Sequential Prompting**, a structured framework for evaluating the ability of off-the-shelf Vision-Language Models (VLMs), specifically Qwen2.5-VL, to autonomously assess robotic task execution from video input.

Rather than relying on explicit task prompts, the framework leverages Qwen2.5-VL to self-infer **task goals and requirements**, **evaluate task success**, and for failed executions, **determine causes**, **localize failures in time and space**, and **propose corrective actions** — all through a multi-stage, sequential prompting strategy.

The framework reveals key insights into VLM reasoning, including:
- Performance degradation in long-horizon tasks.
- Challenges in mid-task failure detection and temporal consistency.
- Superior accuracy when the model self-infers (vs. guided analysis).

---

## How It Works 🧠

The evaluation is decomposed into six sequential stages:

1. **Task Identification, Goals & Requirements**  
   → "What task is being performed?"  
   → "What are the goals and requirements for success?"

2. **Task Evaluation**  
   → "Was the task completed successfully?"

3. **Failure Cause Analysis**  
   → "If not, what went wrong?"

4. **Temporal Localization**  
   → "When did the failure occur?"

5. **Spatial Grounding** 
   → "Where is the issue in the scene?"

6. **Corrective Action**  
   → "What should be done to fix it?"

Each stage is prompted sequentially, with the model's output from one stage used as context for the next.

Crucially, the model is **not given the task description upfront**, enabling a fair evaluation of its **autonomous reasoning capabilities**.

---

## ✅ Example 1: Correct Evaluation

**Scenario**: A robot attempts to match cups and plates by color but drops the yellow cup.

**Model Output**:
- **Task Identification**: 

  > "Task: The task involves a robotic arm manipulating cups and plates on a table."
  
  > "Goal: The goal appears to be organizing the cups and plates by placing each cup on a corresponding plate."
  
  > "Requirements for goal: The robotic arm must accurately identify and pick up each cup, then place it on the correct plate. The sequence should ensure that no cups are left unpaired or placed incorrectly on the wrong plate."

- **Task Evaluation**: 

    > "The robotic arm has successfully picked up and placed some cups on plates, but there are still some cups that remain unpaired and not correctly placed on their respective plates. For instance, the yellow cup is not on its plate, and the green cup is also not on its plate. Additionally, the robotic arm seems to have difficulty in consistently identifying and picking up the cups without errors."

- **Failure Cause**: 

    > "The robotic arm failed to correctly identify and pick up the yellow cup, causing it to remain on the table instead of being placed on its corresponding plate. This led to an incomplete pairing of the yellow cup with its plate."

- **Temporal Localization**: 

    > "The failure described occurs between 00:25.00 and 00:30.00 in the video."

- **Spatial Grounding**: 

    > ![alt text](yellow_cup_failure.png)
    

- **Corrective Action**: 

    > "The robot should attempt to pick up the yellow cup again, ensuring it is securely grasped before moving it to its corresponding plate. If the initial attempt fails, the robot should try a different approach, such as adjusting its grip or repositioning itself closer to the cup. Once the cup is successfully picked up, the robot should align it with the correct plate and place it down carefully to avoid any further misplacement."

---

## ❌ Example 2: Erroneous Evaluation 

**Scenario**: The robot attempts to stack numbered cups in sequential order but places cup 3 out of order.

**Model Output**:
- **Task Identification**: 

    > "Task: The task involves a robotic arm picking up cups from a table and stacking them on top of each other."

    > "Goal: The goal is to stack the cups in a specific order, as indicated by the numbers on the table."

    > "Requirements for goal: The robotic arm must accurately identify the correct cup based on its number, lift it without dropping it, and place it on the stack of cups. The cups need to be stacked in ascending order according to their numbers (1, 2, 3)."

- **Task Evaluation**: 

    > "The robotic arm is successfully identifying the cups based on their numbers, lifting them without dropping them, and placing them on the stack in ascending order according to their numbers (1, 2, 3). Each cup is picked up and placed correctly onto the stack, indicating that the task is being executed successfully."

- **Failure Cause**:  N/A
- **Temporal/Spatial**:  N/A
- **Corrective Action**:  N/A

---

## How to Use This Framework

This repository provides a complete framework for evaluating Qwen2.5-VL's ability to autonomously assess robotic task execution using Dynamic Sequential Prompting. The evaluation can be run in two modes:
- `evaluation_self_inferred.py`: Evaluates the model **without** an explicit task prompt (self-inference mode).
- `evaluation_prompted.py`: Evaluates the model **with** an explicit task prompt (guided analysis mode).

Follow the steps below to install and run the framework.

---

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Fotinii-K/qwen-vl-robot-task-evaluation.git
   cd qwen-vl-robot-task-evaluation
2. Create and activate a virtual environment:
    ```bash
    python -m venv venv  
    source venv/bin/activate    # Linux/Mac  
    venv\Scripts\activate       # Windows
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
----

### Running the Evaluation

To run the evaluation in self-inferred mode (no task prompt provided):

    python evaluation_self_inferred.py

To run the evaluation in prompted mode (with explicit task description):

    python evaluation_prompted.py



#### Note: 
The framework also allows for batch inference, where multiple task video clips can be tested in a single run. To evaluate multiple videos, add additional paths to the video_paths list in either evaluation_self_inferred.py or evaluation_prompted.py, depending on the evaluation mode.

---

### License & Attribution
This project uses code from the  QwenLM/Qwen2.5-VL repository under the Apache-2.0 License.
See the LICENSE file for details.