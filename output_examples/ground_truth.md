### Task: Match Cups and Plates by Color

Three colored cups (yellow, green, pink) and matching plates are arranged in two parallel rows.  
**Goal**: Place each cup on the plate of the same color.

 **Correct Execution Steps**:
1. Yellow cup → yellow plate  
2. Green cup → green plate  
3. Pink cup → pink plate  

---

#### Example 1: Z1_T11 – Successful Execution

**Ground Truth**:
- All cups were successfully placed on their corresponding plates.
- No failures occurred.

---

#### Example 2: Z2_T13 – Execution Failure (Pink Cup Not Grasped)

**Ground Truth**:
- The pink cup was not grasped and remained in its initial position.
- Obvious physical failure.

---

#### Example 3: Z1_T14 – Execution Failure (Yellow Cup Dropped)

**Ground Truth**:
- The yellow cup was dropped and not placed on the yellow plate.
- Obvious physical failure.

---

### Task: Stack Cups in Ascending Order

Three numbered cups (1: blue, 2: pink, 3: gray) are available.  
**Goal**: Stack them in ascending order (1 → 2 → 3) on top of a target cup within a marked goal area.

**Correct Execution Steps**:
1. Pick and place cup 1 (blue)  
2. Pick and place cup 2 (pink) on top of cup 1  
3. Pick and place cup 3 (gray) on top of cup 2  

---

#### Example 1: Z2_T22 – Sequencing Error (2 → 1 → 3)

**Ground Truth**:
- All cups were picked up and placed.
- The stacking order was incorrect: 2 → 1 → 3.

---

#### Example 2: Z2_T24 – Execution Failure (Gray Cup Dropped)

**Ground Truth**:
- The robot successfully placed cups 1 (blue) and 2 (pink).
- During the third step, the gray cup (3) was dropped and not placed.
- Obvious physical failure.
