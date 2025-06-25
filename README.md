# TRI_PROJ_2 Project - Group P1L

## Group Members

- Cristiano Rocha - up202108813
- Gonçalo Santos - up202108839
- João Lima - up202108891
- Tiago Moreira - up202107533

---

## Description

This project contains the implementation of controllers for robots in a reinforcement learning (RL) environment.

### Controller Structure

- **rival_controller**  
   Contains the logic for the rival robot.

- **train_controller**  
   Responsible for training the RL robot.

- **use_controller**  
   Uses the trained model (without retraining) and collects performance metrics over multiple episodes.

---

## Requirements

To run this project, you must have the following installed:

- [Webots](https://cyberbotics.com/)
- Python 3.x
- The following Python libraries:
  - `numpy`
  - `matplotlib`
  - `torch`
  - `gymnasium`
  - `stable-baselines3`
- Additional Python modules:
  - `os`
  - `sys`
  - `math`
  - `random`
  - `time`

You can install the main required Python libraries using:

```bash
pip install numpy matplotlib torch gymnasium stable-baselines3
```

---

## Usage Instructions

1. Clone the [repository](https://github.com/Cristy17001/TRI_PROJ_2.git).
2. Install the required dependencies (see Requirements above).
3. Use each controller as needed:
   - For rival logic, use `rival_controller`.
   - To train the RL robot, use `train_controller`.
   - To evaluate and collect metrics, use `use_controller`.
4. To switch between training and metric collection, change the controller assigned to the main robot in the `world.wbt` file to either `train_controller` or `use_controller` as appropriate.

---
