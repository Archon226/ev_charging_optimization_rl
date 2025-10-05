# EV Charging Optimization with Reinforcement Learning

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview
This is a **group project** focused on optimizing electric vehicle (EV) charging schedules using **Deep Reinforcement Learning (DQN)**. The goal is to manage charging times efficiently to reduce grid congestion and improve energy utilization. The project implements a DQN-based model and explores alternative algorithms for performance improvement.

---

## Key Features
- Deep Reinforcement Learning model for EV charging optimization  
- Custom simulation environment to model EV charging and grid load  
- Performance evaluation: peak load reduction and energy efficiency  
- Collaborative development with emphasis on algorithm design and analysis  

---


## Project Structure
```bash

EV_Literature_Review_Papers/      # Papers collected for literature review
References/                       # Reference materials
ev_charging_rl_project/           # Main project folder with code and scripts
original_Datasets/                # Raw datasets used in the project
Dataprep.ipynb                     # Notebook for data preparation
citydata.ipynb                     # Notebook for city-specific data analysis
ev_charging_rl_project.zip         # Backup of main project folder
requirements.txt                   # Project dependencies
README.md                           # Project documentation
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Archon226/ev_charging_optimization_rl.git
cd ev_charging_optimization_rl
````

### 2. Set up Python environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Explore Notebooks

* **Dataprep.ipynb**: Prepares datasets for training and simulation.
* **citydata.ipynb**: Analyzes city-specific data for EV charging patterns.

### 2. Run Main Project

* Navigate to `ev_charging_rl_project/` folder.
* **Train the model**:

```bash
python train.py
```

* **Evaluate the model**:

```bash
python evaluate.py
```

* **Visualize results**: Use scripts or notebooks in `ev_charging_rl_project/` to generate graphs and performance metrics.

---


## Contributions

This project was a **collaborative effort**. Team members contributed to:

* Algorithm implementation and optimization
* Simulation environment development
* Data analysis and visualization
* Literature review and documentation

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

* Thanks to all team members for their contributions.
* Special mention to [Hatim Shahera](https://github.com/hatimshahera) for guidance on the original repository.

---

## Optional Enhancements

* Add screenshots or output plots inside `ev_charging_rl_project/` and reference them in README.
* Include badges for Python version, license, and build status (if using CI/CD).
* Link to demo videos or hosted applications (if available).


