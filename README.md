# MOLLM

This repository contains the source code and scripts for the MOLLM project. The project is organized as follows:

## Project Structure

/home/v-nianran/src/MOLLM <br>
├── algorithm <br>
│ ├── base.py <br>
│ └──  MOO.py <br>
├── dataset <br>
│ └── collect_data.py <br>
├── eval.py <br>
├── main.py <br>
├── model <br>
│ ├── MOLLM.py <br>
│ ├── MOScigpt.py <br>
│ ├── load_Scigpt.py <br>
│ ├── util.py <br>
│ └── LLM.py <br>
└── test.ipynb <br>

### Running the experiments
  - `python main.py --config config/base.yaml`: You only need to config the YAML file to change the settings and goals.

### Description of Files and Directories

- **dataset/**
  - `collect_data.py`: Script for collecting and processing datasets used in the MOLLM project.

- **eval.py**
  - Tools for evaluating the model's performance on test datasets.

- **main.py**
  - The main entry point for training and testing the MOLLM model.

- **model/**
  - `MOLLM.py`: Contains the implementation of the MOLLM model.
  - `LLM.py`: Implemetation of LLM model
  - `MOScigpt.py`: MO SciGPT model and its algorithm.
  - `load_Scigpt.py`: Utility script to load the SciGPT model.
  - `util.py`: Utility functions used across the project, including NSGA-II currently.

- **test.ipynb**
  - Jupyter notebook containing experiments, visualizations, or other exploratory analysis related to the MOLLM project.
  - Including the tutorial of how to modify the main components

## Getting Started

### Prerequisites

- Python 3.9
- Requirements are same as those in SFM_framework, please note, loading scigpt will need to put the SFM_framework folder under the main directory. Which is a TODO.


