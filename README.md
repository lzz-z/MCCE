# MOLLM

This repository contains the source code and scripts for the MOLLM project. The project is organized as follows:

## Project Structure

/home/v-nianran/src/MOLLM <br>
├── algorithm <br>
│ ├── base.py <br>
│ ├── MOO.py <br>
│ └──  PromptTemplate.py <br>
├── data <br>
│ ├── data_goal5.json <br>
│ └── zinc.tab <br>
├── problem <br>
│ ├── molecules <br>
│ │ ├── evaluator.py <br>
│ │ ├── goal5_gemini.yaml <br>
│ │ └── molecule.yaml <br>
├── eval.py <br>
├── main.py <br>
└── test.ipynb <br>

### Running the experiments
  - `python main.py base.yaml`: You only need to config the YAML file to change the settings and goals, path is relative path under config/
  - `python main.py base.yaml --resume`: to automatically resume from last saved .pkl file

### Description of Files and Directories

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

# 🧬 YAML File Specification for Multi-Objective Molecular Optimization

This project uses a structured YAML format to define tasks for multi-objective molecular optimization. Each YAML file outlines the task description, output format, mutation/crossover guidelines, and optimization objectives. This format ensures consistency and clarity when interfacing with molecule generation and optimization models.

---

## 📁 File Structure Overview

A standard YAML file should contain the following fields:

| Field Name              | Required | Description |
|-------------------------|----------|-------------|
| `description`           | ✅       | A brief overview of the task; describe what the model should achieve |
| `example_output`        | ✅       | Defines the expected output format (must include `<candidate>` tags and your answer format) |
| `mutation_instruction`  | ⭕       | Suggested mutation operations to guide molecule structure changes |
| `crossover_instruction` | ⭕       | Suggested crossover operations (optional, can be left empty) |
| `other_requirements`    | ⭕       | Any additional constraints, e.g., "Molecules must be valid" |
| Optimization Objectives (e.g. `qed`, `sa`) | ✅ | One or more task-specific objectives, each described in a dedicated field |

---

## 📝 Example YAML File

```yaml
description: This task is to propose better molecules according to multiple objectives.

example_output: 'Each output new candidate must start with <candidate> and end with </candidate> in SMILES format. Example: <candidate>c1ccccc1</candidate>'

mutation_instruction: 'Example operations include:
  1. Modify functional groups
  2. Replace atoms or bonds
  3. Add/remove small substituents
  4. Ring modifications
  5. Stereochemistry changes
  6. Property-specific optimizations
  '

crossover_instruction: ''

other_requirements: The output molecules should be valid.

qed: QED (Quantitative Estimate of Drug-likeness) is a measure that quantifieshow
  'drug-like' a molecule is based on properties such as molecular weight,solubility,
  and the number of hydrogen bond donors and acceptors.Adding functional groups that
  improve drug-like properties (e.g., small molecular size,balanced hydrophilicity)
  can increase QED, while introducing large, complex, or highly polar groups can decrease
  it.
logp: LogP is the logarithm of the partition coefficient, measuring the lipophilicityor
  hydrophobicity of a molecule, indicating its solubility in fats versus water.Adding
  hydrophobic groups (e.g., alkyl chains or aromatic rings) increases LogP,while adding
  polar or hydrophilic groups (e.g., hydroxyl or carboxyl groups) decreases it.


