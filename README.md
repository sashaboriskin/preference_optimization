# preference_optimization

This repository demonstrates various algorithms for preference optimization, including Direct Preference Optimization (DPO), Kahneman-Tversky-based Preference Optimization (KTO), and Simple Preference Optimization (SimPO) using TRL trainers.

## Experiment Overview

The experiment aims to compare the effectiveness and convergence of the three preference optimization algorithms:

- **DPO (Direct Preference Optimization)**
- **KTO (Kahneman-Tversky-based Preference Optimization)**
- **SimPO (Simple Preference Optimization)**

The evaluation is based on the training loss and other performance metrics logged during the experiment.

## Reproducing the Experiment

Follow the steps below to reproduce the experiment:

### 1. Data preparation

Run script to preprocess kaggle data format to csv data format
```python
python po_data_prepare.py
```

### 2. Install Dependencies

Ensure all required libraries are installed by running:

```bash
bash setup.sh
```

### 3. Run the Experiment

To run the experiment, execute the `run_script.sh` file with needed py file and config:

For example: 

```bash
bash run_script.sh kto.py kto_v1.yaml
```

Before running the script, ensure the following:

- Correct Accelerate Configuration: Verify that the correct configuration for accelerate is set up.
- Experiment Script Selection: Choose the appropriate Python script for the experiment:
	- `dpo.py` for Direct Preference Optimization.
    - `kto.py` for Kahneman-Tversky-based Preference Optimization.
	- `simpo.py` for Simple Preference Optimization.
- Experiment Configuration File: Confirm the corresponding yaml configuration file for the selected experiment is correctly set:
	- `dpo_v1.yaml` for Direct Preference Optimization.
	- `kto_v1.yaml` for Kahneman-Tversky-based Preference Optimization.
	- `simpo_v1.yaml` for Simple Preference Optimization.

The script will execute the training process and automatically:

	1.	Merge the LoRA adapter with the base model.
	2.	Save the final model in the `models/` directory.

## Experiment Results

The following results were observed during the experiment:
•	DPO and KTO: Both algorithms demonstrated stable convergence.

### DPO
![DPO](images/dpo_train_loss.png)

### KTO
![KTO](images/kto_train_loss.png)


•	SimPO: This algorithm did not converge within the experiment’s settings.
