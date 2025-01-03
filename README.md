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
![KTO](images/kto_train_loss_v2.png)


•	SimPO: This algorithm did not converge within the experiment’s settings.

## Generation examples

| Prompt                                                                                                                                                          | DPO Answer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Mistral Base Answer                                                                                                                                                                                                                           | KTO Answer (Qwen2.5-0.5B) |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| Write a negative review on product with name Amazon 5W USB Official OEM Charger and Power Adapter for Fire Tablets and Kindle eReaders                         | I have to say this is the worst charger I have ever used. I have a Kindle Oasis and it takes forever to charge. I have a 2 amp charger that charges my Kindle in about 2 hours but this charger takes 4 hours to charge the same device. I have tried it on 3 different outlets and it still takes forever. I have a 2 amp charger that I bought from a different company and it works great.                                                                                                                        | Firstly, the charging speed is disappointingly slow. It takes an excessively long time to charge my Fire tablet, which is inconvenient and frustrating. I had expected a faster charging speed, given that                                  | Amazon 5 W USB Official OEM Charger and Power Adapter are slightly better than the originals, but this lamp still doesn’t truly meet my hallmarks for compatibility. I’ve seen no official firmware updates to it. Spend the extra money on the NP-418 USB laptop to me. |
| Write a negative review on product with name AmazonBasics AAA Performance Alkaline Batteries (36 Count)                                                        | I was very disappointed with these batteries. I bought them to use in my smoke detector but they only lasted a few days. I have been using Energizer batteries for years and they last for a year or more. I would not recommend these batteries. I will be going back to Energizer.                                                                                                                                                                                                                                      | Title: Disappointing Quality - AmazonBasics AAA Performance Alkaline Batteries  \n\nRating: 1 out of 5 stars  \n\nReview:  \n\nI was initially drawn to the AmazonBasics AAA Performance Alkaline Batteries (36 Count) due to their competitive price and the promise of high performance. However, my experience with these batteries has been far from satisfactory. Upon receiving the batteries, I immediately noticed tha |AmazonBasics is a great brand that produces abundant and reliable aluminum-alloy batteries. Unfortunately, this is one of their more flawed goods, and you would rather have hassle-free shots no matter how you wire and charge them. -Alex Young|
