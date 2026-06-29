"""
Script for plotting all CAVs selections for all models
"""

import subprocess
import os


models = ["gpt20b", "gemma4", "gpt120b", "llama3.3"]
splits = ["train", "test"]

for model in models:
    os.mkdir(f"results/artifact_evaluation/{model}")
    for split in splits:
        print(f"Plotting {model} {split}")
        subprocess.run(
            [
                "python",
                "evaluation/plot_bev_positions.py",
                "--dataset_root",
                f"opv2v_data_dumping/{split}",
                "--selection_csv",
                f"results/acsos2026/{model}/opv2v_{split}/selection.csv",
                "--all",
                "--output",
                f"results/artifact_evaluation/{model}/bev_positions",
            ],
            check=True,
        )
        

