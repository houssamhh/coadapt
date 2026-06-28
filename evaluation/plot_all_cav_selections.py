"""
Script for plotting all CAVs selections for all models
"""

import subprocess
import os

models = ["gpt20b", "gpt120b", "gemma4", "llama3.3b"]
splits = ["train", "test"]

for model in models:
    print(f"Plotting {model}")
    subprocess.run(
        [
            "python",
            "evaluation/plot_cav_selection.py",
            "--selection_csv",
            f"results/acsos26/{model}/opv2v_test/selection.csv",
            f"results/acsos26/{model}/opv2v_train/selection.csv",
            "--output",
            f"results/artifact_evaluation/cav_selection_{model}.pdf",
        ],
        check=True,
    )
        

