"""
Script for plotting all CAVs selections for all models
"""

import subprocess
import os

models = ["gpt20b", "gemma4", "gpt120b", "llama3.3"]
splits = ["train", "test"]

for model in models:
    print(f"Plotting {model}")
    subprocess.run(
        [
            "python",
            "evaluation/plot_fusion_strategy.py",
            "--selection_csv",
            f"results/acsos2026/{model}/opv2v_test/selection.csv",
            f"results/acsos2026/{model}/opv2v_train/selection.csv",
            "--output",
            f"results/artifact_evaluation/fusion_strategy_{model}.pdf",
        ],
        check=True,
    )
        

