
# CoAdapt

## Project Description  

**CoAdapt** is an LLM-driven framework for adaptive collaborative perception in IIoT robotic swarms. 
A Large Language Model serves as a runtime fusion controller, jointly deciding which robots participate in the data fusion process and which fusion algorithm to apply based on the current spatial configuration of the robots and the network state.
**CoAdapt** is built on top of the [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) library for running data fusion on LiDAR data, and uses the [OPV2V dataset](https://mobility-lab.seas.ucla.edu/opv2v/).

## Repository Structure

Ths repository contains the following directories:

```

coadapt/ # Core CoAdapt scripts

	run_pipeline.py # Evaluation of the end-to-end CoAdapt pipeline

	run_baseline.py # Evaluation of a static baseline that selects intermediate fusion regardless of bandwidth fluctuations.

	run_rulebased_selection.py # Evaluation of rule-based approaches that select late fusion when bandwidth is low, and intermediate fuion otherwise

	run_intermediate_eval.py # Evaluation of intermediate fusion algorithms

	llm_robot_and_strategy_selector.py # CoAdapt's LLM-based robot and data fusion strategy selector

	scene_abstraction_module.py # CoAdapt's Scene Abstraction Module for LiDAR-to-text conversion

  

lidar_processing/ # Point cloud utilities
 

evaluation/ # Scripts used for evaluation and generating plots

trained-models/	#  Pre-trained fusion model checkpoints (from OpenCOOD) used when evaluating CoAdapt


results/	# Saved results of experiment runs

opv2v_data_dumping/	# OPV2V datasets

```

  

---

  

## Getting Started

  **Note**: Because CoAdapt uses requies computer vision models and deploys an LLM locally, a machine equipped with an NVIDIA GPU (>= 16 GB VRAM recommended) is required. 
  For evaluation purposes, we provide the results of the LLM selection and data fusion phases.


  CoAdapt uses the [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/) collaborative perception dataset. For this purpose, please download the dataset from the OPV2V website. Because the OPV2V dataset is large and for the purpose of evaluating CoAdapt, we suggest download the split called `test_culver_city`. Place it in the folder called `opv2v_data_dumping`, unzip it, and rename it to `test`. For reproducing the full results in the CoAdapt paper, it's required to download the full `train` and `test` splits.


## Installation

	Start by cloning this repository:

  ```bash
	$ git clone https://github.com/houssamhh/coadapt
	```

	We provide a Docker image for running CoAdapt. Therefore, Docker needs to be installed on your host machine. After installing Docker, you can build the image using the following command:

  ```bash
	$ docker build -t coadapt .
  ```


## Using CoAdapt

  
  To run CoAdapt, run and enter the container using the following command:
  ```bash
  $ docker run --rm -it
  -v "${PWD}/results:/workspace/results"
  --entrypoint bash
  coadapt
  ```

  This will mount the `results` directory so that you can see the figures generated.

  If your machine is equipped with an NVIDIA GPU, run the following command so that the container can you use the GPU of the host machine:
  ```bash
  $ docker run --rm -it
  -v "${PWD}/results:/workspace/results"
  --gpus all\
  --entrypoint bash
  coadapt
  ```

  Because OpenCOOD uses Python 3.7 which doesn't support modern LLM libraries and frameworks, we provide two virtual environments: one for running LLM inference, and one for evaluating collaborative perception models in OpenCOOD.

  CoAdapt is built on top of the OpenCOOD library. For this purpose, start first by cloning the OpenCOOD repository:
  ```bash
  $ git clone https://github.com/DerrickXuNu/OpenCOOD
  ```
  Then, move the OpenCOOD core components to the current directory:
  ```bash
  $ mv OpenCOOD/opencood .
  ```

  ### Phase 1: LLM Inference

  To run LLM inference (robot and data fusion strategy selection), activate the virtual environment first:
  ```bash
  $ source /opt/conda/etc/profile.d/conda.sh
  $ conda activate decision_making
  ```
  
  **Note**: To deploy the LLM locally, you need a HuggingFace token. Then, you can save the token as an environment variable called `HF_TOKEN` (e.g., by using `export HF_TOKEN=<your_token>` in Linux environments). Alternatively, we support using Anthropic cloud-based models. For this purpose, you need an Anthropic API Key, which you can then save in an environment variable called `ANTHROPIC_API_KEY`.
  To use a local LLM, e.g., Google's Gemma 3 model, run the following command:
  ```bash
  $ python coadapt/run_pipeline.py --dataset_root opv2v_data_dumping/test --llm_model google/gemma-3-4b-it --output_dir results/artifact_evaluation --reselect_every 5
  ```

 This will run the pipeline on the collaborative perception scenarios found under `opv2v_data_dumping/test`, using the Gemma 3 4B LLM, and save the selection results to the directory `results/artifact_evaluation`. The LLM will choose which robots participate in the fusion process and the fusion strategy every 5 frames (you can modify this variable to select the frequency that you'd like to set). 
 Alternatively, to use Anthropic models, you can run the same command, but by specifying the Claude model that you'd like to use:
 ```bash
 $ python coadapt/run_pipeline.py --dataset_root opv2v_data_dumping/test --llm_model claude-sonnet-4-6 --output_dir results/artifact_evaluation --reselect_every 5
 ```
The results will be saved in a `selection.csv` file, which will contain a list of robots that participate in the fusion process (other than the ego robot), and the data fusion strategy.

### Phase 2: Running Collaborative Perception Algorithms


Activate the OpenCOOD virtual environment:

```bash
$ conda deactivate
$ conda activate opencood
```
Then, install OpenCOOD utils:
```bash
$ cython opencood/utils/box_overlaps.pyx
```
```bash
$ gcc -shared -fPIC -O2 -I$(python -c "import numpy; print(numpy.get_include())") -I$(python -c "import sysconfig; print(sysconfig.get_path('include'))") opencood/utils/box_overlaps.c -o opencood/utils/box_overlaps.cpython-37m-x86_64-linux-gnu.so
```
```bash
$ export PYTHONPATH=$(pwd):$PYTHONPATH
```
You can then run the collaborative perception algorithms, based on the LLM's selection:
```bash
$ python coadapt/run_pipeline.py --dataset_root opv2v_data_dumping/test --skip_selection --selection_csv results/acsos2026/gemma4/opv2v_test/selection.csv  --model_intermediate trained-models/Models/voxelnet_attentive_fusion/ --model_early trained-models/Models/pointpillar_early_fusion --model_late trained-models/Models/pointpillar_late_fusion --output_dir results/artifact_evaluation/inference_results
```

Note that the models used in CoAdapt can be found in the OpenCOOD library. We include only the pre-trained models used in the paper experiments under `trained-models/`. Additional pre-trained models can be tested by placing them in the `trained-models` directory and specifying their path after the --model_early/--model_intermediate/--model_late options in the previous command.

This will generate an `evaluation.csv` file containing performance metrics related to precision and communication cost for each control cycle (in this case, 5 frames / control cycle).

## Generating Plots

The `evaluation` directory contains the scripts for generating plots found in the paper.

All results and plots used in the paper can be found under the `results/acsos2026` directory.

To generate plots showing the number of robots that participate in the fusion process for each LLM model (Figure 4) in the paper, run the following command:
```bash
$ python evaluation/plot_all_cav_selections.py
```
This will generate the results in the `results/artifact_evaluation` directory. One file per model will be generate, named `cav_selection_<model_name>.pdf`

To generate plots showing the spatial positions of CoAdapt-selected robots (Figure 5), run the following command:
```bash
$ python evaluation/plot_all_bev_positions.py
```

Note that this requires that the `train` and `test` splits of the OPV2V datasets to be downloaded and place under the `opv2v_data_dumping` directory.

Alternatively, you can generate a figure for a single frame by running the following command:

```bash
$ python evaluation/plot_bev_positions.py \
    --dataset_root opv2v_data_dumping/test \
    --selection_csv results/acsos2026/gemma4/opv2v_test/selection.csv \
    --scenario 2021_09_03_09_32_17 \
    --frame 006220 \
    --output results/artifact_evaluation/bev_example.pdf
```

This will generate the results in the `results/artifact_evaluation/<model_name>/bev_positions` directory, with figures showing the positions of robots per selection frame.

To generate plots showing the LLM-selected fusion strategy vs. bandwidth (Figure 6), run the following command
```bash
$ python evaluation/plot_all_strategy_selections.py
```

This will generate the results in the `results/artifact_evaluation/` directory. One file per model will be generated, named `fusion_strategy_<model_name>.pdf`

To generate the plot showing the average communication cost / frame vs. average precision (Figure 7), run the following command:
```bash
$ python evaluation/plot_approach_comparison.py \
    --results_root results/acsos2026 \
    --splits opv2v_train opv2v_test \
    --output results/artifact_evaluation/approach_comparison.png
```
This will generate the results in the `results/artifact_evaluation` directory.

Finally, to evaluate the communication cost / frame / fusion strategy vs. precision (Figure 8), run the following command:
```bash
$ python evaluation/plot_comm_cost.py \
    --eval_csv     results/acsos2026/<model_name>/opv2v_train/eval.csv \
                   results/acsos2026/<model_name>/opv2v_test/eval.csv \
    --baseline_csv results/acsos2026/<model_name>/opv2v_train/eval_baseline.csv \
                   results/acsos2026/<model_name>/opv2v_test/eval_baseline.csv \
    --output       results/artifact_evaluation/comm_cost_<model_name>.png
```
Replace `<model_name>` with the LLM that you'd like to evaluate (one of: [gemma4, gpt-oss-20b, gpt-oss-120b, llama3.3]). 

For example, to evaluate Gemma 4, run the following command:
```bash
python evaluation/plot_comm_cost.py \
    --eval_csv     results/acsos2026/gemma4/opv2v_train/eval.csv \
                   results/acsos2026/gemma4/opv2v_test/eval.csv \
    --baseline_csv results/acsos2026/gemma4/opv2v_train/eval_baseline.csv \
                   results/acsos2026/gemma4/opv2v_test/eval_baseline.csv \
    --output       results/artifact_evaluation/comm_cost_gemma4.png
```