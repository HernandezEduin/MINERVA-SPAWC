# MINERVA-InfoCost

Official implementation of the paper:

**"The Information Cost of Multi-Hop Reasoning: Distributed Graph Navigation under Communication Constraints"**

This repository builds on MINERVA and provides the scripts, configs, and pretrained assets used for evaluation and policy-entropy analysis.

## Dependencies

Initialize the MINERVA submodule:

```bash
git submodule update --init --recursive
```

Install the MINERVA requirements before running anything in this repository:

```bash
cd minerva
pip install -r requirements.txt
cd ..
```

## Data and Pretrained Assets

Download the `.cache`, `datasets`, and `saved_models` directories from:

https://storage.googleapis.com/halcyon_data/multihop_ds/conferences/spawc/minerva_spawc.zip

Extract the archive in the repository root so the directories appear at the top level.

## Run

Use a dataset config to run evaluation and policy-entropy analysis:

```bash
bash run_infocost.sh configs/<dataset>.yaml
```

Example:

```bash
bash run_infocost.sh configs/kinshiphinton.yaml
```

Available configs:

- `configs/kinshiphinton.yaml`
- `configs/metaqa.yaml`
- `configs/mquake_st_single.yaml`
- `configs/mquake_st_multi.yaml`

To evaluate existing checkpoints under greedy and hard per-hop Top-K action budgets
without retraining, run:

```bash
bash run_rate_sweep.sh configs/mquake_st_single.yaml 0 \
  --rate_top_k 1 2 4 8 16 32 64 128 \
  --rate_include_unrestricted true \
  --rate_seed 42
```

Pass an empty GPU argument (for example `""`) to force CPU execution. Rate-sweep
outputs are written to the new timestamped run's `rate_sweep/` directory as a common
CSV/JSON table plus budget-performance plots. The task-agnostic coder uses only the
active graph action representation; it never fits evaluation action choices or labels.

Outputs are written under `./saved_models/<dataset>/<run_name>/`:

- Entropy analysis artifacts: `./saved_models/<dataset>/<run_name>/policy_entropy/`
- Evaluation results: `./saved_models/<dataset>/<run_name>/scores.txt`
- Individual question answers: `./saved_models/<dataset>/<run_name>/test_beam/test_paths.txt`

## Folder Structure

The repository is organized as follows:
```
minerva_infocost/
├── .cache/                 # Cached files for processed datasets
├── code/
│   ├── evaluation_infocost.py # Evaluation entrypoint for InfoCost experiments
│   ├── evaluation_rate_sweep.py # Greedy/Top-K evaluation from existing checkpoints
│   └── policy_entropy/
│       ├── eval.py         # Policy-entropy evaluation pipeline and summaries
│       ├── metrics.py      # Core entropy/surprisal and identifier-bit utilities
│       ├── artifacts.py    # Save/load helpers for policy-entropy artifacts
│       ├── plotting.py     # Plot generation for policy-entropy analysis
│       ├── rate_constraints.py # Pure NumPy action/coding helpers
│       ├── rate_eval.py    # MINERVA-compatible rate/performance evaluation
│       └── rate_plotting.py # Rate-sweep artifacts and plots
├── configs/                # YAML config files for different datasets
├── datasets/               # Datasets for evaluation (preprocessed and ready to use)
├── minerva/                # MINERVA codebase (submodule)
├── run_infocost.sh            # Convenience script to launch configured runs
├── run_rate_sweep.sh           # Convenience script for rate-sweep evaluation
└── saved_models/            # Pretrained models and evaluation outputs
    ├── kinshiphinton/      # Outputs for the Kinship-Hinton dataset
    ├── metaqa/             # Outputs for the MetaQA dataset
    └── mquake/             # Outputs for the MQuAKE dataset

```

## Citation

If you use this code for your research, please cite our paper:

```
@inproceedings{minerva_infocost_2026,
  title={The Information Cost of Multi-Hop Reasoning: Distributed Graph Navigation under Communication Constraints},
  author={Hernandez, Eduin E and Garcia, Luis F, and Askar, Nurassyl, and Rini, Stefano},
  year={2026},
  url={https://github.com/HalcyonSolutions/MINERVA-InfoCost}
}
