# baa
Make sure to install nbstripout
```bash
nbstripout --install --python python3 --attributes .gitattributes
```

Install dependencies with pip
```bash
pip install -r requirements_with_versions_lac2.txt
```

Best way to easily start exploring and running the code is to use devcontainers.

# Experiments
First of all make sure to add your Huggingface API key and the wandb API key to the environment variables. You can do this by creating a .env file in the root of the project and adding the following lines:
```bash
export HF_HOME='path_to_your_huggingface_folder'
export WANDB_API_KEY='your_wandb_api_key'
export HF_TOKEN='your_huggingface_api_key'
```

Then you can run the experiments with one of the following commands:
```bash
wandb sweep wandb-original-model-benchmarks.yaml
wandb sweep wandb-first-experiment-prod-config.yaml
wandb sweep wandb-second-experiment-prod-config.yaml
wandb sweep wandb-third-experiment-prod-config.yaml
```

# Results
All results are saved in the logs folder.

# Github Copilot
Github Copilot was used to generate the code for the experiments. The code is generated in the copilot folder.