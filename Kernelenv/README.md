Training:
```py
TRITON=1 python ppo-lstm-pytorch.py --path /absolute/path/to/trial/folder/
```

Inference on resnet:
```py
TRITON=1 python ppo-lstm-pytorch.py --path /absolute/path/to/trial/folder/ --infer
```

Pytorch/tinygrad version

##############  Main idea:
Why do you want beam? for deep optimal solution. [Timeframe: long]
why do you want RL model free? for fast iterative development.[Timeframe: fast / medium]
Why do you want RL model base? for best optimal solution. [Timeframe: fast]
##############  

## Model-Free Reinforcement Learning: (fast training model)
- **General RL Based on Large Kernel**
  - **Inference**
    - Use the pre-trained general model directly to do inference.
  - **Fine-Tuning**
    - Fine-tune(re-train) the general model on a specific, smaller kernel.
    - **Inference**
      - Use the fine-tuned model to do inference on kernel it recognize.

- **RL Based on Small Kernel**
  - **Inference**
    - Train a model specifically on a small kernel and use it for inference.

- **Meta-Inference with RL and Beam Search**
  - **Reinforcement Learning (RL)**
    - **RL Inference Strategies**
      - Best Policy: Choose the action with the highest probability.
      - Best Value: Choose the action that maximizes the expected return.
      - Value Filtering: Eliminate bad moves based on low value estimates and use the remaining options.
      - Mixed Strategy: Combine Best Policy and Best Value to make a decision.
    - **Fallback**
      - Switch to Tensore_core/hand_optimizer to find a satisfactory solution.
    - **Optional**
      - Beam search if allow when RL fails to find a satisfactory solution.

##############

## Model-Based Reinforcement Learning: (long training model)
- **Optimal solution for cloud compute with tinyrack**
  - **Find optimal strategy in a general setting**
    - MCTS (MuZero-like)
    - Sequential (like model-free but with a representation layer)
  - **Representation layer based on: (aka simulator)**
    - Linear (partial embedding)
    - Diffusion model (deep simulation)
    - Transformer model (deep simulation)
- **Generate a local kernel set bahavior of the device with greedy search/ ucb search**
  - **Optimize learning on cloud with offline RL**
- **Transfer model to local user or keep an API call for web-beam**

##############

There is a possibility of mcts for fast training with optimal hyperparameter.
