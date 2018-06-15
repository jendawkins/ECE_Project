# ECE_Project
# Continuous Model Free Episodic Control
Current RL algorithms take hundreds of thousands of interactions with the environment to learn. If a model happens upon a highly successful string of sequences, that string simply goes into an update rule; the model cannot necessarily directly repeat that episode because it has no episodic memory. Humans take a long time to learn many things, but can learn other things (like not to drink rotten milk, or to steer clear of our neighbor’s Ponzi scheme) from only one ‘episode’. Model Free Episodic Control by Blundell et al. is able to rapidly capitalize on successful episodes, but is poorly suited to continuous state spaces due to it being a tabular based method.

## Getting Started

### Prerequisites

Install: 

```
import gym
import numpy as np
from scipy.spatial.distance import cdist
```

### Files
* episodic_mods = all modifications added (Filter and NN)
* episodic_ctrl_clean.py = our code for original MFEC paper (KNN)
* neural_net.py = the neural network
* test_heuristic.py = code for analyzing the efficacy of filter.

## MFEC Paper
 Blundell, C., Uria, B., Pritzel, A., Li, Y., Ruderman, A., Leibo, J. Z., Rae, J., Wierstra, D., Hassabis, D. (2016) Model-free episodic control. arXiv preprint 1606.04460. Available at: https://arxiv.org/abs/1606.04460.


## Authors
Jennifer Dawkins, Jordan Prazak, Alice Yepremyan

## Acknowledgments

* Dr. Michael Yip and Ojash Neopane
