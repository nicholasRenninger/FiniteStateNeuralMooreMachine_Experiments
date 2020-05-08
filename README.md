# NeuralMooreMachine_Experiments
A project experimenting with Extracting Moore Machines from Recurrent Sequence Models, based heavily on reproducing and extending the experiments in the paper ["Learning Finite State Representations of Recurrent Policy Networks"](https://arxiv.org/abs/1811.12530) ([repo](https://github.com/koulanurag/mmn)).

Currently, the container-based environment has been tested to work on both Ubuntu (GPU / CPU) and macOS (CPU-only) hosts. For GPU support, you will need a compiant NVIDIA GPU. See the [Installation Section](https://github.com/nicholasRenninger/NeuralMooreMachine_Experiments/blob/master/README.md#installation) for more details.

**Table of Contents**
* [About](https://github.com/nicholasRenninger/NeuralMooreMachine_Experiments/blob/master/README.md#about)
* [Results](https://github.com/nicholasRenninger/NeuralMooreMachine_Experiments/blob/master/README.md#methodology)
* [Methodology](https://github.com/nicholasRenninger/NeuralMooreMachine_Experiments/blob/master/README.md#methodology)
* [Container Usage](https://github.com/nicholasRenninger/NeuralMooreMachine_Experiments/blob/master/README.md#container-usage)
* [Installation](https://github.com/nicholasRenninger/NeuralMooreMachine_Experiments/blob/master/README.md#installation)


## About

This repo contains the docker container and python code to fully experiment with [mnn](https://github.com/koulanurag/mmn). The whole experiment is contained in `MNN_testing.ipynb`.

This project is based on [stable-baselines](https://stable-baselines.readthedocs.io/), [OpenAI Gym](https://github.com/openai/gym), [MiniGym](https://github.com/maximecb/gym-minigrid), [tensorflow](https://www.tensorflow.org/), and [wombats](https://github.com/nicholasRenninger/wombats)



## Results

See [my paper](https://github.com/nicholasRenninger/CSCI_5922_Final_Project_Report/blob/511a71f2eea62637146e52432be817ac258fdb47/CSCI_5922_Final_Project_Report.pdf) or [the notebook](https://github.com/nicholasRenninger/NeuralMooreMachine_Experiments/blob/master/MMNN_testing.ipynb) for more of the training results and hyperparameter choices.


### Final Policies

#### CNN-LSTM Policy
This agent observes pixels from the game and outputs an action at each timestep. The agents transforms pixels to LSTM hidden state using the [Original Atari CNN Architecture](https://www.nature.com/articles/nature14236) and then the LSTM outputs its updated hidden state to the actor-critic architecture, where the action distribution and value function are both estimated by the final network layers. These networks are trained using the [PPO2](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html) actor-crtic RL algorithm, chosen here because of its performance, ease of hyperparameter tuning, and easy parallelizability. Below is a gif of the trained CNN-LSTM agent in a single pong environment:

<img src="https://github.com/nicholasRenninger/NeuralMooreMachine_Experiments/blob/master/media/basepolicy_ppo2_PongNoFrameskip-v4_single-step-0-to-step-1000.gif">

#### Moore Machine Network (MMN) Policy
This agent again observes pixels from the game and outputs an action at each timestep. However, there are now two quantized bottleneck networks (QBNs) placed after the CNN feature extractor and after the LSTM state output. These QBNs are quantized autoencoders, where the latent state of each autoencoder network has neurons that are quantized to have activated values of either -1, 0, or 1. This means that the entire policy network - called a moore machine network (MMN) - is now technically a finite state machine, specifically a [moore machine](https://en.wikipedia.org/wiki/Moore_machine), that uses the CNN as its discrete observation function and the LSTM as the resultant state transition function. The two QBNs are each trained separately until their reconstruction loss is quite low, and then they are inserted into the original CNN-LSTM network as described above to form the final MMN. Below is a gif of the trained MMN agent in a single pong environment:

<img src="https://github.com/nicholasRenninger/NeuralMooreMachine_Experiments/blob/master/media/mmn_ppo2_PongNoFrameskip-v4_single-step-0-to-step-1000.gif">

### Evaluation

Below is a table showing the mean non-discounted reward for each agent over 10 monte-carlo rollouts:

|     Original CNN-LSTM Agent     |    MMN Agent   |
|:-------------------------------:|:--------------:|
|            20.3 ± 0.2.          |  18.90 ± 1.14  |

Thus, the MMN seems to have pretty comparable performance to the original policy, despite it now being represented by a finite state machine. However, looking at the agents, we can see that the MMN certainly looks to be less "smooth" overall, something we expect given the compressed, finite observation and state space of the MMN policy. No fine-tuning of the MMN policy network was implemented, so the MMN could certainly be improved by some more training in the environment.


## Methodology

Here is a high-level overview of the steps taken in the learning of moore machine network (MMN) controller:

1. Learn an feature_extractor-rnn_policy for a RL environment using a standard RL algorithm capable of learning with a recurrent policy (e.g. [ACKTR](https://openai.com/blog/baselines-acktr-a2c/) or [PPO2](https://openai.com/blog/openai-baselines-ppo/)). Here the feature extraction network is known as `F_ExtractNet` and the RNN policy that takes these features and produces the next action is known as `RNN_Policy`. *If your environment already has simple, discrete observations, you will not need `F_ExtractNet` and can directly feed the observation into the `RNN_Policy`.*

2. Generate "Bottleneck Data". This is where you simulate many trajectories in the RL environment, recording the observations and the actions taken by the `RNN_Policy`. This is for training the "quantized bottleneck neural networks" (`QBNs`) next.

3. Learn `QBNs`, which are essentially applied autoencoders (AE), to quantize (discretize):

    * the observations of the environmental feature extractor:
        * CNN if using an agent that observes video of the environment. 
        * MLP if getting non-image state observations
    This is called `b_f` in the paper and `OX` in the mnn code.
    
    * the hidden state of the `RNN_Policy`. This is called `b_h` in the paper and `BHX` in the mnn code

This is done by 

4. Insert the trained `OX` QBN *before* the feature extractor and the trained `BHX` QBN *after* the RNN unit in the feature_extractor-rnn_policy network to create what is now called the moore machine network (`MMN`) policy.

5. Fine-tune the `MMN` policy by re-running the rl algorithm using the `MMN` policy as a starting point for RL interactions. *Importantly, for training stability the `MMN` is fine-tuned to match the softmax action distribution of the original `RNN_Policy`, not the argmax -> optimize with a categorical cross-entropy loss between the RNN and `MMN` output softmax layers*. 

6. Extract a classical moore machine from the `MMN` policy by doing:

    1. Generate trajectories in the RL environment using rollout simulations of `MMN` policy. For each rollout simulation timestep, we extract a tuple `(h_{MMN, t-1}, f_{MMN, t}, h_{MMN, t}, a_{MMN, t})`:
        * `h_{MMN, t-1}`: the quantized hidden state of the RNN QBN at the previous timestep
        * `f_{MMN, t}`: the quantized observation state of the feature extractor QBN at the current timestep.
        * `h_{MMN, t}`: the quantized hidden state of the RNN QBN at the current timestep.
        * `a_{MMN, t}`: the action outputted by the MNN policy at the current timestep.
    
    2. As you can see, we now have *most* of the elements needed to form a Moore machine:
        * `h_{MMN, t-1}` -> prior state of the moore machine, `h_{MM, t-1}`
        * `f_{MMN, t}` -> input transition label of the transition from moore machine state `h_{MM, t-1}` to moore machine state `h_{MM, t}`, `o{MM, t}`.
        * `h_{MMN, t}` -> current state of the moore machine, `h_{MM, t}`.
        * `a_{MMN, t}` -> output label of the current moore machine state `h_{MM, t}`, `a_{MM, t}`.
    
    3. What we are missing is a transition function `delta()` and an initial state of the moore machine, `h_{MM, 0}`. 
     
        * `delta()`: A moore machine needs a transition function `delta(h_{MM, t - 1}, o_{MM, t}) -> h_{MM, t}` that maps the current state and observed feature to the next state. Here we will end up with a set of trajectories containing `p` distinct quantized states (`h_{MM}`) and `q` distinct quantized features (`o_{MM}`). These trajectories are then converted to a transition table representing `delta`, which maps any observation-state tuple `(h_{MM}, o_{MM})` to a new state `h_{MM}'`.

        * `h_{MM, 0}`: In practice, this is done by encoding the start state of `RNN_Policy` using `BHX`: `h_{MM, 0} = BHX(h_{`MMN`, 0}`.

7. Minimize the extracted moore machine to get the smallest possible model. "In general, the number of states `p` will be larger than necessary in the sense that there is a much smaller, but equivalent, minimal machine". Thus, use age old moore machine minimization techniques to learn the moore machine. **This process is exactly the process in Grammatical Inference, thus we can use my own [wombats](https://github.com/nicholasRenninger/wombats/tree/master) tool.**

8. You're done. You now have a moore machine that operated on the abstract, quantized data obtained from the `QBNs`. To use the moore machine in an environment:

    1. Start by using `OX` and the feature extractor to take the initial environmental observation `f_{env, 0}` and get the moore machine feature observation `o_{MM, 0} = OX.encode(F_ExtractNet(f_{env, 0}))`.

    2. Use `delta` with `o_{MM, 0}` and `h_{MM, 0}` (part of the definition of the moore machine) to get the action, `delta(o_{MM, 0}, h_{MM, 0}) = a_{MM, 0}`.

    3. Take a step in the environment using `step(env, a_{MM, 0)` to produce a new observation `f_{env, 1}` and the environmental reward, `r_t`.
    
    4.  As in step 1-3, we do for `t = 1` onwards:
        1.  `o_{MM, t} = OX.encode(F_ExtractNet(f_{env, t}))`
        2.  `a_{MM, t} = delta(o_{MM, t}, h_{MM, t})`
        3.  `f_{env, t+1}, r_t = step(env, a_{MM, t})`


## Container Usage

* **run with a GPU-enabled image and start a jupyter notebook server with default network settings:**
  
  ```bash
  ./docker_scripts/run_docker.sh --device=gpu
  ```

* **run with a CPU-only image and start a jupyter notebook server with default network settings:**
  
  ```bash
  ./docker_scripts/run_docker.sh --device=cpu
  ```
  
* run with a GPU-enabled image with the jupyter notebook served over a desired host port, in this example, port 8008, with tensorboard configured to run on port 6996. You might do this if you have other services on your host machine running over `localhost:8888` and/or `localhost:6666`:
  
   ```bash
   ./docker_scripts/run_docker.sh --device=gpu --jupyterport=8008 --tensorboardport=6996
   ```

* run with a GPU-enabled image and drop into the terminal:
  
  ```bash
  ./docker_scripts/run_docker.sh --device=gpu bash
  ```

* run a bash command in a CPU-only image interactively:
  
  ```bash
  ./docker_scripts/run_docker.sh --device=cpu $OPTIONAL_BASH_COMMAND_FOR_INTERACTIVE_MODE
  ```

* run a bash command in a GPU-enabled image interactively:
  
  ```bash
  ./docker_scripts/run_docker.sh --device=gpu $OPTIONAL_BASH_COMMAND_FOR_INTERACTIVE_MODE
  ```

---

### Accessing the Jupyter and Tensorboard Servers

**To access the jupyter notebook:**
make sure you can access port 8008 on the host machine and then modify the generated jupyter url:

```bash
http://localhost:8888/?token=TOKEN_STRING
```

with the new, desired port number:

```bash
http://localhost:8008/?token=TOKEN_STRING
```

and paste this url into the host machine's browser. 

**To access tensorboard:**
make sure you can access port 6996 on the host machine and then modify the generated tensorboard  url:

(e.g. TensorBoard 1.15.0) 
```bash
http://0.0.0.0:6006/
```

with the new, desired port number:
```bash
http://localhost:6996
```

and paste this url into the host machine's browser. 


## Installation

This repo houses a docker container with `jupyter` and `tensorbaord` services running. If you have a NVIDIA GPU, check [here](https://developer.nvidia.com/cuda-gpus#compute) to see if your GPU can support CUDA. If so, then you can use the GPU-only instruction below.

### Install Docker and Pre-requisties

Follow steps one (and two if you have a CUDA-enabled GPU) from [this guide](https://www.tensorflow.org/install/docker) from tensorflow to prepare your computer for the tensorflow docker base container images. **Don't** actually install the tensorflow container, that will happen automatically later.

### Post-installation 

Follow the *nix [docker post-installation guide](https://docs.docker.com/engine/install/linux-postinstall/).

### Building the Container

Now that you have docker configured, you can need to clone this repo. Pick your favorite directory on your computer (mine is `/$HOME/Downloads` ofc) and run:
 ```bash
git clone --recurse-submodules https://github.com/nicholasRenninger/NeuralMooreMachine_Experiments
cd NeuralMooreMachine_Experiments
 ```
 
 The container builder uses `make`:
 * If you **have a CUDA-enabled GPU** and thus you followed step 2 of the docker install section above, then run:
 ```bash
make docker-gpu
```

* If you **don't have a CUDA-enabled GPU** and thus you **didn't** follow step 2 of the docker install section above, then run:
 ```bash
make docker-cpu
```
