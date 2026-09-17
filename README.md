# Reinforcement learning - Final project

This repository contains the presentation and project for the Reinforcement Learning course at the University of Trieste. Code, examples, documentation and the final presentation were written by Marco Cartago and Marco Chiorri.

## Description

The project consists in an implementation of the tabular method `VAPOR` from the article [Probabilistic Inference in Reinforcement Learning
Done Right](https://arxiv.org/pdf/2311.13294), where the autors work on improving the formulation from the tutorial and review [Reinforcement learning and control as probabilistic inference: Tutorial and review.](https://arxiv.org/abs/1805.00909). A probabilistic interpretation of reinforcement learning. 

The project we developed is an implementation of a deep sea treasure variant, where the agent instead of traversing an empty gridworld travels trough a maze. The problem is used to showcase how suited are different learning algorithm to learn in an enviroment that inherently requires deep exploration in order to find the optimum.

With enough time it would be interesting to apply this same set of methods and approximate tecniques to chess variants with a sufficently small state space with the use of the [python bindings to fairy stockfish](https://pypi.org/project/pyffish/).

## A closer look at the enviroment

```text
███████████████████████████████████████████████████
███ ⋅ [o] ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  $ ███
███████████████ ⋅ ███ ⋅  ⋅ ███ ⋅ ███ ⋅ ███ ⋅  ⋅ ███
███ ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅ ███
███ ⋅ ███ ⋅  ⋅ ███ ⋅ ██████ ⋅ ███ ⋅ ███████████████
███ ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅ ███
█████████ ⋅  ⋅ ███ ⋅ ██████ ⋅  ⋅  ⋅ ███ ⋅  ⋅  ⋅ ███
███ ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅ ███
██████ ⋅ ███ ⋅ ███ ⋅  ⋅ █████████ ⋅ ███ ⋅  ⋅  ⋅ ███
███ ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅ ███
███ ⋅ ██████ ⋅ ███████████████ ⋅ ██████ ⋅  ⋅ ██████
███ ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅ ███
██████ ⋅ ██████ ⋅  ⋅ ██████ ⋅ ████████████ ⋅  ⋅ ███
███ ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅ ███
█████████████████████ ⋅ ████████████ ⋅ ██████ ⋅ ███
███ ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅ $$$███
███████████████████████████████████████████████████
```

In our setup the agent (`[o]`) navigates a bidimensional maze. In the lower right corner is placed a big reward `$$$` while in the uppper part of the labyrinth, in a spot that is always much easier to reach, is placed a smaller less valuable treasure ` $ `. Faliure to reach a goal will consist in a negative reward. This type of enviroment specifically evaluates the capability of a learning algorithm to perform deep exploration. A simpler version of this setup is used by the authors of VAPOR.

## Running the code

After activating the enviroment and installing the required packages (in `requirements.txt`) run:

 - `test_agent.py` For a full demo of an agent training (either a classic Qlearning agent or a VAPPOR like one).
 - `bandit_agents.py` Runs a demo for one of the algorithms on a simple Bandit-like problem.


 - `model_perf.py` Contains the code used for the simulations and the graphs in the project

Unfortunately we were not able to optimise sufficenty the memory usage: the `CVXPY` requires a lot of memory even for small board sizes, so keep the size <10.


## AI Usage

We used Claude Sonnet 5 and Gemma 4 31B for:
 - Finding bugs in the implementation of the enviroment
 - Looking for possible optimizations when performing convex optimization in the `Vapor` class.
 - Producing visualizations for the enviroment.