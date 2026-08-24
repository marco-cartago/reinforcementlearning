# Reinforcement learning - Final project

This repository contains the presentation and the project for the Reinforcement Learning course at the University of Trieste. Code, examples, documentation and the final presentation were written by Marco Cartago and Marco Chiorri.

## Description

The project consists in an implementation of the tabular method `VAPOR` from the article [Probabilistic Inference in Reinforcement Learning
Done Right](https://arxiv.org/pdf/2311.13294), where the autors work on improving the formulation from the tutorial and review [Reinforcement learning and control as probabilistic inference: Tutorial and review.](https://arxiv.org/abs/1805.00909). A probabilistic interpretation of reinforcement learning. 

The project we developed is an implementation of a deep sea treasure variant, where the agent instead of traversing an empty gridworld travels trough a maze. The problem is used to showcase how suited are different learning algorithm to learn in an enviroment that inherently requires deep exploration in order to find the optimum.

With enough time it would be interesting to apply this same set of methods and approximate tecniques to chess variants with a sufficently small state space with the use of the [python bindings to fairy stockfish](https://pypi.org/project/pyffish/).

## A closer look at the enviroment



## Running the code

After activating the enviroment and installing the required packages (in `requirements.txt`) run:

 - `main.py` For a demo of an agent in a gridworld-like labyrinth.
 - `test_agent.py` For a full demo of an agent training (either a classic Qlearning agent or a VAPPOR like one).