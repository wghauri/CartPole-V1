# CartPole-V1

## Description
For this project, I utilized Deep Q-Learning to train an agent in the CartPole gym environment. The agent was only provided with pixels as input. A replay buffer was used to store the agent’s experiences, and later replay them for training purposes. I implemented the epsilon greedy strategy to have the agent sufficiently explore the environment before it begins to exploit the environment.

As for the network used in this project, it takes the original state as input and outputs the q-values for each possible action. Comparing the original Q-value for the action taken by the agent in the experience to the optimal Q-value helps with training and calculating loss. 

## Overview
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart

The action takes values 0 and 1:                                                         
0 - Push cart to the left                                                                
1 - Push cart to the right

The observation corresponds to Cart Position, Cart Velocity, Pole Angle, and Pole Angular Velocity
The velocity of the cart depends on the angle the pole is pointing towards.

The episode terminates if the center of the cart reaches the edge of the display
The episode terminates if the pole angle is greater than ±12°

---
### Epsilon Decay
<img width="1438" alt="eps" src="https://github.com/wghauri/CartPole-V1/assets/88692517/c639e5c1-a1c3-41b9-842d-36c663b613ee">

### Average Rewards
<img width="1438" alt="avg_rewards" src="https://github.com/wghauri/CartPole-V1/assets/88692517/d6201621-2d0b-4d99-a750-e4bf4af72168">

### Loss
<img width="1438" alt="loss" src="https://github.com/wghauri/CartPole-V1/assets/88692517/c0fefc39-34eb-4c24-8e30-0744e845d8b5">

---
### Short video of agent at step ~1.77 million
https://github.com/wghauri/CartPole-V1/assets/88692517/cc5dc60e-1d56-4f42-8ac2-84812220046c
