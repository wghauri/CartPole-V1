# CartPole-V1

## Description
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart

## Overview
The action takes values 0 and 1:
0 - Push cart to the left
1 - Push cart to the right

The observation corresponds to Cart Position,  Cart Velocity, Pole Angle, Pole Angular Velocity
The velocity that is of the cart depends on the angle the pole is pointing towards.

The episode terminates if the center of the cart reaches the edge of the display
The episode terminates if the pole angle is greater than ±12°

### Epsilon Decay
<img width="1438" alt="eps" src="https://github.com/wghauri/CartPole-V1/assets/88692517/c639e5c1-a1c3-41b9-842d-36c663b613ee">

### Average Rewards
<img width="1438" alt="avg_rewards" src="https://github.com/wghauri/CartPole-V1/assets/88692517/d6201621-2d0b-4d99-a750-e4bf4af72168">

### Loss
<img width="1438" alt="loss" src="https://github.com/wghauri/CartPole-V1/assets/88692517/c0fefc39-34eb-4c24-8e30-0744e845d8b5">

### Short video of agent at step ~1.77 million
https://github.com/wghauri/CartPole-V1/assets/88692517/cc5dc60e-1d56-4f42-8ac2-84812220046c
