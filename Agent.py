import gym 
import ipdb
import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from tqdm import tqdm
from typing import Any
from random import sample,random
from dataclasses import dataclass
from warnings import filterwarnings

@dataclass
class Sarst:
    # sarst - state, action, reward, next state
    state: Any
    action: int
    reward: float
    next_state: Any
    terminated: bool
    
class Model(nn.Module):
    def __init__(self, obs_shape, num_actions, lr):
        super().__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(obs_shape[0],256),
            nn.ReLU(),
            nn.Linear(256,num_actions)
        )

        self.optimizer = torch.optim.Adam(params=self.net.parameters(),lr=lr)

    def forward(self,x):
        return self.net(x)

class ReplayBuffer:
    # A replay buffer is a list of sarst

    def __init__(self, buffer_size=100000):
        self.buffer_size = buffer_size
        self.buffer = [None]*buffer_size
        self.idx = 0

    def insert(self, sarst):
        self.buffer[self.idx % self.buffer_size] = sarst
        self.idx += 1

    def sample(self, num_samples):
        assert num_samples < min(self.idx, self.buffer_size)
        if self.idx < self.buffer_size:
            return sample(self.buffer[:self.idx], num_samples)
        return sample(self.buffer, num_samples)
            
    
def update_tgt_model(m, tgt):
    tgt.load_state_dict(m.state_dict())

def train_step(model, tgt, state_transitions, num_actions, gamma=0.99):
    cur_states = torch.stack([torch.Tensor(s.state) for s in state_transitions])
    rewards = torch.stack([torch.Tensor([s.reward]) for s in state_transitions])
    mask = torch.stack([torch.Tensor([0]) if s.terminated else torch.Tensor([1]) for s in state_transitions])
    next_states = torch.stack([torch.Tensor(s.next_state) for s in state_transitions])
    actions = [s.action for s in state_transitions]

    with torch.no_grad():
        qvals_next = torch.max(torch.Tensor(tgt(next_states)),-1)[0] # output shape N and qvals: (N, num_actions)

    model.optimizer.zero_grad()

    qvals = model(cur_states) # (N, num_actions)

    one_hot_actions = f.one_hot(torch.LongTensor(actions), num_actions)

    loss = torch.mean((rewards + mask[:,0]*qvals_next * gamma - torch.sum(qvals*one_hot_actions, -1))**2)
    # qvals_next needs to be 0 when game done so mult by mask
    # mask shape is [1000] change mask to mask[:,0] to 
    loss.backward()
    model.optimizer.step()

    return loss

def main(test=False, chkpt=None):
    # Ignoring a harmless warning
    filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')
    if not test:
        wandb.init(project="dqn-project", name="dqn-cartpole")

    min_rb_size = 20000
    sample_size = 750
    lr = 0.001

    eps_decay = 0.999999

    env_steps_before_train = 100
    tgt_model_update = 500

    env = gym.make("CartPole-v1",render_mode="human")
    last_observation, last_info = env.reset()

    m = Model(obs_shape=env.observation_space.shape, num_actions=env.action_space.n, lr=lr)
    if chkpt is not None:
        m.load_state_dict(torch.load(chkpt))
    tgt = Model(obs_shape=env.observation_space.shape, num_actions=env.action_space.n, lr=lr)
    update_tgt_model(m, tgt)

    rb = ReplayBuffer()

    steps_since_train = 0
    epochs_since_tgt = 0

    step_num = -1 * min_rb_size

    episode_rewards = []
    rolling_reward = 0

    tq = tqdm()
    try:
        while True:
            if test:
                env.render()
                time.sleep(0.05)
            
            tq.update()

            eps = eps_decay**(step_num)
            if test:
                eps = 0

            if random() < eps:
                action = env.action_space.sample()
            else:
                action = torch.max(m(torch.Tensor(last_observation)),-1)[-1].item()

            # env.action_space.n to get num of actions (2)
            observation, reward, terminated, truncated, info = env.step(action)
            rolling_reward += reward

            reward = reward * 0.1

            rb.insert(Sarst(last_observation, action, reward, observation, terminated))

            last_observation = observation
            
            if terminated:
                episode_rewards.append(rolling_reward)
                if test:
                    print(rolling_reward)
                rolling_reward = 0
                observation, info = env.reset()

            steps_since_train += 1
            step_num += 1

            # Steps above are before training

            if (not test) and rb.idx > min_rb_size and steps_since_train > env_steps_before_train:
                # Every {env_steps_before_train} steps we train on {sample_size}
                loss = train_step(m, tgt, rb.sample(sample_size), env.action_space.n)
                wandb.log({'loss':loss.detach().cpu().item(), 'eps': eps, 'avg_reward': np.mean(episode_rewards)}, step=step_num)

                episode_rewards = []
                epochs_since_tgt += 1

                if epochs_since_tgt > tgt_model_update:
                    print("Updating target model")
                    update_tgt_model(m, tgt)
                    epochs_since_tgt = 0
                    torch.save(tgt.state_dict(), f"models/{step_num}.pth")

                steps_since_train = 0

    except KeyboardInterrupt:
        pass 

    env.close()

if __name__ == '__main__':
    main(True, "models/1770935.pth")