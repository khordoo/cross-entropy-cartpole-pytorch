import torch
import torch.nn as nn
import numpy as np
import collections
import gym
from datetime import datetime
from tensorboardX import SummaryWriter

ENV_NAME = 'CartPole-v1'
NETWORK_HIDDEN_SIZE = 24
BATCH_SIZE = 200
EPSILON_INITIAL = 1
EPSILON_FINAL = 0.01
EPSILON_FINAL_STEP = 2000
REPLAY_BUFFER_CAPACITY = 2000
SYNC_NETWORKS_EVERY_STEP = 2000
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.01
DESIRED_TARGET_REWARD = 199
DEVICE = 'cpu'


class NetLinear(nn.Module):

    def __init__(self, observation_size, hidden_size, action_size):
        super(NetLinear, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.seq(x)


class EpsilonGreedy:
    def __init__(self, start_value, final_value, final_step):
        self.start_value = start_value
        self.final_value = final_value
        self.final_step = final_step

    def get(self, step):
        epsilon = 1 + step * (self.final_value - self.start_value) / self.final_step
        return max(self.final_value, epsilon)


class Episode:
    def __init__(self, discount_factor):
        self.discount_factor = discount_factor
        self.total_reward = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.discounted_rewards = []

    def add(self, state, action, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.total_reward += reward
        if done:
            self._calculate_discounted_rewards()

    def _calculate_discounted_rewards(self):
        self.discounted_rewards = np.zeros_like(self.rewards)
        steps = len(self.states)
        self.discounted_rewards[-1] = self.rewards[-1]
        for i in range(steps - 2, -1, -1):
            self.discounted_rewards[i] = self.rewards[i] + self.discount_factor * self.discounted_rewards[i + 1]


class ReplayBuffer:
    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.buffer = collections.deque(maxlen=capacity)

    def append(self, episode):
        self.buffer.append(episode)

    def sample(self, sample_size):
        indexes = np.random.choice(self.capacity, sample_size, replace=False)
        samples = [self.buffer[idx] for idx in indexes]
        return self._unpack(samples)

    def _unpack(self, samples):
        states, actions, rewards, discounted_rewards, dones, next_states = [], [], [], [], [], []
        total_rewards = []
        for episode in samples:
            states.extend(episode.states)
            actions.extend(episode.actions)
            rewards.extend(episode.rewards)
            discounted_rewards.extend(episode.discounted_rewards)
            dones.extend(episode.dones)
            next_states.extend(episode.next_states)
            total_rewards.append(episode.total_reward)

        states = torch.FloatTensor(np.array(states, copy=False)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states, copy=False)).to(self.device)
        actions = torch.LongTensor(np.array(actions, copy=False)).to(self.device)
        discounted_rewards = torch.FloatTensor(np.array(discounted_rewards, copy=False)).to(self.device)
        dones = torch.BoolTensor(np.array(dones, copy=False)).to(self.device)
        return states, actions, discounted_rewards, dones, next_states, np.array(total_rewards).mean()

    def __len__(self):
        return len(self.buffer)


class Session:
    def __init__(self, env, buffer, net, target_net, epsilon_tracker, device, batch_size, sync_every, discount_factor,
                 learning_rate):
        self.env = env
        self.buffer = buffer
        self.net = net
        self.target_net = target_net
        self.epsilon_tracker = epsilon_tracker
        self.device = device
        self.batch_size = batch_size
        self.sync_steps = sync_every
        self.discount_factor = discount_factor
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.writer = SummaryWriter(comment='' + datetime.now().isoformat(timespec='seconds'))

    def train(self, target_reward):
        step = 0

        while True:
            self.optimizer.zero_grad()
            epsilon = self.epsilon_tracker.get(step)
            episode = self.play_episode(epsilon)
            self.buffer.append(episode)
            if len(self.buffer) < self.buffer.capacity:
                continue

            states, actions, discounted_rewards, dones, next_states, mean_reward = self.buffer.sample(self.batch_size)
            predicted_state_values = self.calculate_state_value_prediction(states, actions)
            approximated_state_values_ballman = self.calculate_state_value_approximation(next_states, dones,
                                                                                         discounted_rewards)
            loss = nn.functional.mse_loss(predicted_state_values, approximated_state_values_ballman)
            loss.backward()
            self.optimizer.step()
            self.sync_target_network(step)

            if mean_reward > target_reward:
                print('Environment Solved!')
                break

            step += 1
            self.report_progress(step, loss.item(), mean_reward)

    @torch.no_grad()
    def play_episode(self, epsilon):
        state = self.env.reset()
        episode = Episode(self.discount_factor)
        while True:
            state_t = torch.FloatTensor(np.array([state], copy=False)).to(self.device)
            q_actions = self.net(state_t)
            action = torch.argmax(q_actions, dim=1).item()
            if np.random.random() < epsilon:
                action = np.random.choice(self.env.action_space.n)
            new_state, reward, done, _ = self.env.step(action)
            episode.add(state, action, reward, done, new_state)
            if done:
                return episode
            state = new_state

    def calculate_state_value_prediction(self, states, actions):
        q_val_all_actions = self.net(states)
        return q_val_all_actions.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def calculate_state_value_approximation(self, next_states, dones, discounted_rewards):
        q_val_next_state = self.target_net(next_states)
        _, q_val_next_state_Max = torch.max(q_val_next_state, dim=1)
        q_val_next_state_Max[dones] = 0
        expected_q_state = discounted_rewards + self.discount_factor * q_val_next_state_Max
        return expected_q_state.detach()

    @torch.no_grad()
    def sync_target_network(self, step):
        if step % self.sync_steps:
            self.target_net.load_state_dict(self.net.state_dict())

    def report_progress(self, step, loss, mean_reward):
        self.writer.add_scalar('Reward', mean_reward, step)
        self.writer.add_scalar('loss', loss, step)
        print(f'\r{step} : loss {loss} , reward: {mean_reward}', end='')


env = gym.make(ENV_NAME)
buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY, device=DEVICE)
net = NetLinear(env.observation_space.shape[0], NETWORK_HIDDEN_SIZE, env.action_space.n).to(DEVICE)
target_net = NetLinear(env.observation_space.shape[0], NETWORK_HIDDEN_SIZE, env.action_space.n).to(DEVICE)
epsilon_tracker = EpsilonGreedy(start_value=EPSILON_INITIAL, final_value=EPSILON_FINAL, final_step=EPSILON_FINAL_STEP)
session = Session(env=env, buffer=buffer, net=net, target_net=target_net, epsilon_tracker=epsilon_tracker,
                  device=DEVICE,
                  batch_size=BATCH_SIZE, sync_every=SYNC_NETWORKS_EVERY_STEP, discount_factor=DISCOUNT_FACTOR,
                  learning_rate=LEARNING_RATE)
session.train(target_reward=DESIRED_TARGET_REWARD)
