import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter

HIDDEN_SIZE = 128
BATCh_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.model(x)


class Episode:
    def __init__(self):
        self.total_reward = 0
        self.observations = []
        self.actions = []


def episode_batch_generator(env, model, batch_size):
    batch = []
    obs = env.reset()
    episode = Episode()
    softmax = nn.Softmax()
    while True:
        actions_val = model(torch.FloatTensor([obs]))
        actions_prob = softmax(actions_val)
        actions_prob = actions_prob.data.numpy()[0]
        action = np.random.choice(env.action_space.n, p=actions_prob)
        obs_next, reward, done, _ = env.step(action)
        episode.total_reward += reward
        episode.actions.append(action)
        episode.observations.append(obs)
        if done:
            if len(batch) == batch_size:
                yield batch
                batch = []
            batch.append(episode)
            episode = Episode()
            obs_next = env.reset()

        obs = obs_next


def filter_elite_episodes(batch, min_reward_percentice):
    actions, observations = [], []
    rewards = list(map(lambda e: e.total_reward, batch))
    reward_boundary = np.percentile(rewards, min_reward_percentice)
    observations, action_spaces = [], []
    for episode in batch:
        if episode.total_reward < reward_boundary:
            continue

        observations.extend(episode.observations)
        actions.extend(episode.actions)
    mean_rewards = np.mean(rewards)
    return torch.FloatTensor(observations), torch.LongTensor(actions), mean_rewards


def train():
    env = gym.make('CartPole-v1')
    model = Net(env.observation_space.shape[0], HIDDEN_SIZE, env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=0.01, amsgrad=True)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter()
    for itr, batch in enumerate(episode_batch_generator(env, model, BATCh_SIZE)):
        observations, actions, mean_rewards = filter_elite_episodes(batch, PERCENTILE)
        optimizer.zero_grad()
        actions_val = model(observations)
        loss = loss_fn(actions_val, actions)
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss', loss.item(), itr)
        writer.add_scalar('reward', mean_rewards.item(), itr)
        print(f'\rItr:{itr}, loss :{loss}, mean reward: {mean_rewards}', end='')
        if mean_rewards > 199:
            print('\nSolved')
            break
    writer.close()


if __name__ == '__main__':
    train()
