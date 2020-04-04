import gym
import torch
import torch.nn as nn
import numpy as np

HIDDEN_SIZE = 128
DISCOUNT_FACTOR = 0.95
LEARNING_RATE = 0.01
ENTROPY_FACTOR_BETA = 0.01
BATCH_SIZE = 16
ENV_NAME = 'CartPole-v1'


class Net(nn.Module):

    def __init__(self, input_size, hidden_size, action_size):
        super(Net, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.seq(x)


class Episode:
    def __init__(self, discount_factor=0.99, scale_rewards=True):
        self.discount_factor = discount_factor
        self.scale_rewards = scale_rewards
        self.total_rewards = 0.0
        self.states = []
        self.actions = []
        self.rewards = []
        self.discounted_rewards = []
        self.scaled_discounted_rewards = []

    def add_step(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        if done:
            self.done()

    def done(self):
        self._calculate_discounted_rewards()
        self.total_rewards = sum(self.rewards)

    def _calculate_discounted_rewards(self):
        steps = len(self.states)
        self.discounted_rewards = np.zeros(steps)
        self.discounted_rewards[-1] = self.rewards[-1]
        for i in range(steps - 2, -1, -1):
            self.discounted_rewards[i] = self.rewards[i] + self.discount_factor * self.discounted_rewards[i + 1]
        if self.scale_rewards:
            self.scaled_discounted_rewards = \
                (self.discounted_rewards - self.discounted_rewards.mean()) / self.discounted_rewards.std()
        self.discounted_rewards = self.discounted_rewards.tolist()
        self.scaled_discounted_rewards = self.scaled_discounted_rewards.tolist()

    def __len__(self):
        return len(self.states)


class Batch:
    def __init__(self, size):
        self.size = size
        self.count = 0
        self.states = []
        self.actions = []
        self.discounted_rewards = []
        self.scaled_discounted_rewards = []
        self.total_rewards = []

    def append(self, episode):
        if self.count < self.size:
            self.count += 1
            self.states.extend(episode.states)
            self.actions.extend(episode.actions)
            self.discounted_rewards.extend(episode.discounted_rewards)
            self.scaled_discounted_rewards.extend(episode.scaled_discounted_rewards)
            self.total_rewards.append(episode.total_rewards)

    def full(self):
        return self.count == self.size

    def mean_rewards(self):
        return np.array(self.total_rewards).mean()


class Session:
    def __init__(self, model, env, batch_size, learning_rate=0.01, discount_factor=0.99, entropy_factor=0.01):
        self.env = env
        self.model = model
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.entropy_factor = entropy_factor
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, amsgrad=True)

    def train(self, target_reward):
        itr = 0
        for batch in self.batch_generator():
            self.optimizer.zero_grad()
            actions_logit = self.model(torch.FloatTensor(batch.states))
            log_prob_actions = torch.log_softmax(actions_logit, dim=1)
            policy_loss = self.policy_loss(log_prob_actions, batch)
            entropy_loss = self.entropy_loss(actions_logit, log_prob_actions)
            total_loss = policy_loss + entropy_loss
            total_loss.backward()
            self.optimizer.step()

            if batch.mean_rewards() > target_reward:
                self.save()
                print('\nSolved!')
                break
            itr += 1
            print(f'\r{itr} iterations, loss: {total_loss.item():.6f}, mean rewards: {batch.mean_rewards():.2f}',
                  end='')

    def batch_generator(self):
        state = self.env.reset()
        batch, episode = self.reset_generator_state()
        while True:
            action_values = self.model(torch.FloatTensor([state]))
            action_prob = nn.Softmax(dim=1)(action_values)
            action = np.random.choice(self.env.action_space.n, p=action_prob.data.numpy()[0])
            new_state, reward, done, _ = self.env.step(action)
            episode.add_step(state, action, reward, done)
            if done:
                if batch.full():
                    yield batch
                    batch, episode = self.reset_generator_state()

                batch.append(episode)
                episode = Episode(self.discount_factor, scale_rewards=True)
                new_state = self.env.reset()
            state = new_state

    def reset_generator_state(self):
        batch = Batch(self.batch_size)
        episode = Episode(self.discount_factor, scale_rewards=True)
        return batch, episode

    def policy_loss(self, log_prob_actions, batch):
        log_prob_executed_actions = torch.gather(log_prob_actions, dim=1,
                                                 index=torch.LongTensor(batch.actions).unsqueeze(-1)).squeeze()
        return -(torch.FloatTensor(batch.scaled_discounted_rewards) * log_prob_executed_actions).mean()

    def entropy_loss(self, actions_logit, log_prob_actions):
        actions_prob = torch.softmax(actions_logit, dim=1)
        entropy = - (actions_prob * log_prob_actions).sum(dim=1).mean()
        entropy_loss = -self.entropy_factor * entropy
        return entropy_loss

    def save(self, reward=0):
        torch.save(self.model.state_dict(), self.env.spec.id + f'_{reward}.dat')

    def play(self, model_state_file_path=None):
        env = gym.wrappers.Monitor(self.env, 'videos', video_callable=lambda episode_id: True, force=True)
        if model_state_file_path:
            stated_state = torch.load(model_state_file_path, map_location=lambda stg, _: stg)
            self.model.load_state_dict(stated_state)
        state = env.reset()
        total_reward = 0
        while True:
            env.render()
            state_v = torch.FloatTensor([state])
            actions_logit = self.model(state_v)
            actions_prob = torch.softmax(actions_logit, dim=1)
            action = np.random.choice(self.env.action_space.n, p=actions_prob.data.numpy()[0])
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        print("Total reward: %.2f" % total_reward)


env = gym.make(ENV_NAME)
model = Net(input_size=env.observation_space.shape[0],
            hidden_size=HIDDEN_SIZE,
            action_size=env.action_space.n)
session = Session(env=env,
                  model=model,
                  batch_size=BATCH_SIZE,
                  learning_rate=LEARNING_RATE,
                  discount_factor=DISCOUNT_FACTOR,
                  entropy_factor=ENTROPY_FACTOR_BETA
                  )
session.train(target_reward=199)
session.play()
