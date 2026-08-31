"""
DQN Agent for WordleEnv.

Architecture: small MLP mapping the 183-dim state feature vector to
Q-values for all len(action_words) actions.

Key practical choices for this problem specifically:
- Action masking: at both action-selection and target-computation time, we
  mask out actions (words) already known to be inconsistent with the
  current state (green/present/absent constraints). This is standard
  practice for large-discrete-action DQN and is what makes ~2300 actions
  tractable to learn over - without it, most of training is wasted on
  actions that are trivially wrong given what's already known.
- Replay buffer + target network: standard DQN stabilizers.
- Epsilon-greedy exploration over the MASKED action set (so exploration
  doesn't waste steps on impossible guesses either).
"""
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple("Transition", ["obs", "action", "reward", "next_obs", "done", "next_mask"])


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, obs_dim, n_actions, device="cpu", lr=1e-3, gamma=0.99):
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma

        self.q_net = QNetwork(obs_dim, n_actions).to(device)
        self.target_net = QNetwork(obs_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

    def act(self, obs, mask, epsilon):
        """Epsilon-greedy over the masked action set."""
        valid_indices = np.flatnonzero(mask)
        if len(valid_indices) == 0:
            valid_indices = np.arange(self.n_actions)  # fallback safety

        if random.random() < epsilon:
            return int(random.choice(valid_indices))

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(obs_t).squeeze(0).cpu().numpy()

        masked_q = np.full(self.n_actions, -1e9, dtype=np.float32)
        masked_q[valid_indices] = q_values[valid_indices]
        return int(np.argmax(masked_q))

    def remember(self, obs, action, reward, next_obs, done, next_mask):
        self.buffer.push(obs, action, reward, next_obs, done, next_mask)

    def train_step(self, batch_size=128):
        if len(self.buffer) < batch_size:
            return None

        batch = self.buffer.sample(batch_size)
        obs = torch.as_tensor(np.array([t.obs for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.array([t.next_obs for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=self.device)
        next_masks = np.array([t.next_mask for t in batch])  # (B, n_actions) bool

        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_obs).cpu().numpy()
            next_q[~next_masks] = -1e9
            max_next_q = torch.as_tensor(next_q.max(axis=1), dtype=torch.float32, device=self.device)
            target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
