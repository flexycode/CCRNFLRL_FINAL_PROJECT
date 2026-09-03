"""
Embedding-based DQN Agent for WordleEnv (v2 — Revamped Architecture).

Instead of mapping state → Q(word_0), Q(word_1), ..., Q(word_N), this agent
scores (state, word_embedding) → scalar Q-value for any candidate word.
This decouples the network from a fixed word list, enabling generalization
to words never seen during training.

Architecture:
    State Encoder:  183 → 256 → 128  (obs features → state embedding)
    Word Encoder:   130 → 128         (word one-hot → word embedding)
    Scorer:         256 → 128 → 1     (concat → Q-value)

Key improvements over v1:
    - Word embedding scoring: can evaluate ANY word, not just indices
    - Double DQN: online net selects action, target net evaluates
    - Prioritized Experience Replay: focus on surprising transitions
    - Soft target updates (Polyak averaging) instead of hard copy
    - Batched candidate scoring: score all valid words in one forward pass
"""
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

WORD_FEAT_DIM = 130  # 5 positions × 26 letters

Transition = namedtuple("Transition",
    ["obs", "word_feat", "reward", "next_obs", "done", "next_valid_word_feats"])


# ---------------------------------------------------------------------------
# Network Architecture
# ---------------------------------------------------------------------------

class EmbeddingQNetwork(nn.Module):
    """Scores a (state, word) pair → scalar Q-value.
    
    The state encoder processes the 183-dim observation into a latent state.
    The word encoder processes the 130-dim word feature into a latent word.
    The scorer takes their concatenation and outputs a single Q-value.
    """
    def __init__(self, obs_dim=183, word_dim=WORD_FEAT_DIM, hidden=256):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.word_encoder = nn.Sequential(
            nn.Linear(word_dim, hidden // 2),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, obs, word_feats):
        """
        Args:
            obs: (B, 183) state observation
            word_feats: (B, N, 130) word feature matrix where N = number of candidate words
                    OR (B, 130) single word features
        Returns:
            q_values: (B, N) Q-values for each candidate OR (B, 1) for single word
        """
        state_emb = self.state_encoder(obs)  # (B, 128)
        
        if word_feats.dim() == 3:
            # Batched scoring: (B, N, 130) → (B, N, 128)
            B, N, D = word_feats.shape
            word_flat = word_feats.reshape(B * N, D)
            word_emb = self.word_encoder(word_flat).reshape(B, N, -1)  # (B, N, 128)
            
            # Expand state to match: (B, 128) → (B, N, 128)
            state_expanded = state_emb.unsqueeze(1).expand(-1, N, -1)
            
            # Concat and score: (B, N, 256) → (B, N, 1) → (B, N)
            combined = torch.cat([state_expanded, word_emb], dim=-1)
            q_values = self.scorer(combined).squeeze(-1)
            return q_values
        else:
            # Single word: (B, 130) → (B, 128)
            word_emb = self.word_encoder(word_feats)  # (B, 128)
            combined = torch.cat([state_emb, word_emb], dim=-1)  # (B, 256)
            return self.scorer(combined).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Prioritized Experience Replay
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """Simple proportional prioritized replay buffer.
    
    Uses TD-error as priority — transitions where the agent was most
    "surprised" get replayed more often, accelerating learning.
    """
    def __init__(self, capacity=200_000, alpha=0.6, beta_start=0.4, beta_frames=100_000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def push(self, *args):
        transition = Transition(*args)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        self.frame += 1
        N = len(self.buffer)
        
        # Compute sampling probabilities
        prios = self.priorities[:N] ** self.alpha
        probs = prios / (prios.sum() + 1e-8)
        
        indices = np.random.choice(N, size=batch_size, p=probs, replace=False)
        
        # Importance-sampling weights for bias correction
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        weights = (N * probs[indices]) ** (-beta)
        weights = weights / (weights.max() + 1e-8)
        
        batch = [self.buffer[i] for i in indices]
        return batch, indices, torch.as_tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, td_errors):
        for idx, td_err in zip(indices, td_errors):
            self.priorities[idx] = abs(td_err) + 1e-6
            self.max_priority = max(self.max_priority, self.priorities[idx])

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Uniform Replay Buffer (simpler fallback)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch, None, None

    def update_priorities(self, indices, td_errors):
        pass  # no-op for uniform buffer

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Agent (Embedding-based, Double DQN)
# ---------------------------------------------------------------------------

class DQNAgent:
    """Embedding-based Double DQN agent for Wordle.
    
    This agent can score ANY word given the current state, not just words
    from a fixed list. This enables generalization to unseen words.
    """
    def __init__(self, obs_dim=183, word_dim=WORD_FEAT_DIM, device="cpu",
                 lr=5e-4, gamma=0.99, tau=0.005, prioritized=True):
        self.device = device
        self.gamma = gamma
        self.tau = tau  # soft update coefficient
        self.word_dim = word_dim

        self.q_net = EmbeddingQNetwork(obs_dim, word_dim).to(device)
        self.target_net = EmbeddingQNetwork(obs_dim, word_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer() if prioritized else ReplayBuffer()

    def act(self, obs, valid_word_feats, epsilon):
        """Epsilon-greedy action selection over valid candidate words.
        
        Args:
            obs: (183,) numpy array — current state observation
            valid_word_feats: (N, 130) numpy array — features of valid candidate words
            epsilon: exploration rate
            
        Returns:
            idx: index into valid_word_feats of the chosen word
        """
        N = len(valid_word_feats)
        if N == 0:
            return 0  # safety fallback
        
        if random.random() < epsilon:
            return random.randint(0, N - 1)
        
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            wf_t = torch.as_tensor(valid_word_feats, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(obs_t, wf_t).squeeze(0)  # (N,)
            return int(q_values.argmax().item())

    def remember(self, obs, word_feat, reward, next_obs, done, next_valid_word_feats):
        """Store a transition. word_feat is the 130-dim encoding of the chosen word."""
        self.buffer.push(obs, word_feat, reward, next_obs, done, next_valid_word_feats)

    def train_step(self, batch_size=128):
        if len(self.buffer) < batch_size:
            return None

        batch, indices, weights = self.buffer.sample(batch_size)
        
        obs = torch.as_tensor(np.array([t.obs for t in batch]),
                              dtype=torch.float32, device=self.device)
        word_feats = torch.as_tensor(np.array([t.word_feat for t in batch]),
                                     dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor([t.reward for t in batch],
                                  dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.array([t.next_obs for t in batch]),
                                   dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([t.done for t in batch],
                                dtype=torch.float32, device=self.device)
        
        # Q(s, a) for the word that was actually chosen
        q_values = self.q_net(obs, word_feats)  # (B,)
        
        # Double DQN target computation
        with torch.no_grad():
            # For non-terminal states, compute max Q over next valid words
            max_next_q = torch.zeros(batch_size, device=self.device)
            for i, t in enumerate(batch):
                if not t.done and len(t.next_valid_word_feats) > 0:
                    next_obs_i = next_obs[i].unsqueeze(0)  # (1, 183)
                    nvwf = torch.as_tensor(t.next_valid_word_feats,
                                           dtype=torch.float32, device=self.device).unsqueeze(0)
                    
                    # Double DQN: use online net to SELECT best action
                    q_online = self.q_net(next_obs_i, nvwf).squeeze(0)
                    best_idx = q_online.argmax()
                    
                    # Use target net to EVALUATE that action
                    q_target = self.target_net(next_obs_i, nvwf).squeeze(0)
                    max_next_q[i] = q_target[best_idx]
            
            target = rewards + self.gamma * max_next_q * (1 - dones)
        
        td_errors = (q_values - target).detach().cpu().numpy()
        
        # Weighted loss for prioritized replay
        loss_per_sample = nn.functional.smooth_l1_loss(q_values, target, reduction='none')
        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_per_sample * weights).mean()
        else:
            loss = loss_per_sample.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities
        if indices is not None:
            self.buffer.update_priorities(indices, td_errors)
        
        return loss.item()

    def soft_update_target(self):
        """Polyak averaging: target = tau * online + (1-tau) * target"""
        for tp, op in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(self.tau * op.data + (1 - self.tau) * tp.data)

    def hard_update_target(self):
        """Full copy of online network weights to target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'q_net' in checkpoint:
            self.q_net.load_state_dict(checkpoint['q_net'])
            self.target_net.load_state_dict(checkpoint.get('target_net', checkpoint['q_net']))
            if 'optimizer' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                except Exception:
                    pass  # optimizer state may not match if architecture changed
        else:
            # Legacy format: raw state_dict (old v1 models)
            # Cannot load into new architecture — raise informative error
            raise ValueError(
                "This model file uses the legacy v1 format (fixed word-index output head). "
                "It is not compatible with the embedding-based v2 architecture. "
                "Legacy models are preserved in data/legacy/ for reference."
            )
