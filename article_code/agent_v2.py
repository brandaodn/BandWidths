from collections import deque
from dataclasses import dataclass
import math
import random
import numpy as np
import torch
from torch import nn
from torch.optim import Adam


class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class DQNConfig:
    state_dim: int
    n_actions: int
    horizon: int = 500
    learning_rate: float = 1e-3
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    replay_capacity: int = 10000
    batch_size: int = 64
    warmup: int = 64
    target_update_interval: int = 25
    gradient_clip: float = 1.0
    hidden: int = 128


class DQNAgentV2:
    """DQN with replay memory, target network and horizon-aware epsilon decay."""

    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.online = QNetwork(cfg.state_dim, cfg.n_actions, cfg.hidden).to(self.device)
        self.target = QNetwork(cfg.state_dim, cfg.n_actions, cfg.hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = Adam(self.online.parameters(), lr=cfg.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = deque(maxlen=cfg.replay_capacity)
        self.learn_steps = 0
        self.env_steps = 0

        if cfg.horizon <= 0:
            self.epsilon_decay = 1.0
        else:
            self.epsilon_decay = (cfg.epsilon_end / cfg.epsilon_start) ** (1.0 / cfg.horizon)

    @property
    def epsilon(self):
        return max(
            self.cfg.epsilon_end,
            self.cfg.epsilon_start * (self.epsilon_decay ** self.env_steps),
        )

    def choose_action(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            action = random.randrange(self.cfg.n_actions)
        else:
            x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action = int(self.online(x).argmax(dim=1).item())
        if explore:
            self.env_steps += 1
        return action

    def remember(self, state, action, reward, next_state, done):
        self.replay.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))

    def learn(self):
        if len(self.replay) < max(self.cfg.warmup, self.cfg.batch_size):
            return None

        batch = random.sample(self.replay, self.cfg.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1).values
            target = rewards + self.cfg.gamma * (1.0 - dones) * next_q

        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.gradient_clip)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.cfg.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())
