from dataclasses import dataclass, field
import numpy as np
import torch
from torch import Tensor as T
from itertools import permutations
from torch.nn import Module, Linear, MSELoss
import torch.nn.functional as F
import torch
from torch.optim import Adam

@dataclass
class Agent:
    """
    Agent class for Q-Learning and Deep Q-Network reinforcement learning.
    This class implements an agent capable of learning through both tabular Q-Learning
    and Deep Q-Networks (DQN). It manages the exploration-exploitation trade-off through
    an epsilon-greedy policy and handles both shallow and deep learning approaches.
    Attributes:
        learning_rate (float): The learning rate for Q-value updates. Controls how much
            new information overrides old Q-values in tabular Q-Learning or the optimizer
            step size in Deep Q-Networks.
        gamma (float): The discount factor (0 to 1) that determines how much the agent
            values future rewards compared to immediate rewards. Higher values prioritize
            long-term rewards.
        n_movements (int): The total number of available movement actions the agent can
            take in the environment.
        n_actions (int): The total number of available actions (including non-movement
            actions). This is calculated based on the action space size.
        n_states (int): The total number of possible states in the environment. Used to
            initialize the Q-table in tabular Q-Learning.
        epsilon (float): The exploration rate for the epsilon-greedy policy. Determines
            the probability of taking a random action (range 0 to 1).
        eps_min (float): The minimum threshold for epsilon. Prevents epsilon from
            decreasing below this value during training.
        eps_dec (float): The decay rate for epsilon (typically between 0.9 and 0.999).
            Multiplied with epsilon at each learning step to reduce exploration over time.
        Q (dict): The Q-table storing Q-values for state-action pairs in tabular
            Q-Learning. For Deep Q-Networks, this stores the neural network model.
            Defaults to an empty dictionary.
        deep (bool): Flag indicating whether to use Deep Q-Network (True) or tabular
            Q-Learning (False). Defaults to True.
        model_path (str): Path to a pre-trained model file for Deep Q-Networks.
            If provided, the model weights will be loaded upon initialization.
            Defaults to None.
    """
    learning_rate: float
    gamma: float
    n_movements: int
    n_actions: int
    n_states: int 
    epsilon: float
    eps_min: float
    eps_dec: float
    Q: dict = field(default_factory=dict)
    deep: bool = True
    model_path: str = None

    def __post_init__(self):
        """Initializes the agent after dataclass instantiation."""
        self.n_actions = self.n_movements
        self.action_space = list(range(self.n_movements))
        self.init_Q()

    def init_Q(self):
        """Initializes Q-values based on the learning approach."""
        if self.deep:
            self.Q = LinearQNet(self.learning_rate, self.n_states, self.n_actions)
            if self.model_path is not None:
                self.Q.load_state_dict(torch.load(self.model_path, map_location=self.Q.device, weights_only=True))
                self.Q.eval()
        else:
            for state in range(self.n_states):
                for action in range(self.n_actions):
                    self.Q[(state, action)] = 0.0

    def choose_action(self, state, deep=None):
        """Select an action with epsilon-greedy exploration."""
        deep = self.deep if deep is None else deep

        if np.random.random() < self.epsilon:
            return int(np.random.choice(self.action_space))

        if deep:
            with torch.no_grad():
                state_tensor = torch.as_tensor(
                    state, dtype=torch.float32, device=self.Q.device
                )
                return int(self.Q(state_tensor).argmax().item())

        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        q_values = [self.Q.get((state_key, action), 0.0) for action in self.action_space]
        return int(np.argmax(q_values))

    def decrement_epsilon(self):
        """Reduce epsilon without crossing the configured minimum."""
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_dec)

    def learn(self, state, action, reward, next_state, done=False):
        """Perform one Q-learning update; terminal states do not bootstrap."""
        if self.deep:
            self.Q.optimizer.zero_grad()

            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.Q.device
            )
            next_state_tensor = torch.as_tensor(
                next_state, dtype=torch.float32, device=self.Q.device
            )

            q_pred = self.Q(state_tensor)[int(action)]

            with torch.no_grad():
                next_q = 0.0 if done else self.Q(next_state_tensor).max().item()
                q_target = torch.tensor(
                    float(reward) + self.gamma * next_q,
                    dtype=torch.float32,
                    device=self.Q.device,
                )

            loss = self.Q.loss(q_pred, q_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=1.0)
            self.Q.optimizer.step()
        else:
            state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
            next_state_key = (
                tuple(next_state)
                if isinstance(next_state, (list, np.ndarray))
                else next_state
            )
            next_q = 0.0 if done else max(
                self.Q.get((next_state_key, a), 0.0) for a in self.action_space
            )
            key = (state_key, int(action))
            old_q = self.Q.get(key, 0.0)
            self.Q[key] = old_q + self.learning_rate * (
                float(reward) + self.gamma * next_q - old_q
            )

        self.decrement_epsilon()

    def get_movements(self):
        """Returns the list of available actions."""
        return self.action_space


class LinearQNet(Module):

    def __init__(self, learning_rate, input_dims, n_actions):
        super(LinearQNet, self).__init__()
        self.fc1 = Linear(input_dims, 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, n_actions)

        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.loss = MSELoss()

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions
