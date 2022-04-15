from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Tuple

from .environment import Action, Environment, State

RewardGrid = List[List[float]]


class Planner:
    env: Environment
    log: List[RewardGrid]

    def __init__(self, env: Environment) -> None:
        self.env = env

    def initialize(self) -> None:
        self.env.reset()
        self.log = []

    @abstractmethod
    def plan(self, gamma: float = 0.9, threshold: float = 1e-4) -> RewardGrid:
        pass

    def transitions_at(
        self, state: State, action: Action
    ) -> Iterator[Tuple[float, State, float]]:
        transition_probabilities = self.env.transit_function(state, action)
        for next_state, probability in transition_probabilities.items():
            reward, _ = self.env.reward_function(next_state)
            yield probability, next_state, reward

    def dict_to_grid(self, state_reward: Dict[State, float]) -> RewardGrid:
        grid = []
        for _ in range(self.env.row_length):
            row = [0.0] * self.env.column_length
            grid.append(row)

        for state, reward in state_reward.items():
            grid[state.row][state.column] = reward

        return grid


@dataclass
class ValueIterationPlanner(Planner):
    env: Environment

    def plan(self, gamma: float = 0.9, threshold: float = 0.0001) -> RewardGrid:
        self.initialize()
        actions = self.env.actions
        V: Dict[State, float] = {}
        for s in self.env.states:
            V[s] = 0

        while True:
            gain = 0.0
            self.log.append(self.dict_to_grid(V))

            for state in V:
                if not self.env._can_action_at(state):
                    continue

                expected_rewards = []
                for action in actions:
                    r = 0.0
                    for probability, next_state, reward in self.transitions_at(
                        state, action
                    ):
                        r += probability * (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                max_reward = max(expected_rewards)
                gain = max(gain, abs(max_reward - V[state]))
                V[state] = max_reward

            if gain < threshold:
                break

        return self.dict_to_grid(V)
