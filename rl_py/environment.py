from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class State:
    row: int = -1
    column: int = -1

    def __hash__(self) -> int:
        return hash((self.row, self.column))

    def clone(self) -> State:
        return State(self.row, self.column)


class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class Cell(Enum):
    ORDINARY = 0
    DAMAGE = -1
    REWARD = 1
    BLOCK = 9


@dataclass
class Environment:
    grid: List[List[Cell]]
    move_probability: float
    agent_state: State = State()
    default_reward: float = -0.04

    def __post_init__(self) -> None:
        self.reset()

    @property
    def row_length(self) -> int:
        return len(self.grid)

    @property
    def column_length(self) -> int:
        return len(self.grid[0])

    @property
    def actions(self) -> List[Action]:
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    @property
    def states(self) -> List[State]:
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                if self.grid[row][column] != Cell.BLOCK:
                    states.append(State(row, column))
        return states

    def _can_action_at(self, state: State) -> bool:
        return self.grid[state.row][state.column] == Cell.ORDINARY

    def _can_move_to(self, state: State) -> bool:
        return (
            (0 <= state.row < self.row_length)
            and (0 <= state.column < self.column_length)
            and self.grid[state.row][state.column] != Cell.BLOCK
        )

    def _move(self, state: State, action: Action) -> State:
        if not self._can_action_at(state):
            raise Exception("Cannot move from here")

        next_state = state.clone()
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        if self._can_move_to(next_state):
            return next_state
        return state

    def _transit_function(self, state: State, action: Action) -> Dict[State, float]:
        transition_probabilities: Dict[State, float] = {}
        if not self._can_action_at(state):
            return transition_probabilities

        opposite_direction = Action(action.value * -1)

        for a in self.actions:
            if a == action:
                probability = self.move_probability
            elif a != opposite_direction:
                probability = (1.0 - self.move_probability) / 2.0
            else:
                probability = 0.0

            next_state = self._move(state, a)
            if next_state not in transition_probabilities:
                transition_probabilities[next_state] = probability
            else:
                transition_probabilities[next_state] += probability

        return transition_probabilities

    def _reward_function(self, state: State) -> Tuple[float, bool]:
        reward = self.default_reward
        done = False

        attribute = self.grid[state.row][state.column]
        if attribute == Cell.REWARD:
            reward = 1
            done = True
        elif attribute == Cell.DAMAGE:
            reward = -1
            done = True

        return reward, done

    def reset(self) -> State:
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    def _transit(
        self, state: State, action: Action
    ) -> Tuple[Optional[State], Optional[float], bool]:
        transition_probabilities = self._transit_function(state, action)
        if len(transition_probabilities) == 0:
            return None, None, True

        next_states = list(transition_probabilities.keys())
        probabilities = list(transition_probabilities.values())
        next_state = np.random.choice(next_states, p=probabilities)
        reward, done = self._reward_function(next_state)

        return next_state, reward, done

    def step(
        self, action: Action
    ) -> Tuple[Optional[State], Optional[float], bool]:
        next_state, reward, done = self._transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state
        return next_state, reward, done
