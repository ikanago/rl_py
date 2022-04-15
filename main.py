import numpy as np

from rl_py.environment import Action, Cell, Environment, State
from rl_py.planner import ValueIterationPlanner


class Agent():
    def __init__(self, env: Environment) -> None:
        self.actions = env.actions

    def policy(self, state: State) -> Action:
        return np.random.choice(self.actions)  # type: ignore


def main() -> None:
    grid = [
        [Cell.ORDINARY, Cell.ORDINARY, Cell.ORDINARY, Cell.REWARD],
        [Cell.ORDINARY, Cell.BLOCK, Cell.ORDINARY, Cell.DAMAGE],
        [Cell.ORDINARY, Cell.ORDINARY, Cell.ORDINARY, Cell.ORDINARY],
    ]
    env = Environment(grid, 0.7)
    planner = ValueIterationPlanner(env)
    print(planner.plan())


if __name__ == "__main__":
    main()

