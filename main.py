import numpy as np

from rl_py.environment import Action, Cell, Environment, State


class Agent():
    def __init__(self, env: Environment) -> None:
        self.actions = env.actions

    def policy(self, state: State) -> Action:
        return np.random.choice(self.actions)


def main() -> None:
    grid = [
        [Cell.ORDINARY, Cell.ORDINARY, Cell.ORDINARY, Cell.REWARD],
        [Cell.ORDINARY, Cell.BLOCK, Cell.ORDINARY, Cell.DAMAGE],
        [Cell.ORDINARY, Cell.ORDINARY, Cell.ORDINARY, Cell.ORDINARY],
    ]
    env = Environment(grid, 0.7)
    agent = Agent(env)

    for i in range(10):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(action)
            if reward is not None:
                total_reward += reward
            if next_state is not None:
                state = next_state

        print(f"Episode {i}: Agent got {total_reward}")


if __name__ == "__main__":
    main()

