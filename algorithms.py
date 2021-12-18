import numpy as np

np.random.seed(1001)


# Implementing the Glouton method
class Glouton:
    def __init__(self, casino, testIterations=5, initialCredits=10000) -> None:
        self._casino = casino
        self._initialCredits = initialCredits
        self._testIterations = testIterations
        self._runsAndRewards = None
        self.initialize()

    def initialize(self):
        self._runsAndRewards = np.zeros(shape=(self._casino.machines_count, 2))

    def do_glouton(self):
        remaining_play_credits = (
            self._initialCredits - self._testIterations * self._casino.machines_count
        )
        for _ in range(self._testIterations):
            for machineIndex in range(self._casino.machines_count):
                self._runsAndRewards[machineIndex, 1] += self._casino.machines[
                    machineIndex
                ].play()
                self._runsAndRewards[machineIndex, 0] += 1

        best_machine_index = np.argmax(self._runsAndRewards[:, 1])
        while remaining_play_credits > 0:
            self._runsAndRewards[best_machine_index, 1] += self._casino.machines[
                best_machine_index
            ].play()
            self._runsAndRewards[best_machine_index, 0] += 1
            remaining_play_credits -= 1
        return self._runsAndRewards[:, 1].sum()


# Implementing the epsilon-Glouton method
class EpsilonGlouton:
    def __init__(
        self, casino, testIterations=5, epsilon=0.1, initialCredits=10000
    ) -> None:
        self._casino = casino
        self._initialCredits = initialCredits
        self._testIterations = testIterations
        self._epsilon = epsilon
        self._runsAndRewards = None
        self.initialize()

    def initialize(self):
        self._runsAndRewards = np.zeros(shape=(self._casino.machines_count, 2))

    def do_epsilon_glouton(self):
        remaining_play_credits = (
            self._initialCredits - self._testIterations * self._casino.machines_count
        )
        for _ in range(self._testIterations):
            for machineIndex in range(self._casino.machines_count):
                self._runsAndRewards[machineIndex, 1] += self._casino.machines[
                    machineIndex
                ].play()
                self._runsAndRewards[machineIndex, 0] += 1

        best_machine_index = np.argmax(self._runsAndRewards[:, 1])
        while remaining_play_credits > 0:
            if np.random.random() > self._epsilon:
                self._runsAndRewards[best_machine_index, 1] += self._casino.machines[
                    best_machine_index
                ].play()
                self._runsAndRewards[best_machine_index, 0] += 1
            else:
                random_index = np.random.randint(0, self._casino.machines_count - 1)
                self._runsAndRewards[random_index, 1] += self._casino.machines[
                    random_index
                ].play()
                self._runsAndRewards[random_index, 0] += 1
            remaining_play_credits -= 1
        return self._runsAndRewards[:, 1].sum()


# Implementing the UCB method
class UCB:
    def __init__(self, casino, initialCredits=10000) -> None:
        self._casino = casino
        self._initialCredits = initialCredits
        self._n = None
        self._runsAndRewards = None
        self.initialize()

    def initialize(self):
        self._n = 0
        self._runsAndRewards = np.zeros(shape=(self._casino.machines_count, 2))

    def _maximal_reward_arg(self, iteration, c):
        arg = None
        maximal_reward = 0
        for a in range(self._casino.machines_count):
            if self._runsAndRewards[a, 0] == 0:
                return a
            else:
                reward = self._runsAndRewards[a, 1] + c * np.sqrt(
                    2 * np.log(iteration) / self._runsAndRewards[a, 0]
                )
                if reward > maximal_reward:
                    maximal_reward = reward
                    arg = a
        return arg

    def do_one_step_ucb(self, c=1):
        if self._n >= self._initialCredits:
            return None, self._n
        arg = self._maximal_reward_arg(self._n, c)
        reward = self._casino.machines[arg].play()
        self._runsAndRewards[arg, 1] = (
            self._runsAndRewards[arg, 0] * self._runsAndRewards[arg, 1] + reward
        ) / (self._runsAndRewards[arg, 0] + 1)
        self._runsAndRewards[arg, 0] += 1
        self._n += 1
        return reward, self._n

    def do_ucb(self, c=1):
        v, n = 0, 0
        while n < self._initialCredits:
            arg = self._maximal_reward_arg(n, c)
            reward = self._casino.machines[arg].play()
            v += reward
            self._runsAndRewards[arg, 1] = (
                self._runsAndRewards[arg, 0] * self._runsAndRewards[arg, 1] + reward
            ) / (self._runsAndRewards[arg, 0] + 1)
            self._runsAndRewards[arg, 0] += 1
            n += 1
        return v
