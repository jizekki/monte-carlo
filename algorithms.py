import numpy as np

from abc import ABC

# Abstract class for all the algorithms
class Algorithm(ABC):
    def __init__(self, casino, initialCredits) -> None:
        self._casino = casino
        self._initialCredits = initialCredits
        self._runsAndRewards = None
        self.initialize()

    def initialize(self):
        self._runsAndRewards = np.zeros((self._casino.machines_count, 2))


# Abstract class for all glouton algorithms
class AbstractGlouton(Algorithm):
    def __init__(
        self,
        casino,
        initialCredits=10000,
        testIterations=5,
    ) -> None:
        super().__init__(casino, initialCredits)
        self._testIterations = testIterations

    def do_abstract_glouton(self):
        self.initialize()
        for _ in range(self._testIterations):
            for machineIndex in range(self._casino.machines_count):
                self._runsAndRewards[machineIndex, 1] += self._casino.machines[
                    machineIndex
                ].play()
                self._runsAndRewards[machineIndex, 0] += 1


# Implementing the Glouton method
class Glouton(AbstractGlouton):
    def __init__(self, casino, testIterations=5, initialCredits=10000) -> None:
        super().__init__(casino, initialCredits, testIterations)

    def do_glouton(self):
        super().do_abstract_glouton()
        remaining_play_credits = (
            self._initialCredits - self._testIterations * self._casino.machines_count
        )
        best_machine_index = np.argmax(self._runsAndRewards[:, 1])
        while remaining_play_credits > 0:
            self._runsAndRewards[best_machine_index, 1] += self._casino.machines[
                best_machine_index
            ].play()
            self._runsAndRewards[best_machine_index, 0] += 1
            remaining_play_credits -= 1
        return self._runsAndRewards[:, 1].sum()


# Implementing the epsilon-Glouton method
class EpsilonGlouton(AbstractGlouton):
    def __init__(
        self,
        casino,
        initialCredits=10000,
        testIterations=5,
        epsilon=0.1,
    ) -> None:
        super().__init__(casino, initialCredits, testIterations)
        self._epsilon = epsilon

    def do_epsilon_glouton(self):
        super().do_abstract_glouton()
        remaining_play_credits = (
            self._initialCredits - self._testIterations * self._casino.machines_count
        )
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
class UCB(Algorithm):
    def __init__(self, casino, initialCredits=10000) -> None:
        super().__init__(casino, initialCredits)
        self._n = None
        self.initialize()

    def initialize(self):
        super().initialize()
        self._n = 0

    def _maximal_reward_arg(self, c=1):
        """
        Gets the machine with the highest reward so far.
        If a machine that has never been tried is found, it will be returned
        """
        arg = None
        maximal_reward = 0
        for a in range(self._casino.machines_count):
            if self._runsAndRewards[a, 0] == 0:
                return a
            else:
                delta = np.sqrt(2 * np.log(self._n) / self._runsAndRewards[a, 0])
                reward = self._runsAndRewards[a, 1] + c * delta
                if reward > maximal_reward:
                    maximal_reward = reward
                    arg = a
        return arg

    def do_one_step_ucb(self, c=1):
        if self._n >= self._initialCredits:
            return None, self._n
        arg = self._maximal_reward_arg(c)
        reward = self._casino.machines[arg].play()
        self._runsAndRewards[arg, 1] = (
            self._runsAndRewards[arg, 0] * self._runsAndRewards[arg, 1] + reward
        ) / (self._runsAndRewards[arg, 0] + 1)
        self._runsAndRewards[arg, 0] += 1
        self._n += 1
        return reward, self._n - 1

    def do_ucb(self, c=1):
        self.initialize()
        v = 0
        reward, _ = self.do_one_step_ucb(c)
        while reward is not None:
            v += reward
            reward, _ = self.do_one_step_ucb(c)
        return v
