from casino import Casino
from algorithms import Glouton, EpsilonGlouton, UCB

if __name__ == "__main__":

    casino = Casino(nb=10)
    glouton = Glouton(casino)
    epsilonGlouton = EpsilonGlouton(casino)
    ucb = UCB(casino)

    print("Glouton reward: {}".format(glouton.do_glouton()))
    print("Epsilon Glouton reward: {}".format(epsilonGlouton.do_epsilon_glouton()))
    print("UCB reward: {}".format(ucb.do_ucb()))
    print(
        "best choice reward : {:.2f}".format(
            sum(casino.real_best_choice().play() for _ in range(10000))
        )
    )
