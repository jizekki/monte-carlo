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
        "maximal reward : {:.2f} (loss=0)".format(casino.real_best_choice().mu * 10000)
    )
