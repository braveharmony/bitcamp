import numpy as np
import scipy.special
# 각 클래스가 선택될 확률
p = np.array([0.35, 0.2167, 0.2167, 0.2167])

def calc_class1_most_prob(trials, p):
    prob_class1_most = 0

    for i in range(trials + 1):
        for j in range(trials + 1 - i):
            for k in range(trials + 1 - i - j):
                l = trials - i - j - k
                prob_case = (p[0] ** i) * (p[1] ** j) * (p[2] ** k) * (p[3] ** l) * \
                            scipy.special.binom(trials, i) * \
                            scipy.special.binom(trials - i, j) * \
                            scipy.special.binom(trials - i - j, k)
                if i > j and i > k and i > l:
                    prob_class1_most += prob_case

    return prob_class1_most

trials = 60
result = calc_class1_most_prob(trials, p)
print("60번 시도에서 클래스 1이 가장 많이 선택될 확률: {:.2%}".format(result))