import numpy as np
import random
import math
import matplotlib.pyplot as plt


class Benford():
    def __init__(self, test_number=1000):
        self.test_number = test_number
        np.random.seed(random.randint(1, 1000000))
        self.randoms = 10 ** np.random.rand(self.test_number)
        self.find_number_distribution()
        self.x_axis = np.array(range(1, 10))
        self.benford = np.array([math.log(1.0 + (1 / i), 10) for i in range(1, 10)])

    def find_number_distribution(self):
        self.distribution = np.zeros(10)
        first_digits = [int(str(format(abs(x), ".6e"))[0]) for x in self.randoms if x != 0]
        for i in first_digits:
            self.distribution[i] += 1
        # for number in self.randoms:
            # number_str = str(number)
            # for char in number_str:
            #     if char != '0' and char != '.':
            #         self.distribution[int(char)] += 1
            #         break
        self.distribution = self.distribution / self.test_number


    def show_distribution_and_formula(self):
        plt.plot(self.x_axis, self.distribution[1:], color='red', label='numbers distribution')
        plt.plot(self.x_axis, self.benford, color='green', label='benford law')
        plt.legend(loc='best')
        plt.show()