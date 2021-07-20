import csv

import matplotlib.pyplot as plt


def plot_records():
    bests = []
    means = []
    worsts = []
    with open("./records.csv", "r") as _file:
        reader = csv.reader(_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for b, w, m in reader:
            bests.append(b)
            means.append(m)
            worsts.append(w)
    plt.plot(bests, color="red")
    plt.plot(worsts, color="blue")
    plt.plot(means, color="green")
    plt.show()


if __name__ == '__main__':
    plot_records()
