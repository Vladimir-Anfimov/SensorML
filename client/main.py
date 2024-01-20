import matplotlib.pyplot as plt
from pyscript import document


def show_plot(title: str):
    plt.plot([1, 2, 3, 4])
    plt.ylabel("some numbers")
    plt.title(title)
    plt.show()
