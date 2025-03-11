import os

import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, epoch_number: int, x_label="", y_label=""):
        self.data = [[] for _ in range(epoch_number)]
        self.x_label = x_label
        self.y_label = y_label

    def add_data(self, epoch_index, data):
        self.data[epoch_index].append(data)

    def set_x_label(self, x_label):
        self.x_label = x_label

    def set_y_label(self, y_label):
        self.y_label = y_label

    def make_plot(self, plot_name: str):
        plt.plot([(sum(i) / len(i)) for i in self.data])
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        if not os.path.exists("./" + "/Plots/"):
            os.makedirs("./Plots/")
        plt.savefig("./Plots/" + plot_name + ".png")
        plt.cla()
