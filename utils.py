import os
import matplotlib.pyplot as plt

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def progress_bar(idx, size, description):
    print('Hello')


def make_train_figure(loss_train, loss_val, acc_train, acc_val, path):
    plt.plot(loss_train, label='Loss Train')
    plt.plot(loss_val, label='Loss Val')
    plt.plot(acc_train, label='Acc Train')
    plt.plot(acc_val, label='Acc Val')
    plt.legend()
    plt.savefig(path)
    plt.show()
