import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


q_list = []


def plot_q_val(title, q_val_file, optimal_steps):
    with open(q_val_file, 'r') as filehandle:
        for line in filehandle:
            q_list.append(int(line.strip()))
    episodes = range(1, len(q_list) + 1)
    print(min(q_list))

    plt.figure()
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Steps to Terminal")
    plt.grid(alpha=0.3)
    plt.axhline(y=optimal_steps, linewidth=0.7, color='r', linestyle=':')
    plt.plot(episodes, q_list, color="#007fff", linewidth=0.3)
    plt.savefig(title + ".png")
    plt.close()


if __name__ == "__main__":
    #plot_q_val("GW Q Learning", "GW_q.txt", 20)
    #plot_q_val("BD Lv1 Q Learning", "BD_1_q.txt", 19)
    plot_q_val("BD Lv3 Q Learning", "BD_3_q.txt", 94)
