import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_final_bounds(file_path):

    data = pd.read_csv(file_path)
    
    epsilons = data['Epsilon'].values
    anon_bounds = data['Anonymization Bound'].values
    rand_bounds = data['Randomization Bound'].values

    plt.figure(figsize=(10, 5))
    plt.plot(epsilons[::-1], anon_bounds[::-1], label='Anonymization Bounds', marker='o')
    plt.plot(epsilons[::-1], rand_bounds[::-1], label='Randomization Bounds', marker='o')
    plt.ylim((0,1))
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Azuma-Hoeffding Bound')
    plt.title('Azuma-Hoeffding Bounds vs Epsilon')
    plt.legend()
    plt.grid(True)
    plt.savefig('azuma_hoeffding_bounds.png')
    plt.close()

if __name__ == "__main__":
    plot_final_bounds('final_azuma_hoeffding_bounds.csv')
    print("Plot saved as 'azuma_hoeffding_bounds.png'.")