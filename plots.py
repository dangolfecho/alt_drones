import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

envs = ["PyFlyt/QuadX-Hover-v4", "PyFlyt/QuadX-Pole-Balance-v4",
        "PyFlyt/QuadX-Ball-In-Cup-v4", "PyFlyt/QuadX-Pole-Waypoints-v4",
        "PyFlyt/QuadX-Waypoints-v4", "PyFlyt/Fixedwing-Waypoints-v3", "PyFlyt/Rocket-Landing-v4"]
algos = ['a2c', 'ddpg', 'sac', 'td3', 'ppo']

DEFAULT_ENV = 0
DEFAULT_ALGO = 0

def main(env_num=DEFAULT_ENV, algo_num=DEFAULT_ALGO):
    env_name = (envs[env_num]).split('/')[1]
    algo_name = algos[algo_num]
    arr = pd.read_csv(f'backup/results/{env_name}/{algo_name}/progress.csv', delimiter=',')
    
    col_names = list(arr.columns.values)

    if(not(os.path.isdir(f'plots/{env_name}'))):
        os.mkdir(f'plots/{env_name}/') 
    if(not(os.path.isdir(f'plots/{env_name}/{algo_name}/'))):
        os.mkdir(f'plots/{env_name}/{algo_name}/')

    time_index = col_names.index('time/total_timesteps')
    for i in range(len(col_names)):
        if(i != time_index):
            name_p1, name_p2 = col_names[i].split('/')
            fig, ax = plt.subplots()
            ax.plot(arr.iloc[:, time_index], arr.iloc[:, i])
            
            ax.set_xlabel('Number of time steps')
            ax.set_ylabel(f'{name_p1}_{name_p2}')
            ax.set_title(f'{env_name} - {algo_name.upper()}')
            plt.savefig(f'plots/{env_name}/{algo_name}/{name_p1}_{name_p2}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='plots.py',
            description='does plotting',
            )
    parser.add_argument('env_num', type=int, default=DEFAULT_ENV, help='which\
            env to plot')
    parser.add_argument('algo_num', type=int, default=DEFAULT_ALGO, help='which\
            algorithm to plot')
    ARGS = parser.parse_args()
    main(**vars(ARGS))
