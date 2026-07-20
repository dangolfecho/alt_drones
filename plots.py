import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

envs = ["PyFlyt/QuadX-Hover-v4", "PyFlyt/QuadX-Pole-Balance-v4",
        "PyFlyt/QuadX-Ball-In-Cup-v4", "PyFlyt/QuadX-Pole-Waypoints-v4",
        "PyFlyt/QuadX-Waypoints-v4", "PyFlyt/Fixedwing-Waypoints-v3", "PyFlyt/Rocket-Landing-v4"]
algos = ['a2c', 'ddpg', 'sac', 'td3', 'ppo']

DEFAULT_MODE = 0
DEFAULT_ENV = 0
DEFAULT_ALGO = 0

def user_model_plot(fig, ax, env_name, algo_name, y_element):
    arr = pd.read_csv(f'backup/2026_07_18/{env_name}/{algo_name}/run_1/progress.csv',
            delimiter=',')

    if(not(os.path.isdir(f'plots/{env_name}'))):
        os.mkdir(f'plots/{env_name}/')

    col_names = list(arr.columns.values)
    time_index = col_names.index('time/total_timesteps')
    rew_index = col_names.index(y_element)

    ax.plot(arr.iloc[:, time_index], arr.iloc[:, rew_index],
            label=f'{algo_name.upper()}')
    

def plot_across_models(env_num=DEFAULT_ENV):#same env
    env_name = (envs[env_num]).split('/')[1]
    fig, ax = plt.subplots()
    for i in algos:
        user_model_plot(fig, ax, env_name, i, 'rollout/ep_rew_mean')
    ax.set_xlabel('Number of time steps')
    ax.set_ylabel(f'Episode Reward Mean')
    ax.set_title(f'{env_name} - Reward comparison across all algorithms')
    ax.legend()
    plt.savefig(f'plots/{env_name}/reward_comp_across_models.png')

def get_files_list(path):
    folders = sorted(os.listdir(path))
    files = []
    for folder in folders:
        if(folder.find('.') != -1):
            continue
        append_path = f'{path}/{folder}'
        sfile_name = os.listdir(f'{path}/{folder}/')[0]
        files.append(f'{append_path}/{sfile_name}')
    return files


def main(mode=DEFAULT_MODE, env_num=DEFAULT_ENV, algo_num=DEFAULT_ALGO):
    if(mode == 1):
        plot_across_models(env_num)
        return
    env_name = (envs[env_num]).split('/')[1]
    algo_name = algos[algo_num]

    files_list = get_files_list(f'results/{env_name}/{algo_name}')
    arr = pd.concat([pd.read_csv(f) for f in files_list], ignore_index=True)

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
    parser.add_argument('mode', type=int, default=DEFAULT_MODE, help='which\
    mode to run')
    parser.add_argument('env_num', type=int, default=DEFAULT_ENV, help='which\
            env to plot')
    parser.add_argument('algo_num', type=int, default=DEFAULT_ALGO, help='which\
            algorithm to plot')
    ARGS = parser.parse_args()
    main(**vars(ARGS))
