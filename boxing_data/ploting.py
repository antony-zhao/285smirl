import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


reward_0 = pd.read_csv("boxing_reward_first_0.csv")
reward_1 = pd.read_csv("boxing_reward_second_0.csv")
train_reward_0 = pd.read_csv("boxing_train_first_0.csv")
train_reward_1 = pd.read_csv("boxing_train_second_0.csv")
loss_0 = pd.read_csv("boxing_loss_first_0.csv")
loss_1 = pd.read_csv("boxing_loss_second_0.csv")
print(np.mean(reward_0)), print(np.std(reward_0))
print(np.mean(reward_1)), print(np.std(reward_1))


for df in [reward_0, reward_1, train_reward_0, train_reward_1]:
    df['epoch'] = df.index * 100


for df in [loss_0, loss_1]:
    df['epoch'] = df.index * 10

def smooth_data(data, window_size=5):
    return data.rolling(window=window_size, min_periods=1).mean()

def confidence_interval(data, window_size=5):
    return data.rolling(window=window_size, min_periods=1).std()

def plot(data_0, data_1, title, legend_0, legend_1, xlabel, ylabel, color_0, color_1, file_name):

    smoothed_data_0 = smooth_data(data_0[data_0.columns[0]])
    smoothed_data_1 = smooth_data(data_1[data_1.columns[0]])

    ci_data_0 = confidence_interval(data_0[data_0.columns[0]])
    ci_data_1 = confidence_interval(data_1[data_1.columns[0]])

    # Plotting
    # Loss
    plt.figure(figsize=(7, 5))
    ax = plt.gca()  # Get the current Axes instance on the current figure

    palette = sns.color_palette("deep")
    ax.set_facecolor('#f0f0f0')  # Axes background color

    #plt.gcf().set_facecolor('#f0f0f0') 
    sns.lineplot(data=data_0, x='epoch', y=data_0[data_0.columns[0]], label=legend_0, color=palette[color_0])
    #plt.fill_between(data_0['epoch'], smoothed_data_0-ci_data_0, smoothed_data_0+ci_data_0, alpha=0.2, color=palette[color_0])

    sns.lineplot(data=data_1, x='epoch', y=data_1[data_1.columns[0]], label=legend_1, color=palette[color_1])
    #plt.fill_between(data_1['epoch'], smoothed_data_1-ci_data_1, smoothed_data_1+ci_data_1, alpha=0.2, color=palette[color_1])

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    plt.grid(True, color='white', linewidth=1.5)
    plt.legend(framealpha=0.5)


    file_path = os.path.join('plots', file_name + '.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()


plot(reward_0, reward_1, "Eval Reward for Boxing: SMiRL Bonus vs Pure Reward", "SMiRL Bonus", "Pure Reward", "Epochs", "Eval Reward", 0, 3, 'SMiRL_vs_DQN_Eval_Reward')
plot(train_reward_0, train_reward_1, "Train Reward for Boxing: SMiRL Bonus vs Pure Reward", "SMiRL Bonus", "Pure Reward", "Epochs", "Train Reward", 1, 2, 'SMiRL_vs_DQN_Train_Reward')
plot(loss_0, loss_1, "Loss for Boxing: SMiRL Bonus vs Pure Reward", "SMiRL Bonus", "Pure Reward", "Epochs", "Loss", 4, 5, 'SMiRL_vs_DQN_Loss')

