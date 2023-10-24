
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def plot_statetxt(file_path='/home/user/test2/state_model_all.txt'):
  
    output_dir = os.path.dirname(file_path)
    # Read and parse the raw data file
    episodes = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        match = re.search(r'ep: (\d+) \| <Compute_time : Done : Action>: \[(.+)\]', line)
        if match:
            episode_number = int(match.group(1))
            tuples_str = match.group(2)
            tuples = eval(f'[{tuples_str}]')
            episodes[episode_number] = tuples

    # Batch and aggregate the data
    batch_size = 100
    batches = {}
    batch_count = 1
    episode_count = 0
    batch_data = []
    for episode, tuples in episodes.items():
        episode_count += 1
        batch_data.extend(tuples)
        if episode_count % batch_size == 0:
            action_speedup = {}
            action_count = {}
            for speedup, state, action in batch_data:
                if action not in action_speedup:
                    action_speedup[action] = []
                    action_count[action] = 0
                if not state:
                    action_speedup[action].append(speedup)
                    action_count[action] += 1
            for action, speedups in action_speedup.items():
                action_speedup[action] = np.mean(speedups) if speedups else 0
            batches[batch_count] = {'action_speedup': action_speedup, 'action_count': action_count}
            batch_data = []
            batch_count += 1

    # Prepare data for the heatmap
    df = pd.DataFrame({
        'Action': [action for batch in batches.values() for action in batch['action_speedup'].keys()],
        'Batch': [batch for batch, data in batches.items() for _ in data['action_speedup'].keys()],
        'Average Speedup': [speedup for batch in batches.values() for speedup in batch['action_speedup'].values()],
        'Action Count': [count for batch in batches.values() for count in batch['action_count'].values()]
    })
    df['Action Frequency per Episode'] = df['Action Count'] / batch_size

    # Calculate the maximum speedup per batch
    max_speedup_per_episode_in_batch = {}
    for batch, _ in batches.items():
        max_speedups = []
        # Extract the relevant episodes for this batch
        relevant_episodes = list(episodes.keys())[batch_size*(batch-1):batch_size*batch]
        for episode in relevant_episodes:
            tuples = episodes[episode]
            max_speedup = 0
            for speedup, state, _ in tuples:
                if speedup > max_speedup:
                    max_speedup = speedup
                if state:
                    max_speedups.append(max_speedup)
                    max_speedup = 0  # Reset for next game in the same episode
        max_speedup_per_episode_in_batch[batch] = np.mean(max_speedups) if max_speedups else 0

    # Generate the heatmap
    plt.figure(figsize=(20, 15))
    ax = sns.heatmap(df.pivot(index='Action', columns='Batch', values='Action Frequency per Episode'),
                    annot=df.pivot(index='Action', columns='Batch', values='Average Speedup'),
                    fmt=".2f", linewidths=.5,
                    cmap="RdYlGn_r", cbar_kws={'label': 'Action Frequency per Episode'},
                    linecolor='white')
    plt.title('Heatmap of Average Speedup and Action Frequency per Action per Batch')
    plt.xlabel('Batch of 300 completed games')
    plt.ylabel('Action')

    # Add a second x-axis at the top of the heatmap for the average maximum speedup per batch
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f"{max_speedup_per_episode_in_batch[i]:.2f}" for i in sorted(max_speedup_per_episode_in_batch.keys())])
    ax2.set_xlabel("Average Maximum Speedup per Batch")
    # plt.show()
    plt.savefig(os.path.join(output_dir, 'heatmap.png'))
    
    # Calculate the maximum and average speedup per batch
    max_speedup_per_episode_in_batch = {}
    avg_speedup_per_action_in_batch = {}
    for batch, data in batches.items():
        max_speedups = []
        total_speedup = 0
        total_actions = 0
        max_speedup = 0
        for speedup, state, action in [(speedup, state, action) for episode, tuples in sorted(episodes.items())[batch_size*(batch-1):batch_size*batch] for speedup, state, action in tuples]:
            if speedup > max_speedup:
                max_speedup = speedup
            if state:
                max_speedups.append(max_speedup)
                max_speedup = 0
            total_speedup += speedup
            total_actions += 1
        max_speedup_per_episode_in_batch[batch] = np.mean(max_speedups) if max_speedups else 0
        avg_speedup_per_action_in_batch[batch] = total_speedup / total_actions if total_actions else 0

    # Generate the line plot
    plt.figure(figsize=(14, 8))
    plt.plot(list(max_speedup_per_episode_in_batch.keys()), list(max_speedup_per_episode_in_batch.values()), label='Maximum Speedup per Batch', marker='o')
    plt.plot(list(avg_speedup_per_action_in_batch.keys()), list(avg_speedup_per_action_in_batch.values()), label='Average Speedup per Action per Batch', marker='x')
    plt.xlabel('Batch of 300 completed games')
    plt.ylabel('Speedup')
    plt.title('Maximum Speedup and Average Speedup per Action Across Batches')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(output_dir, 'line_plot.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data and generate plots.')
    parser.add_argument('--file_path', type=str, default='/home/user/test2/state_model_all.txt', help='Path to the text file containing the data.')
    args = parser.parse_args()
    plot_statetxt(args.file_path)
