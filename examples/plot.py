from tqdm import tqdm
import wandb
api = wandb.Api(timeout=180)
import os
import pandas as pd
import wandb
import yaml
from pathlib import Path
from copy import deepcopy
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import itertools
import  matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import itertools 
import matplotlib as mpl
import matplotlib.ticker as ticker 
from scipy.interpolate import make_interp_spline, BSpline 
mpl.style.use('seaborn-whitegrid')

def download_wandb_summary(user, project, summary_file, key_focus=[], keyval_focus={}, reload=False):
    """
    Download a summary of all runs on the wandb project
    """
    if not reload:
        return pd.read_csv('logs/wandb_data/'+summary_file, header=0)
    runs = api.runs(user+'/'+project, per_page=1e7)
    summary_list, config_list, name_list, id_list, commits = [], [], [], [], []
    assert len([run for run in runs])
    for run in tqdm(runs):
        run = api.run(user+'/'+project+"/"+run.id)
        conf = {k: v for k, v in run.config.items()}
        include = True
        # check for simple key requirements
        if not all(item in list(conf.keys()) for item in key_focus):
            include = False
        # check if we have key-value requirements
        for key in keyval_focus.keys():
            if key not in list(conf.keys()):
                include = False
            else:
                include *= bool(np.sum([conf[key] == d for d in keyval_focus[key]]))
        # if this data-point is to be included then append it to summary file.
        if include:
            summary_list.append(run.summary._json_dict)
            config_list.append(conf)
            name_list.append(run.name)
            id_list.append(run.id)
            if run.commit is not None:
                commits.append(run.commit)
            else:
                commits.append('None')
    assert len(summary_list)
    commits_df = pd.DataFrame.from_records(commits)
    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({"name": name_list, "id": id_list})
    all_df = pd.concat([name_df, config_df, summary_df, commits_df], axis=1)
    Path('logs/wandb_data/').mkdir(parents=True, exist_ok=True)
    all_df.to_csv('logs/wandb_data/'+summary_file)

def download_wandb_records(user, project, summary_file, columns_of_interest = [], reload=True):
    """
    Download data for all runs in summary file
    """
    if not reload:
        return pd.read_csv('logs/wandb_data/__full__'+summary_file, header=0)
    # load it all in and clean it up
    runs_df = pd.read_csv('logs/wandb_data/'+summary_file, header=0)
    runs_df = runs_df.loc[:,~runs_df.columns.duplicated()] 
    # set which columns we will store for vizualization
    list_of_dataframes = []
    # iterate through all runs to create individual databases
    for ex in tqdm(range(len(runs_df))):
        
        # get the associated runs 
        run = api.run(user+'/'+project+'/'+runs_df.loc[runs_df.iloc[ex,0],:]['id'])
        run_df = []
        
        # iterate through all rows in online database
        base_info = {}
        for key in columns_of_interest:
            base_info.update({key:runs_df.loc[runs_df.iloc[ex,0],:][key]}) 
        run_df = run.history(samples=1e7, keys=None, x_axis="_step", pandas=(True), stream="default")
        if 'global_step' not in run_df.columns:
            continue  
        if run_df['global_step'].max() < 9e5:
            continue
        for k in base_info.keys():
            if k not in run_df.columns:
                run_df[k] = base_info[k] 
        
        # k = pd.concat([run_df, pd.DataFrame([{base_info[k] for k in run_df.columns}])])
        run_df = run_df.dropna(subset=["eval_val/reached_waypoint_num"])  
        final_log = pd.DataFrame([{k:base_info[k] for k in run_df.columns if k in columns_of_interest}])
        run_df = pd.concat([run_df,final_log], ignore_index=True)
        run_df = run_df.dropna(subset=["eval_val/reached_waypoint_num"]).sort_values(by='global_step')  
        
        # convert format to dataframe and add to our list
        list_of_dataframes.append(run_df) 
    
    # combine and then store
    wandb_records = pd.concat(list_of_dataframes)
    wandb_records.to_csv('logs/wandb_data/__full__'+summary_file)
    
    # return single data frame for vizualization
    return wandb_records

def generate_plots(user, project, summary_file='runs-summary.csv', plot_type='train', multi_agent=True, k=2):
    
    download_wandb_summary(user=user, project=project, summary_file=summary_file,
                        keyval_focus={}, reload=False)
    
    columns_of_interest = ['global_step', 'eval_val/traffic_light_violation_rate', 
                       'eval_val/collision_rate', 'eval_val/mean_episode_length',
                       'eval_val/offroad_rate', 'eval_val/mean_episode_reward', 
                       'eval_val/reached_waypoint_num', 'eval_train/traffic_light_violation_rate', 
                        'eval_train/collision_rate', 'eval_train/mean_episode_length',
                       'eval_train/offroad_rate', 'eval_train/mean_episode_reward',
                       'eval_train/speed_smoothness', 'eval_val/speed_smoothness',
                       'eval_train/reached_waypoint_num', '_n_updates', 'env-ego_only',
                       'algo', 'seed', 'n_envs', 'total_timesteps', '_runtime']
    
    wandb_records = download_wandb_records(user=user, project=project, summary_file=summary_file,
                        columns_of_interest = columns_of_interest, reload=True)
    
    wandb_records['global_step'] = wandb_records['global_step'].fillna(0)
      
    colors = {'SAC': '#009E73', 'PPO': '#0072B2', 'TD3': '#E69F00', 'A2C': '#CC79A7'}
    
    rename_metrics = {'eval_val/collision_rate': 'Collision', 
                    'eval_val/traffic_light_violation_rate': 'Traffic Light', 
                    'eval_val/offroad_rate': 'Offroad',
                    'eval_val/reached_waypoint_num': 'Waypoint #',
                    'eval_val/mean_episode_reward': 'Return',
                    'eval_val/mean_episode_length': 'Horizon',
                    'eval_train/collision_rate': 'Collision', 
                    'eval_train/traffic_light_violation_rate': 'Traffic Light', 
                    'eval_train/offroad_rate': 'Offroad',
                    'eval_train/reached_waypoint_num': 'Waypoint #',
                    'eval_train/mean_episode_reward': 'Return',
                    'eval_train/mean_episode_length': 'Horizon',
                    'eval_train/speed_smoothness': 'Smoothness', 
                    'eval_val/speed_smoothness': 'Smoothness',
                    }
    
    algos = ['SAC', 'PPO', 'TD3', 'A2C']
  
    metrics = [['eval_'+plot_type+'/collision_rate', 'eval_'+plot_type+'/traffic_light_violation_rate'],
           ['eval_'+plot_type+'/reached_waypoint_num',  'eval_'+plot_type+'/offroad_rate'],
           ['eval_'+plot_type+'/speed_smoothness', 'eval_'+plot_type+'/mean_episode_length']]
    
    x_tick_info = ([1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5], \
               ['1', '2', '3', '4', '5', '6', '7', '8', '9'])

    y_tick_info = {
        'single-agent-env':
            {
            'eval_train/collision_rate': ([0., 0.25, 0.5, 0.75, 1.0]),
            'eval_train/traffic_light_violation_rate': ([0., 0.2, 0.4, 0.6, 0.8]),
            'eval_train/offroad_rate': ([0., 0.25, 0.5, 0.75, 1.0]),
            'eval_train/reached_waypoint_num': ([0., 2.5, 5.0, 7.5, 10.0]),
            'eval_train/speed_smoothness': ([0, 1.5, 3, 4.5]),
            'eval_train/mean_episode_length': ([0, 50, 100, 150]),
            'eval_val/collision_rate': ([0., 0.25, 0.5, 0.75, 1.0]),
            'eval_val/traffic_light_violation_rate': ([0., 0.2, 0.4, 0.6, 0.8]),
            'eval_val/offroad_rate': ([0., 0.25, 0.5, 0.75, 1.0]),
            'eval_val/reached_waypoint_num': ([0., 2.5, 5.0, 7.5, 10.0]),
            'eval_val/speed_smoothness': ([0, 1.5, 3, 4.5]),
            'eval_val/mean_episode_length': ([0, 50, 100, 150])
        },
        'multi-agent-env':
            {
            'eval_train/collision_rate': ([0., 0.25, 0.5, 0.75, 1.0]),
            'eval_train/traffic_light_violation_rate': ([0., 0.2, 0.4, 0.6, 0.8]),
            'eval_train/offroad_rate': ([0., 0.25, 0.5, 0.75, 1.0]),
            'eval_train/reached_waypoint_num': ([0., 2.5, 5.0, 7.5, 10.0]),
            'eval_train/speed_smoothness': ([0, 1.5, 3, 4.5]),
            'eval_train/mean_episode_length': ([0, 50, 100, 150]),
            'eval_val/collision_rate': ([0., 0.25, 0.5, 0.75, 1.0]),
            'eval_val/traffic_light_violation_rate': ([0., 0.2, 0.4, 0.6, 0.8]),
            'eval_val/offroad_rate': ([0., 0.25, 0.5, 0.75, 1.0]),
            'eval_val/reached_waypoint_num': ([0., 2.5, 5.0, 7.5, 10.0]),
            'eval_val/speed_smoothness': ([0, 1.5, 3, 4.5]),
            'eval_val/mean_episode_length': ([0, 50, 100, 150])
        }
    } 

    y_tick_info = y_tick_info['multi-agent-env'] if multi_agent else y_tick_info['single-agent-env'] 
  
    f = plt.figure()
    fig, axs = plt.subplots(3, 2, figsize=(24, 14))

    for row in range(len(metrics)):
        for col in range(len(metrics[0])):
            for algo in algos:  
                
                # 
                metric = metrics[row][col]   
                # filter dataframe values 
                filtered_df = wandb_records[wandb_records['algo'] == algo] 
                
                filtered_df = filtered_df[filtered_df['env-ego_only'] == int(multi_agent)]
                filtered_df = filtered_df[['global_step', metric]]  
                filtered_df['global_step'] = (filtered_df['global_step'] / 25000).astype(int) * 25000  
                 
                # generate the stuff used to plot 
                col_mean = filtered_df.groupby(['global_step'])[metric].mean().values 
                col_std = filtered_df.groupby(['global_step'])[metric].std().values 
                col_x = filtered_df.groupby(['global_step'])[metric].mean().index.values
                 
                # fix the example where all values were the same std = 0 
                col_q1 = filtered_df.groupby(['global_step'])[metric].quantile(0.05).values
                col_q2 = filtered_df.groupby(['global_step'])[metric].quantile(0.95).values 

                # value was clipped in a2c -- but it has the same init policy as ppo 
                if algo == 'PPO':
                    ppo_mean, ppo_std = col_mean, col_std
                    ppo_q1, ppo_q2 = col_q1, col_q2
                elif algo == 'A2C':
                    col_x = np.insert(col_x, 0, 0)
                    col_std = np.insert(col_std, 0, ppo_std[0])
                    col_mean = np.insert(col_mean, 0, ppo_mean[0])
                    col_q1 = np.insert(col_q1, 0, ppo_q1[0])
                    col_q2 = np.insert(col_q2, 0, ppo_q2[0]) 
                 
                # smooth it (mean) 
                smooth_col_x = np.linspace(col_x.min(), col_x.max(), len(col_x)) 
                spl = make_interp_spline(col_x, col_mean, k=k)  # type: BSpline
                smooth_col_mean = spl(smooth_col_x)
                 
                # make sure the std didnt drive anything past min or max values 
                q1 = col_q1
                q2 = col_q2
                smooth_col_mean= np.minimum(q2,smooth_col_mean)
                smooth_col_mean= np.maximum(q1,smooth_col_mean)
                 
                # plot it 
                axs[row][col].plot(smooth_col_x, smooth_col_mean,
                                label='_nolegend_', linestyle='solid', color=colors[algo], alpha=0.8,
                                linewidth=4, marker=None)
                
                # add error pars
                axs[row][col].fill_between(smooth_col_x, q1, q2,
                        alpha = 0.4, label='_nolegend_', linestyle='solid', color=colors[algo]) 
                
                # fix limits
                axs[row][col].set_xlim([0, 9.49e5])
                axs[row][col].tick_params(axis='both', which='both',labelsize=22)
                
                # add x axis labels 
                if row == len(metrics)-1:
                    axs[row][col].set_xlabel('Time-steps', fontsize=24) 
                
                # y axis label
                axs[row][col].set_ylabel(rename_metrics[metric], fontsize=24) 
 
                # grid lines 
                axs[row][col].grid( )

    #
    plt.legend()

    prefix = 'multi' if not multi_agent else 'single'
    plt.savefig(prefix+'-agent-eval-'+plot_type+'.pdf', format="pdf", bbox_inches="tight")


if __name__=='__main__':
    user, project = 'iai', 'paper-experiments-rerun-sparse' 
    generate_plots(user, project, summary_file='runs-summary.csv', plot_type='train', multi_agent=True)
    generate_plots(user, project, summary_file='runs-summary.csv', plot_type='val', multi_agent=True)
    generate_plots(user, project, summary_file='runs-summary.csv', plot_type='train', multi_agent=False)
    generate_plots(user, project, summary_file='runs-summary.csv', plot_type='val', multi_agent=False)