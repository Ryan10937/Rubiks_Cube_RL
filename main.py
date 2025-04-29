


from pdb import run

from episode import run_episode,run_episode_without_history_saving
import time

if __name__ == '__main__':
    print('Starting the CUBE')
    #history generation
    # for i in range(30):
    #     run_episode(max_timesteps=30,num_episodes=100,show_plot=False,eps=1.0,eps_decay=1.0,train=False,num_shuffle=i)
    # run_episode(max_timesteps=30,num_episodes=1000,show_plot=False,eps=1.0,eps_decay=0.995,train=True,num_shuffle=3)

    # run_episode(max_timesteps=30,num_episodes=2,show_plot=True,eps=0.01,eps_decay=0.995,train=False,num_shuffle=7)
    run_episode_without_history_saving(max_timesteps=30,num_episodes=100,show_plot=False,eps=0.0,eps_decay=1.0,train=False,num_shuffle=20)
    

