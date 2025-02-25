


from pdb import run

from episode import run_episode
import time

if __name__ == '__main__':
    print('Starting the CUBE')
    # run_episode(max_timesteps=20,num_episodes=100,show_plot=False,eps=1.0)
    run_episode(max_timesteps=30,num_episodes=1000,show_plot=True,eps=0.1)


