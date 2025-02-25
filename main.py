


from pdb import run

from episode import run_episode
import time

if __name__ == '__main__':
    print('Starting the CUBE')
    run_episode(max_timesteps=30,num_episodes=10000,show_plot=False,eps=1.0,train=False)

    # run_episode(max_timesteps=30,num_episodes=2,show_plot=True,eps=0.1)

