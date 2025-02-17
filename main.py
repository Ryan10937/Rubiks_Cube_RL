


from pdb import run

from episode import run_episode
import time

if __name__ == '__main__':
    print('Starting the CUBE')
    run_episode(max_timesteps=1000,num_episodes=50)


