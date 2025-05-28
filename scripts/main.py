


from ast import arg
from pdb import run

from rubiks_utils.episode import create_training_data,train_using_history,evaluate
import time
import argparse

if __name__ == '__main__':
# 192.168.1.213
    parser = argparse.ArgumentParser(description='Rubik\'s Cube Solver')
    parser.add_argument('--generate_data', action='store_true', help='generate training data',required=False,)
    parser.add_argument('--upload_ip',type=str, help='ip for uploading generated data',required=False,)
    parser.add_argument('--train', action='store_true', help='train the model',required=False)
    parser.add_argument('--evaluate', action='store_true', help='evaluate the model against an e-greedy policy',required=False)
    # parser.add_argument('--infer', type=int, default=1000, help='infer on specified state',required=False)
    parser.add_argument('--num_episodes', type=int, default=100, help='number of episodes to run',required=False)
    parser.add_argument('--num_timesteps', type=int, default=1000, help='number of timesteps to run',required=False)
    parser.add_argument('--num_shuffle', type=int, default=100, help='number of times to shuffle the cube',required=False)
    parser.add_argument('--show_plot', action='store_true', help='show plot of the cube',required=False)
    parser.add_argument('--eps', type=float, default=0.01, help='epsilon for e-greedy policy',required=False)
    parser.add_argument('--eps_decay', type=float, default=0.995, help='epsilon decay rate',required=False)
    print('Starting the CUBE')


    args = parser.parse_args()
    if args.generate_data:
        print('Generating training data')
        create_training_data(max_timesteps=args.num_timesteps, num_episodes=args.num_episodes, num_shuffle=args.num_shuffle,upload_ip=args.upload_ip)
    if args.train:
        print('Training the model')
        train_using_history(num_episodes=args.num_episodes)
    if args.evaluate:
        print('Evaluating the model')
        evaluate(max_timesteps=args.num_timesteps, num_episodes=args.num_episodes, show_plot=args.show_plot, eps=args.eps, eps_decay=args.eps_decay, train=False, num_shuffle=args.num_shuffle)
        
    #history generation
    # for i in range(30):
        # create_training_data(max_timesteps=30, num_episodes=100, num_shuffle=1)
    #     run_episode(max_timesteps=30,num_episodes=100,show_plot=False,eps=1.0,eps_decay=1.0,train=False,num_shuffle=i)
    # run_episode(max_timesteps=30,num_episodes=1000,show_plot=False,eps=1.0,eps_decay=0.995,train=True,num_shuffle=3)

    # run_episode(max_timesteps=30,num_episodes=2,show_plot=True,eps=0.01,eps_decay=0.995,train=False,num_shuffle=7)
    # run_episode_without_history_saving(max_timesteps=30,num_episodes=100,show_plot=False,eps=0.0,eps_decay=1.0,train=False,num_shuffle=20)
    
    

