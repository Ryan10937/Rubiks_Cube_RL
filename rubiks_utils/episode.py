from pdb import run
from rubiks_utils.solver import RubiksCubeSolver
import copy
def is_homogenous(state):
    for row in state:
        if len(set(row)) != 1:
            return False
    return True

def train_until_solved(solver, max_timesteps,show_plot=True,train=True,num_shuffle=100):
    timestep_count=0
    solver.sphere.scramble(n=num_shuffle)
    #train model with history
    if train:
        solver.train_with_history()
    while True:
        if show_plot:
            #render rubiks cube
            solver.sphere.render()
        timestep_count += 1

        #get action from model
        action = solver.infer(solver.sphere.get_state())

        #perform that action on the env
        solver.sphere.step(action)

        #break if solved or too many timesteps
        if timestep_count > max_timesteps:
            break
        if solver.sphere.done==True:
            print('Solved')
            break

    if show_plot:
        solver.sphere.close_plot()
    return solver.sphere.reward_history
        
    
def run_episode(max_timesteps,num_episodes,show_plot=True,eps=0.1,eps_decay=0.995,train=True,num_shuffle=100):
    for episode in range(num_episodes):
        print('Episode: ',episode)
        cube = RubiksCubeSolver(show_plot=show_plot,eps=eps,eps_decay=eps_decay,episode=episode)
        if show_plot:
            cube.sphere.init_plot()
        train_until_solved(cube,max_timesteps,show_plot=show_plot,train=train,num_shuffle=num_shuffle)
    cube.sphere.plot_reward_history()
    return cube.sphere.reward_history

def evaluate(max_timesteps,num_episodes,show_plot=True,eps=0.1,eps_decay=0.995,train=True,num_shuffle=100):
    for episode in range(num_episodes):

        #create cubes
        cube = RubiksCubeSolver(show_plot=show_plot,eps=eps,eps_decay=eps_decay,episode=episode)
        cube.sphere.scramble(n=num_shuffle)
        cube_state_copy = copy.copy(cube.sphere.points)
        duplicate_cube = RubiksCubeSolver(show_plot=show_plot,eps=eps,eps_decay=eps_decay,episode=episode)
        duplicate_cube.sphere.points = cube_state_copy


        model_reward_history = train_until_solved(cube,max_timesteps,show_plot=show_plot,train=train,num_shuffle=0)
        random_reward_history = train_until_solved(duplicate_cube,max_timesteps,show_plot=show_plot,train=train,num_shuffle=0)
        
        #print results
        print('Episode: ', episode)
        print('Model total Reward',sum(model_reward_history))
        print('Random total Reward',sum(random_reward_history))


def create_training_data(max_timesteps=30, num_episodes=100, num_shuffle=100,upload_ip=''):
    for episode in range(num_episodes):
        print('Episode: ', episode)
        cube = RubiksCubeSolver(episode=episode)
        timestep_count = 0
        cube.sphere.scramble(n=num_shuffle)   
        while True:
            timestep_count += 1

            #get action from model
            action = cube.generate_history(cube.sphere.get_state())

            #perform that action on the env
            cube.sphere.step(action)

            #break if solved or too many timesteps
            if timestep_count > max_timesteps:
                break
            if cube.sphere.done==True:
                print('Solved')
                break
        
        cube.save_history()
    assert upload_ip != '', 'Upload IP must be provided to upload history'
    cube.upload_history(upload_ip)

def train_using_history(num_episodes):
    for epoch in range(num_episodes):
        print('Epoch: ', epoch)
        #load history
        cube = RubiksCubeSolver(show_plot=False,episode=epoch)
        #train model
        cube.train_with_history()