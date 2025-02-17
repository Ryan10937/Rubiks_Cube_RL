from solver import RubiksCubeSolver
from sphere import RubiksCube
def is_homogenous(state):
    for row in state:
        if len(set(row)) != 1:
            return False
    return True

def train_until_solved(solver, max_timesteps):
    timestep_count=0
    solver.sphere.scramble()
    while True:
        #render rubiks cube
        solver.sphere.render()
        timestep_count+=1


        #get action from model
        action = solver.infer(solver.sphere.get_state())

        #perform that action on the env
        solver.sphere.move(action)

        #get state
        current_state = solver.sphere.get_state()

        #break if solved or too many timesteps
        if timestep_count > max_timesteps:
            break
        if is_homogenous(current_state):
            break
    #train model with history
    solver.train_with_history()
    #save history
    solver.save_history()
        
    
def run_episode(max_timesteps,num_episodes):
    for episode in range(num_episodes):
        print('Episode: ',episode)
        cube = RubiksCubeSolver()
        cube.sphere.init_plot()
        train_until_solved(cube,max_timesteps) 