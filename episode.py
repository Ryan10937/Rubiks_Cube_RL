
def is_homogenous(state):
    for row in state:
        if len(set(row)) != 1:
            return False
    return True

def train_until_solved(solver, max_timesteps):
    timestep_count=0
    solver.sphere.scramble()
    while True:
        print(solver.sphere.get_state())
        solver.sphere.render()
        timestep_count+=1
        solver.train_with_history()
        action = solver.infer(solver.sphere.get_state())
        solver.sphere.move(action)
        current_state = solver.sphere.get_state()
        if timestep_count > max_timesteps:
            break
        if is_homogenous(current_state):
            break
        
    
def run_episode(solver):
    pass 