
def is_homogenous(state):
    for row in state:
        if len(set(row)) != 1:
            return False
    return True

def train_until_solved(solver, max_timesteps):
    timestep_count=0
    solver.sphere.scramble()
    while True:
        timestep_count+=1
        solver.train_with_history()
        solver.infer(solver.sphere.get_state())
        current_state = solver.sphere.get_state()
        if is_homogenous(current_state):
            break
        if timestep_count > max_timesteps:
            break
    solver.sphere.render()
    
def run_episode(solver):
    pass