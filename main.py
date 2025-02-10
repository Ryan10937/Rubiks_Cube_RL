


from solver import RubiksCubeSolver
from sphere import Sphere
from episode import train_until_solved


if __name__ == '__main__':
    print('Starting the CUBE')
    cube = RubiksCubeSolver()
    # cube.sphere.render()
    # cube.sphere.move_column(coord=-0.25,plane='x',angle_degrees=90)
    # cube.sphere.render()
    # cube.sphere.move_column(coord=0.25,plane='z',angle_degrees=270)
    # cube.sphere.render()
    # cube.sphere.move_column(coord=-0.25,plane='y',angle_degrees=90)
    # cube.sphere.render()
    
    print('cube state:', cube.sphere.get_state())
    train_until_solved(cube,1000)



