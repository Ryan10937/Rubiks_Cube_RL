


from solver import RubiksCubeSolver
from sphere import Sphere
from episode import train_until_solved


if __name__ == '__main__':
    print('Starting the CUBE')
    cube = RubiksCubeSolver()
    cube.sphere.init_plot()
    train_until_solved(cube,100)



