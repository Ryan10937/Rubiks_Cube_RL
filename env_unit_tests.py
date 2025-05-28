
from click import pause
from sphere import RubiksCube

if __name__ == "__main__":

    # make sure indecies on cube are consistent with colors in state
    # center color should be center of each face state
    cube = RubiksCube()
    cube.scramble(100)
    # print(cube.points[0])
    print(cube.get_state())
    print(cube.color_list)
    cube.init_plot()
    cube.render(pause_timer=5000)
    #WILO: center colors are not consistent. I think this is because i sort them by x,y,z 
    # regardless of the face they are on.


    # make sure actions act like they should
      # rotations work (4 rotations should return to original state)
      # taking actions and then taking the inverse should return to original state
