
from click import pause
from rubiks_utils.sphere import RubiksCube
import matplotlib
matplotlib.use('TkAgg')
if __name__ == "__main__":

    # make sure indecies on cube are consistent with colors in state
       # 
    cube = RubiksCube()

    # center color should be center of each face state
    cube.scramble(100)
    print(cube.points[0])
    for face in cube.get_state(): 
      print(face)
    print(cube.color_list)
    cube.init_plot()
    cube.render(pause_timer=5000)
    # regardless of the face they are on.

    # make sure actions act like they should
      # rotations work (4 rotations should return to original state)
      # taking actions and then taking the inverse should return to original state
    cube.scramble(100)
    for action in range(6):
      cube.move(action)
      cube.render(pause_timer=10)
      cube.move(action+6)
      cube.render(pause_timer=10)


    # def move(self,action):
    #     if action==0:
    #         self.move_column(coord=-0.25,plane='x',angle_degrees=90)
    #     elif action==1:
    #         self.move_column(coord=0.25,plane='x',angle_degrees=90)
    #     elif action==2:
    #         self.move_column(coord=-0.25,plane='y',angle_degrees=90)
    #     elif action==3:
    #         self.move_column(coord=0.25,plane='y',angle_degrees=90)
    #     elif action==4:
    #         self.move_column(coord=-0.25,plane='z',angle_degrees=90)
    #     elif action==5:
    #         self.move_column(coord=0.25,plane='z',angle_degrees=90)
    #     elif action==6:
    #         self.move_column(coord=-0.25,plane='x',angle_degrees=270)
    #     elif action==7:
    #         self.move_column(coord=0.25,plane='x',angle_degrees=270)
    #     elif action==8:
    #         self.move_column(coord=-0.25,plane='y',angle_degrees=270)
    #     elif action==9:
    #         self.move_column(coord=0.25,plane='y',angle_degrees=270)
    #     elif action==10:
    #         self.move_column(coord=-0.25,plane='z',angle_degrees=270)
    #     elif action==11:
    #         self.move_column(coord=0.25,plane='z',angle_degrees=270)