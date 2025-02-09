


from rubiks_cube import RubiksCube
from sphere import Sphere


if __name__ == '__main__':
    print('Starting the CUBE')
    # cube = RubiksCube()
    # # cube.turn('south',col_num=0)
    # cube.turn('cw',row_num=2)

    sphere = Sphere()

    
    sphere.move_column(coord=-0.25,plane='x',angle_degrees=270)
    # sphere.move_column(coord=-0.25,plane='z',angle_degrees=270)
    # sphere.move_column(coord=-0.25,plane='z',angle_degrees=270)
    # def rotate_point_around_axis(self,point_idx, plane_normal, angle_degrees):
    # sphere.rotate_point_around_axis(10,plane_normal=[1,0,0],angle_degrees=90)
    sphere.render()