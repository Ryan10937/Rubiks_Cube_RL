import enum
import re
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D

class Sphere:
    def __init__(self, radius=1.0, center=(0, 0, 0)):
        self.radius = radius
        self.center = np.array(center)
        self.points = []
        self.place_points_on_cube_faces(grid_range=0.3)
        self.round_points()
        self.color_list = self.create_color_list()

    def init_plot(self):
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Equal aspect ratio
        self.ax.set_box_aspect([1,1,1])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
    def create_color_list(self):
        def color_conditions(coords):
            face_depth_dict = {
                 '0_-1':'red',
                 '0_1':'green',
                 '1_-1':'blue',
                 '1_1':'yellow',
                 '2_-1':'orange',
                 '2_1':'purple',
                             }
            plane_ind = np.argmax(np.abs(coords))
            plane_depth = 1 if 0<coords[plane_ind] else -1
            return str(face_depth_dict[str(plane_ind)+'_'+str(plane_depth)])
        
        return [color_conditions(coords) for coords in self.points]
    def render(self):
        self.ax.cla()
        # Draw the sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.radius * np.outer(np.cos(u), np.sin(v)) + self.center[0]
        y = self.radius * np.outer(np.sin(u), np.sin(v)) + self.center[1]
        z = self.radius * np.outer(np.ones(np.size(u)), np.cos(v)) + self.center[2]
        self.ax.plot_surface(x, y, z, color='b', alpha=0.1)
        # Draw the points
        if self.points:
            points_array = np.array(self.points)
            self.ax.scatter(
                points_array[:, 0], 
                points_array[:, 1], 
                points_array[:, 2], 
                c=self.color_list, 
                s=20
            )
        plt.show(block=False)
        plt.pause(0.1)

    def place_points_on_cube_faces(self, grid_range=0.5):
        """Place 9 points on each of the 6 cube-like faces of the sphere.
           Use `grid_range` to control how tightly the points are clustered (0.1 to 1.0)."""
        # Grid from -grid_range to grid_range in 3 steps (local coordinates)
        steps = np.linspace(-grid_range, grid_range, 3)
        
        # For each face (x+, x-, y+, y-, z+, z-)
        for face in ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']:
            for a in steps:
                for b in steps:
                    # Get local coordinates on the cube face
                    if face == 'x+':
                        x, y, z = 1, a, b
                    elif face == 'x-':
                        x, y, z = -1, a, b
                    elif face == 'y+':
                        x, y, z = a, 1, b
                    elif face == 'y-':
                        x, y, z = a, -1, b
                    elif face == 'z+':
                        x, y, z = a, b, 1
                    elif face == 'z-':
                        x, y, z = a, b, -1
                    
                    # Normalize to project onto the sphere
                    norm = np.linalg.norm([x, y, z])
                    x_norm = x / norm
                    y_norm = y / norm
                    z_norm = z / norm
                    
                    # Scale by radius and shift to center
                    x_global = self.radius * x_norm + self.center[0]
                    y_global = self.radius * y_norm + self.center[1]
                    z_global = self.radius * z_norm + self.center[2]
                    
                    self.points.append((x_global, y_global, z_global))

    def rotate_point_around_axis(self,point_idx, plane_normal, angle_degrees):
        """
        Rotate a point on a sphere around a line orthogonal to a specified plane.

        Parameters:
            point (list or np.array): The coordinates of the point [x, y, z].
            plane_normal (list or np.array): The normal vector of the plane [a, b, c].
            angle_degrees (float): The angle of rotation in degrees.

        Returns:
            np.array: The new coordinates of the point after rotation.
        """
        point = self.points[point_idx]
        plane_normal = [1 if plane_normal == i else 0 for i in range(3)]
        # plane_normal = [0,0,1]
        # Convert angle to radians
        angle_radians = np.radians(angle_degrees)
        
        # Normalize the plane normal vector (this is the axis of rotation)
        axis = np.array(plane_normal)
        # axis = axis / np.linalg.norm(axis)
        
        # Extract components of the axis
        ax, ay, az = axis
        
        # Compute rotation matrix
        cos_theta = np.cos(angle_radians)
        sin_theta = np.sin(angle_radians)
        
        rotation_matrix = np.array([
            [cos_theta + ax**2 * (1 - cos_theta), ax * ay * (1 - cos_theta) - az * sin_theta, ax * az * (1 - cos_theta) + ay * sin_theta],
            [ay * ax * (1 - cos_theta) + az * sin_theta, cos_theta + ay**2 * (1 - cos_theta), ay * az * (1 - cos_theta) - ax * sin_theta],
            [az * ax * (1 - cos_theta) - ay * sin_theta, az * ay * (1 - cos_theta) + ax * sin_theta, cos_theta + az**2 * (1 - cos_theta)]
        ])
        
        # Apply the rotation matrix to the point
        rotated_point = np.dot(rotation_matrix, point)
        
        # Normalize the point to ensure it stays on the sphere
        rotated_point = rotated_point / np.linalg.norm(rotated_point)
        self.round_points()
        self.points[point_idx] = rotated_point

    def show_point_distribution(self):
        x = [p[0] for p in self.points]
        y = [p[1] for p in self.points]
        z = [p[2] for p in self.points]
        plt.figure()
        plt.subplot(3,1,1)
        plt.hist(x)
        plt.subplot(3,1,2)
        plt.hist(y)
        plt.subplot(3,1,3)
        plt.hist(z)

        plt.show()
        #all points are on either -0.25 or 0.25 and on either -1.0, 1.0\
        # there are also points on 0.0 and their respective face's radius length
    def move_column(self,coord,plane,angle_degrees):
        planes_dict={'x':0,'y':1,'z':2}
        assert plane in planes_dict.keys()
        for i,pt in enumerate(self.points):
            if pt[planes_dict[plane]]==coord:
                self.rotate_point_around_axis(point_idx=i,plane_normal=planes_dict[plane],angle_degrees=angle_degrees)
            #rotate the face on the side of the rotation
            elif pt[planes_dict[plane]]==1.0 and coord>0.0:
                self.rotate_point_around_axis(point_idx=i,plane_normal=planes_dict[plane],angle_degrees=angle_degrees)
            elif pt[planes_dict[plane]]==-1.0 and coord<0.0:
                self.rotate_point_around_axis(point_idx=i,plane_normal=planes_dict[plane],angle_degrees=angle_degrees)

    def round_points(self):
        #round to +/-0.25 or +/-1.0, or 0.0
        def round_point(point):#this might change the type of the point
            rounding_numbers = np.array([0.0,-0.25,0.25,-1.0,1.0])
            new_point = []
            for dim in range(len(point)):
                new_point.append(rounding_numbers[np.argmin(abs(rounding_numbers-point[dim]))])
            return new_point
        for i,pt in enumerate(self.points):
            self.points[i] = round_point(pt)

    def scramble(self,n=100):

        for i in range(n):
            self.move(random.randint(0,11))
    
    def group_points_by_face(self,points):
        self.face_colors = {
            '0_-1':'none',
            '0_1':'none',
            '1_-1':'none',
            '1_1':'none',
            '2_-1':'none',
            '2_1':'none'
        }
        face_dict = {
            '0_-1':[],
            '0_1':[],
            '1_-1':[],
            '1_1':[],
            '2_-1':[],
            '2_1':[]
        }

        for pt,color in zip(points,self.color_list):
            face = str(np.argmax([abs(p) for p in pt]))+'_'+str(1 if pt[np.argmax([abs(p) for p in pt])]>0 else -1)
            face_dict[face].append(color)
            if 0.25 not in pt and -0.25 not in pt:
                self.face_colors[face] = color
        return face_dict
    def get_state(self):
        
        face_point_groups = self.group_points_by_face(self.points)
        state = [lst for key,lst in face_point_groups.items()]
        self.color_to_index = [self.face_colors[key] for key,lst in face_point_groups.items()]
        return np.array(state)
    
    def get_reward(self,state):
        reward = 0
        faces_completed = 0
        for i,lst in enumerate(state):
            face_reward=0
            face_completed = False
            for pt in lst:
                if pt==self.color_to_index[i]:
                    reward+=1
                    face_reward+=1
            if face_reward==9:
                face_completed = True
                reward+=10
                faces_completed+=1
        if faces_completed==6:
            reward += 1000
        return reward
    def move(self,action):
        if action==0:
            self.move_column(coord=-0.25,plane='x',angle_degrees=90)
        elif action==1:
            self.move_column(coord=0.25,plane='x',angle_degrees=90)
        elif action==2:
            self.move_column(coord=-0.25,plane='y',angle_degrees=90)
        elif action==3:
            self.move_column(coord=0.25,plane='y',angle_degrees=90)
        elif action==4:
            self.move_column(coord=-0.25,plane='z',angle_degrees=90)
        elif action==5:
            self.move_column(coord=0.25,plane='z',angle_degrees=90)
        elif action==6:
            self.move_column(coord=-0.25,plane='x',angle_degrees=270)
        elif action==7:
            self.move_column(coord=0.25,plane='x',angle_degrees=270)
        elif action==8:
            self.move_column(coord=-0.25,plane='y',angle_degrees=270)
        elif action==9:
            self.move_column(coord=0.25,plane='y',angle_degrees=270)
        elif action==10:
            self.move_column(coord=-0.25,plane='z',angle_degrees=270)
        elif action==11:
            self.move_column(coord=0.25,plane='z',angle_degrees=270)
        self.round_points()
