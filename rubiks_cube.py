import numpy as np
import copy
class RubiksCube:
    def __init__(self):
        self.side_length=3
        self.colors = ['W', 'R', 'B', 'G', 'Y', 'O']  # White, Red, Blue, Green, Yellow, Orange
        self.cube = self.get_cube()
        self.scramble(100)
    def __repr__(self):
        return str(self.cube)
    def get_cube(self):
        #9 faces represented by 2D arrays
        #place no more than 9 of each color
        #each color is center on one side
        #one of these sides is considered the front of the cube
        
        #make 6 3x3 arrays, assign one color to each side
        cube = np.ndarray((6,3,3),dtype=object)
        for i in range(cube.shape[0]):
            for j in range(cube.shape[1]):
                for k in range(cube.shape[2]):
                    cube[i,j,k]=self.colors[i]
                    # if j==2:
                        # cube[i,j,k]='X'

        #cube sides are in this order:
            #front
            #right
            #back
            #left
            #top
            #bottom
        #these are in no particular order, there might be a better way to do this
        self.key = {
            'front':0,
            'right':1,
            'back':2,
            'left':3,
            'top':4,
            'bottom':5,
        }
        #all neighbor directions are relative to the front face
        self.face_neighbors = {
            0:{#front
                'north':['top',0],#rotate 90 degrees zero times
                'south':'bottom',
                'east':'right',
                'west':'left',
            },
            1:{#right
                'ccw':'top',
                'cw':'bottom',
                'east':'back',
                'west':'front',
            },
            2:{#back
                'north':'bottom',
                'south':'top',
                'east':'left',
                'west':'right'
            },
            3:{#left
                'ccw':'bottom',
                'cw':'top',
                'east':'front',
                'west':'back',
            },
            4:{#top
                'north':'back',
                'south':'front',
                'ccw':'left',
                'cw':'right',
            },
            5:{#bottom
                'north':'front',
                'south':'back',
                'cw':'right',
                'ccw':'left',
            },
        }



        return np.array(cube)
        
    def scramble(self,n):
        pass

    def turn(self,direction,col_num=None,row_num=None):


        invert_direction={
            'north':'south',
            'south':'north',
            'east':'west',
            'west':'east',
            'ccw':'cw',
            'cw':'ccw',
        }
        # moving_face_to_index = {
            # 'top':[0,None],
            # 'bottom':[self.side_length-1,None],
            # 'left':[None,0],
            # 'right':[None,self.side_length-1],
            # 'back':[0,None],
            # 'top':[0,None],
        # }
        direction=invert_direction[direction]
        new_cube = copy.deepcopy(self.cube)
        print(new_cube)
        for face in range(len(self.cube)):
            if direction in self.face_neighbors[face]:
                if col_num is not None:
                    new_cube[face,:,col_num] = self.cube[self.key[self.face_neighbors[face][direction]],:,col_num]
                elif row_num is not None:
                    new_cube[face,row_num,:] = self.cube[self.key[self.face_neighbors[face][direction]],row_num,:]
                else:
                    print('row num and col num are None')
            else:
                continue
        print(new_cube)

        
