import taichi as ti
import numpy as np
import time
import utils

node = ti.types.struct(g=ti.float32, h=ti.float32,f=ti.float32, father=ti.int32)

@ti.data_oriented
class AStar:
    # not support ORCA yet
    def __init__(self,enable,scene,filename=''):
        self.scene = scene
        # Record the A* unit direction vector of each pixel grid reaching the target location
        self.map = ti.Vector.field(2,dtype=ti.f32) 
        ti.root.dense(ti.i, self.scene.batch_size).dense(ti.j,self.scene.pixel_number).place(self.map)
        self.map_done = ti.field(ti.i8)
        # Record whether the shortest path of the grid has been calculated
        # 1: visited
        ti.root.dense(ti.ij,(self.scene.window_size,self.scene.window_size)).place(self.map_done)

        _list_offset = np.array([(-1, 0), (0, -1), (0, 1), (1, 0), (-1, 1), (1, -1), (1, 1), (-1, -1)])
        self.list_offset = ti.Vector.field(2, ti.i8,shape=8)
        self.list_offset.from_numpy(_list_offset)
        self.dist_list_offset = ti.field(ti.f32,shape = 8)
        self.init_list_offset()

        # Create a matrix to store nodes
        self.node_matrix = node.field(shape = scene.pixel_number)  
        # Node to be traversed.
        self.open_list = ti.field(ti.i32)
        # Node has been traversed.
        self.close_list = ti.field(ti.i32)
        ti.root.pointer(ti.i, scene.pixel_number).place(self.open_list)
        ti.root.pointer(ti.i, scene.pixel_number).place(self.close_list)

        # calculate target position
        self.target = ti.field(ti.i32,scene.batch_size)     
        self.group_target = ti.Vector.field(2,ti.i32,scene.batch_size)
        print(scene.group_target * scene.window_size)
        self.group_target.from_numpy(np.array(scene.group_target * scene.window_size))
        # if wish to calculate A*
        if enable == 1: 
            self.init_map(filename)

    @ti.kernel
    def init_list_offset(self):
        """Extension of adjacent grids during A* calculation"""
        for i in self.list_offset:
            self.dist_list_offset[i] = ((self.list_offset[i][0]**2+self.list_offset[i][1]**2)*100)**0.5

    @ti.kernel
    def init_target(self):
        """Expand the standardized target location coordinates recorded in the scene into integer subscripts according to the grid size"""
        for i in range (self.scene.batch_size):
            self.target[i] = int(self.group_target[i][0]*self.scene.window_size + self.group_target[i][1])
            assert self.scene.obstacle_exist[int(self.group_target[i][0]),int(self.group_target[i][1])] == 0
        

    def init_map(self,filepath=''):
        self.init_target()
        print(self.scene.pixel_number,"targets are nodes:",self.target)
        start = time.time()
        for batch in range (self.scene.batch_size):
            # read astar result from csv, filepath is assgined in config.py, self.astar_file
            if batch == 0:
                print("reading astar result from csv ",filepath)
                self.map = utils.read_astar_from_csv(self.map,self.scene.pixel_number,batch,filepath)
                break

            self.map_done.fill(0)
            
            for index in range(self.scene.pixel_number):
                print(index)
                # Calculate A* once for each grid
                i = int(ti.floor(index / self.scene.window_size))
                j = int(index % self.scene.window_size)
                # If in the obstacle list, skip
                if self.scene.obstacle_exist[i,j] == 0 and self.map_done[i,j] == 0:  
                    self.cal_next_loc(index,self.target[batch],batch)
            
            # you can record the first A* result as csv
            # the file name is fixed here
            utils.export_Astar(self.map,self.scene.pixel_number,0,'./data/astar_runway.csv')
        end = time.time()
        print(end-start)
    
    def cal_next_loc(self,start_pos,target_pos,batch):
        #If the starting point = the target point, return directly
        if start_pos == target_pos:
            return (0,0)

        #Initialize the reusable data structure
        # (taichi kernel does not support the definition of temporary field data structure)
        self.node_matrix.fill(0)
        self.open_list.fill(0)
        self.close_list.fill(0)
        
        self.next_loc(start_pos,target_pos,batch)

    @ti.kernel
    def next_loc(self,start_pos:ti.i32,target_pos:ti.i32,batch:ti.i32): 
        """
        Input: index of the starting location, index of the target location

        The index calculation: index = x * window_size + y, 
        (x, y are coordinates like [350,50], window_size is the grid length)

        Use A* to calculate the shortest path from the starting location to the target location

        After calculating the shortest path, record the direction vectors of all nodes on the path to the target location in self.map as the target force direction vector of the crowd       
        """
        self.open_list[0] = start_pos # Add the starting point to the open list
        open_list_len = 1  # Initialize the open list. Node to be traversed.
        close_list_len = 0  # Initialize the close list. Node has been traversed.
        target_x = ti.floor(target_pos / self.scene.window_size)
        target_y = target_pos % self.scene.window_size

        # begin loop
        while True:
            # Determine whether to stop.
            # If the target node is in the closed list, stop the loop.
            if utils.check_in_list(self.close_list,target_pos,close_list_len) == 1:
                break

            now_loc = self.open_list[0]
            place=0           
            #   （1）Get the point with the smallest f value
            for i in range(0, open_list_len): 
                if self.node_matrix[self.open_list[i]].f < self.node_matrix[now_loc].f:
                    now_loc = self.open_list[i]
                    place = i   
            #   （2）Switch to close list
            open_list_len+=-1
            self.open_list[place]=self.open_list[open_list_len]
            self.close_list[close_list_len] = now_loc
            close_list_len+=1  

            grid_x = ti.floor(now_loc / self.scene.window_size)
            grid_y = now_loc % self.scene.window_size
            for i in range(8):#   （3）For each of the 3*3 adjacent grids
                index_i = grid_x+self.list_offset[i][0]
                index_j = grid_y+self.list_offset[i][1]
                if index_i < 0 or index_i >= self.scene.window_size or index_j < 0 or index_j >= self.scene.window_size: # If out of bounds, skip
                    continue
                if self.scene.obstacle_exist[int(index_i),int(index_j)] == 1:  # If in obstacle list, skip
                    continue
                index = int(index_i * self.scene.window_size + index_j)
                if utils.check_in_list(self.close_list,index,close_list_len):  # If in close list, skip
                    continue

                # The node is not in the open list, add it, and calculate various values
                if not utils.check_in_list(self.open_list,index,open_list_len):
                    self.open_list[open_list_len] = index
                    open_list_len+=1

                    self.node_matrix[index].g = self.node_matrix[now_loc].g +self.dist_list_offset[i]
                    self.node_matrix[index].h = (abs(target_x - index_i)+abs(target_y-index_j))*10 #采用曼哈顿距离
                    self.node_matrix[index].f = (self.node_matrix[index].g +self.node_matrix[index].h)
                    self.node_matrix[index].father = now_loc
                    continue
                # If in the open list, compare and recalculate
                if self.node_matrix[index].g > self.node_matrix[index].g +self.dist_list_offset[i]:
                    self.node_matrix[index].g = self.node_matrix[index].g +self.dist_list_offset[i]
                    self.node_matrix[index].father = now_loc
                    self.node_matrix[index].f = (self.node_matrix[index].g +self.node_matrix[index].h)

        # Found the shortest path.
        # Traverse the parent nodes one by one to find the next position.
        # The fathers in the close list can be reused.
        next_move = target_pos
        current = self.node_matrix[next_move].father
        while next_move != start_pos:
            next_move = current
            current = self.node_matrix[next_move].father
            index_i = ti.floor(next_move / self.scene.window_size)
            index_j = next_move % self.scene.window_size
            i = int(ti.floor(current / self.scene.window_size))
            j = int(current % self.scene.window_size)
            # Record the normalized direction vector to the next position
            re = ti.Vector([int(index_i - i), int(index_j-j)])
            self.map[batch,current],_ = utils.normalize(re)
            self.map_done[i,j] = 1
