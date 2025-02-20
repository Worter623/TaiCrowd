import taichi as ti
import numpy as np
import math
from PIL import Image
import utils

@ti.data_oriented
class Scene:
    def __init__(self, config):
        self.window_size = config.window_size # grid's length
        self.pixel_number = config.pixel_number # grid's size  pixel_number = window_size^2
        self.batch_size = config.batch
        self.group_target = config.group_target

        if config.dynamic_search == 0:
            self.grid_count = ti.field(dtype=ti.i32,shape=(self.window_size, self.window_size)) 
            self.list_head = ti.field(dtype=ti.i32, shape=self.pixel_number)
            self.list_cur = ti.field(dtype=ti.i32, shape=self.pixel_number)
            self.list_tail = ti.field(dtype=ti.i32, shape=self.pixel_number)
            self.column_sum = ti.field(dtype=ti.i32, shape=self.window_size, name="column_sum") 
            self.prefix_sum = ti.field(dtype=ti.i32, shape=(self.window_size, self.window_size), name="prefix_sum")
            self.particle_id = ti.field(dtype=ti.i32, shape=self.pixel_number, name="particle_id")
        else:
            self.grid_count = ti.field(ti.i32, self.pixel_number) 
            self.grid_matrix = ti.field(ti.i32)
            self.block = ti.root.dense(ti.i, self.pixel_number)
            pixel = self.block.dynamic(ti.j, config.N)
            pixel.place(self.grid_matrix)

        # obstacle_setting = 1
        self.obstacle_exist = ti.field(ti.f32, shape=(self.window_size, self.window_size))  
        self.grid_pos = ti.Vector.field(2, ti.float32)  
        ti.root.pointer(ti.ij,(self.window_size,self.window_size)).place(self.grid_pos)

        if config.obstacle_setting == 1:
            if config.im is None:
                self.init_obstacles(config)
            else:
                self.init_from_pic(config)
            self.make_obstacle_pos()

    def init_obstacles(self, config):
        """
        mode==1:
        Input an list of (startx, starty, endx, endy) as start and end of a line
        mode==2:
        init StaticObstacle class from vertex
        """
        if config.obstacle_setting == 1:
            _obstacles = []
            for startx, starty, endx, endy in config.obstacles:
                samples = int(np.linalg.norm(
                    (startx - endx, starty - endy)) * config.resolution)
                line = np.array(
                    list(
                        zip(np.linspace(startx, endx, samples),
                            np.linspace(starty, endy, samples))
                    )
                )
                _obstacles.append(line)
            _obstacles = np.vstack(_obstacles)
            for obstacle in _obstacles:
                x = math.floor(obstacle[0] * self.window_size)
                y = math.floor(obstacle[1] * self.window_size)
                if x >= self.window_size:
                    x = self.window_size-1
                if y >= self.window_size:
                    y = self.window_size-1
                if self.obstacle_exist[x,y] == 0:
                    self.obstacle_exist[x,y] = 1
           
    def init_from_pic(self,config):
        """
        Read the image, binarize it, and automatically identify obstacles.
        The image coordinate system read by PIL is different from that of Taichi.
        It needs to be rotated 90° counterclockwise.
        """
        im = np.array(Image.open(config.path).convert('1'))
        im = np.rot90(im, -1)
        ti_im = ti.field(ti.i8, shape=(config.WINDOW_WIDTH,config.WINDOW_HEIGHT))
        ti_im.from_numpy(im.astype(np.int8))
        self.make_pos_with_pic(ti_im,config.WINDOW_WIDTH,config.WINDOW_HEIGHT)       

    @ti.kernel
    def make_pos_with_pic(self,ti_im:ti.template(),WINDOW_WIDTH:ti.i32,WINDOW_HEIGHT:ti.i32):
        """
        For each pixel of the incoming image, determine whether it is an obstacle
        (0 represents the black grid in the image)
        If it is, de-sampling:
        calculate the corresponding grid range after it is expanded through the x coordinate range (x, x+1) and y coordinate range (y, y+1) 
        Mark the grid in this range as having an obstacle (self.obstacle_exist[x,y] = 1)
        """
        for i,j in ti_im:
            if ti_im[i,j] == 0:
                start_i = ti.round(i / WINDOW_WIDTH * self.window_size)
                end_i = ti.round((i+1) / WINDOW_WIDTH * self.window_size)
                start_j = ti.round(j / WINDOW_HEIGHT* self.window_size)
                end_j = ti.round((j+1) / WINDOW_HEIGHT * self.window_size)
                for x in range (start_i,end_i):
                    for y in range(start_j,end_j):
                        self.obstacle_exist[x,y] = 1

    @ti.kernel
    def make_obstacle_pos(self):
        """
        For obstacle grids, calculate the center position of all grids for subsequent calls. 
        For non-obstacle grids, A* its direction vector. 
        Ideally, people will not hit the wall, so there is no need to access the A* result of the obstacle grid.
        """
        for grid_index in range(self.pixel_number):
            i = int(ti.floor(grid_index/ self.window_size))
            j = int(grid_index % self.window_size)
            if self.obstacle_exist[i,j] == 1 :
                self.grid_pos[i,j] = ti.Vector([i,j])/ self.window_size