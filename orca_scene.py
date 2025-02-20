import taichi as ti
import numpy as np
import math
from PIL import Image
import utils

@ti.dataclass
class StaticObstacle:
    next:int
    previous:int
    direction:utils.vec2
    point:utils.vec2
    id:int
    convex:bool

@ti.data_oriented
class Scene:
    def __init__(self, config):
        self.window_size = config.window_size # grid's length
        self.pixel_number = config.pixel_number # grid's size  pixel_number = window_size^2
        self.batch_size = config.batch
        self.group_target = config.group_target # target position for each batch of agent

        # obstacle setting = 2
        self.obstacles_number = 0
        self.obstacles = StaticObstacle.field(shape = config.max_obstacle)

        # obstacle_setting=0 means there is no obstacle.
        # ORCA currently only supports obstacle_setting=2.
        # cannot read from img yet.
        if config.obstacle_setting == 2: 
            self.init_obstacles(config) 
        
    def add_obstacle(self, vertices,count,w,h):
        """
        Adds a new obstacle to the simulation.
        vertices (list): List of the vertices of the polygonal obstacle in counterclockwise order.
        Remarks:
            To add a "negative" obstacle, e.g. a bounding polygon around the environment, the vertices should be listed in clockwise order.
        """
        cur_count = count
        if len(vertices) < 2:
            print('error! Must have at least 2 vertices.')
        else:
            obstacleNo = count
            for i in range(len(vertices)):
                self.obstacles[cur_count].point = vertices[i]
                self.obstacles[cur_count].point[0] *= w
                self.obstacles[cur_count].point[1] *= h

                if i != 0:
                    self.obstacles[cur_count].previous = cur_count - 1
                    self.obstacles[self.obstacles[cur_count].previous].next = cur_count

                if i == len(vertices) - 1:
                    self.obstacles[cur_count].next = obstacleNo
                    self.obstacles[self.obstacles[cur_count].next].previous = cur_count

                vec = vertices[0 if i == len(vertices) - 1 else i + 1] - vertices[i]
                self.obstacles[cur_count].direction,_ = utils._normalize(vec)

                if len(vertices) == 2:
                    self.obstacles[cur_count].convex = True
                else:
                    self.obstacles[cur_count].convex = utils.left_of(
                        vertices[len(vertices) - 1 if i == 0 else i - 1],
                        vertices[i],
                        vertices[0 if i == len(vertices) - 1 else i + 1]) >= 0.0

                self.obstacles[cur_count].id = cur_count
                cur_count += 1
        return cur_count

    def init_obstacles(self, config):
        """
        mode==2:
        init StaticObstacle class from vertex
        """
        count = 0
        for items in config.obstacles:
            count = self.add_obstacle(items,count,config.WINDOW_WIDTH/config.expand_size,config.WINDOW_HEIGHT/config.expand_size)
            if count >= config.max_obstacle:
                break
        self.obstacles_number = count