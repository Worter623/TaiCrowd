import numpy as np
import taichi as ti
import random
import math
from PIL import Image
from numpy.random import default_rng
import utils
import json

@ti.data_oriented
class Config:
    def __init__(self):
        # Whether to use A* to pre-calculate each goal
        # 1 means yes
        self.Astar = 0
        # Whether to use static or dynamic memory allocation when neighborhood searching
        # 0 means using static memory allocation method
        # 1 means using dynamic memory allocation method
        self.dynamic_search = 0
        self.analysis = 0

        # simulation step
        self.sim_step = 0.2  
        # Predefined window size
        # If reading an image, the window size will be adjust to the image resolution
        self.WINDOW_HEIGHT = 200 
        self.WINDOW_WIDTH = 200 
        self.expand_size = 1
        # How many grids each edge is divided into
        # When this value is increased, it affects the neighborhood search
        # This value currently also represents the size of the map (meters)
        self.window_size = 200
        self.pixel_number = self.window_size ** 2
        # Neighboorhood Search radius 
        # eg.search_radius=2 means searching in a 5*5 grid
        self.search_radius = 20 
        # When the value of gent/obstacle reaches max_neighbour, it will no longer be calculated to prevent crowd density or obstacle density from oscillating.
        self.max_neighbour = 10 
        # If agent are close enough to its target, stop
        self.stop_radius = 0.01 
        # the number of groups
        self.set_batch = 5

        self.path = ""
        self.im = None
        self.astar_file = ""

        # social force params

        # desired force
        self.relaxation_time = 0.5
        # Deceleration threshold
        self.goal_threshold = 0.01 
        self.desired_factor = 10

        # fij
        self.social_radius = 0.01
        self.lambda_importance = 2.0
        self.gamma = 0.35
        self.n_fij = 2
        self.n_prime = 3
        self.fij_factor = 10

        # obstacle force
        self.shoulder_radius = 0.001
        self.sigma = 0.2
        # When obstacles/agents are densely packed, may need to reduce the value to achieve better results.
        self.obstacle_threshold = 0.02
        self.obstacle_factor = 10

        # whether to add gender in simulation
        self.use_gender = 0
        self.gender_gap = 1.5

        # steering params

        # flocking only added when self.flocking=1
        self.flocking = 0 
        self.flocking_radius = 0.15
        self.alignment_factor = 10
        self.separation_factor = 10
        self.cohesion_factor = 10

        # steering force only added when self.steering_force=1
        self.steering_force = 0
        self.flee_factor = 1
        self.seek_factor = 0
        self.arrival_factor = 1
        self.behind_distance = 0.01
        self.slowing_distance = 0.8
        # which agent is the leader
        self.leader_index = 0 
        self.leader_following_factor = 5
        # obstacle_avoidance_factor
        self.oa_factor = 15 
        self.obstacle_radius = 1/self.window_size
        self.oa_forsee_factor = 2
        self.oa_radius = self.oa_forsee_factor*self.obstacle_radius
        
        # ORCA params

        # The default minimal amount of time for which a new agent's velocities that are computed by the simulation are safe with respect to other agents. 
        # The larger this number, the sooner an agent will respond to the presence of other agents, but the less freedom the agent has in choosing its velocities. Must be positive.
        self.timeHorizon = 20 
        # The default minimal amount of time for which a new agent's velocities that are computed by the simulation are safe with respect to obstacles.
        self.timeHorizonObst = 10 
        self.ORCA_constraints = self.max_neighbour + self.max_neighbour + 2
        # to avoid in how many time step
        self.invTimeStep = 20 
        # influence disk per agent=self.ORCA_radius/2
        self.ORCA_radius = 5 
        self.obstacle_orcaradius = 5

        # 1: use the density filter; 0: not use
        self.dense_sense = 1
        # density filter params
        self.dense_theta = 1
        self.dense_k = 1
        self.dense_p = 1
        
        # 0: no obstacle; 
        # 1: obstacle is in grid,set by self.obstacles or read from img; 
        # 2: obstacle is in Convex polygon
        self.obstacle_setting = 0 
        # maximum obstacle vertices
        self.max_obstacle = 60
        # How many obstacle points will an obstacle line segment be divided into?
        # usually, as long as this value is larger than window_size, it's ok.
        self.resolution = self.window_size+20
        # eg.[[0, 0, 1, 0],[0, 1, 1, 1],[1,0,1,1],[0,0,0,1],[0.5,0,0.5,0.45],[0.5,1,0.5,0.55]] 
        # (startx, starty, endx, endy)
        self.obstacles = []
        # manually set these when obstacle_setting=2, 
        # see function init_obstacle_line() and value == "block" in function set_scene()
        self.obstacles_x = []
        self.obstacles_y = []

        # render settings
        self.people_radius = 3
        # Whether to draw a force triangle in the dot representing the person
        self.draw_triangle = 0 
        self.triangle_size = self.people_radius / self.WINDOW_WIDTH 
        
        # people number
        self.N = 200  
        self.max_speed_factor = 0.4   
        self.vel_0 = np.zeros(dtype=np.float32, shape=(self.N, 2))
        self.pos_0 = np.zeros(dtype=np.float32,shape=(self.N, 2)) 
        self.desiredpos_0 = np.zeros(dtype=np.float32, shape=(self.N, 2))
        # expected speed for each person, in m/s
        self.desiredvel = np.zeros(dtype=np.float32, shape=self.N)   
        # The random number range of the speed difference value of the crowd
        self.speed_range = 15 

        # write each person's target coordinates in the corresponding position here
        self.group_target = np.zeros(shape=(self.N, 2), dtype=np.float32)
        # By default, everyone's target initial velocity is different
        self.group_vel = np.zeros(shape=(self.N, 2), dtype=np.float32) 
        
        # Number of people from different batch
        self.group = [] 
        self.batch = self.N
        # 0 represents male, 1 represents female
        self.gender = [] 

        # state machine for steering; 1:state_machine on
        self.state_machine = 1 
        self.state_num = 2
        self.state_target = [self.group_target,np.array([[0.5, 0.5],[0.5, 0.5]])]
        self.state_velfactor = [0.4,0.8]
          
    def post_init(self):
        """
        init parameters after finishing reading all modifiable parameters.
        This is the function called after reading the ini file to overwrite various parameters in config.py
        """
        self.ORCA_constraints = self.max_neighbour + self.max_neighbour + 2
        self.obstacle_radius = 1/self.window_size
        self.oa_radius = self.oa_forsee_factor*self.obstacle_radius
        self.fill_desiredvel()
        if self.state_machine == 0:
            self.state_num = 1

    def fill_gender(self):
        """
        fixed now, can't be modified.
        The first batch were all men, the second batch were all women
        """
        self.gender = []
        for i in range (self.batch): 
            for _ in range (self.group[i]):
                if i == 0:
                    self.gender.append(0)
                else:
                    self.gender.append(1)

    def fill_pos0_from_csv(self,filepath):
        """
        fill initial position from csv
        csv filename is assigned in ini file(section[SCENE] item[pos_path])
        """
        print("reading initial position from csv ",filepath)
        self.pos_0,self.batch,self.group = utils.read_pos0_from_csv(self.pos_0,filepath)

    def read_img(self):
        """
        read img and fill self.im
        the path of img is self.path, must be assigned before calling this function
        """          
        im = np.array(Image.open(self.path).convert('L'))
        # The image coordinate system read by PIL is different from that of Taichi.
        # It needs to be rotated 90° counterclockwise.
        im = np.rot90(im,-1) 
        self.WINDOW_WIDTH = im.shape[0]
        self.WINDOW_HEIGHT = im.shape[1]
        self.im = ti.field(ti.f32, shape=im.shape)
        self.im.from_numpy(im)
        print("reading pic from {}, pic resolution in pixel is {}*{}".format(self.path,self.WINDOW_WIDTH,self.WINDOW_HEIGHT))

    def init_obstacle_line(self):
        """
        fill self.obstacle_x and self.obstacle_y
        so that when obstacle_setting = 2(init with convex mode)and init without picture,
        GUI can show the obstacle line
        """
        self.obstacles_x = []
        self.obstacles_y = []
        for items in self.obstacles:
            for i in range(len(items)-1):
                self.obstacles_x.append([items[i][0],items[i][1]])
                self.obstacles_y.append([items[i+1][0],items[i+1][1]])
            self.obstacles_x.append(items[len(items)-1])
            self.obstacles_y.append(items[0])

        self.obstacles_x = np.array(self.obstacles_x)
        self.obstacles_y = np.array(self.obstacles_y)

    def fill_group(self):
        """Fill batch according to N total number of people and group_target. The number of people in each batch is the same."""
        self.group = []
        self.batch = self.group_target.shape[0] #有几批人
        sum = 0
        batch = math.floor(self.N/self.batch)
        for _ in range (self.batch-1):
            self.group.append(batch)
            sum+=batch
        self.group.append(self.N-sum)

    def shuffle_group(self,target):
        """
        Generate random groups based on custom batch number, group_target remains unchanged. 
        The position of each agent is defined here
        """
        self.group = []
        sum = 0
        count = 1

        # example: Generate self.batch-random.randint(0,5) groups, and the rest of the people are in the same group
        # for _ in range (self.set_batch-random.randint(0,5)):  
        # for _ in range (self.set_batch):
        #     temp_num = random.randint(2,5)
        #     sum += temp_num
        #     count += 1
        #     self.group.append(temp_num) #generate group with 2-5 people per group
        # self.group.append(self.N-sum)

        # example v2: A maze group ini file
        A_maze_group_list=[4,2,4,3,3]
        for i in range(5):
            self.group.append(A_maze_group_list[i])
            sum+=A_maze_group_list[i]
            count += 1
        self.group.append(self.N-sum)

        self.batch = count
        self.group_target = np.array([target]*self.batch) #每批人的目的地
        self.group_vel = np.array([[0.01, 0.]]*self.batch) #每批人的初始速度

        sum_idx = 0
        for count_batch in range(self.batch):
            # first, for small group
            if self.group[count_batch] <= 5: 
                # example v1: escape scene position
                # rdm = random.randint(0,2)
                # if rdm == 0:
                #     people_count[0] += self.group[count_batch]
                #     for batch_idx in range(self.group[count_batch]):
                #         self.pos_0[sum_idx+batch_idx][0] = random.uniform(0, 0.05)
                #         self.pos_0[sum_idx+batch_idx][1] = 0.5 + random.uniform(-0.02, 0.02)
                # elif rdm == 1:
                #     people_count[1] += self.group[count_batch]
                #     for batch_idx in range(self.group[count_batch]):
                #         self.pos_0[sum_idx+batch_idx][0] = 0.85 + random.uniform(0, 0.05)
                #         self.pos_0[sum_idx+batch_idx][1] = 0.5 + random.uniform(-0.02, 0.02)
                # else:
                #     people_count[2] += self.group[count_batch]
                #     for batch_idx in range(self.group[count_batch]):
                #         self.pos_0[sum_idx+batch_idx][0] = 0.5 + random.uniform(-0.05, 0.05)
                #         self.pos_0[sum_idx+batch_idx][1] = 0.6 + random.uniform(-0.05, 0.05)

                # example v2: maze scene position
                rdm = random.uniform(-0.1,0.1)
                for batch_idx in range(self.group[count_batch]):
                    self.pos_0[sum_idx+batch_idx][0] = 0.85 + rdm + random.uniform(-0.015, 0.015)
                    self.pos_0[sum_idx+batch_idx][1] = 0.5 + rdm + random.uniform(-0.015, 0.015)
                sum_idx += self.group[count_batch]

        # then, for large group

        # example v1: escape scene position
        # num_per_corner = self.N // 3 
        # for i in range(3):
        #     gen_people_num = num_per_corner - people_count[i] if i!=2 else self.N-sum_idx
        #     for corner_idx in range (gen_people_num):
        #         self.pos_0[sum_idx+corner_idx][1] = 0.5 + random.uniform(-0.05, 0.05)
        #         if i == 0:
        #             self.pos_0[sum_idx+corner_idx][0] = random.uniform(0, 0.15)
        #         elif i == 1:
        #             self.pos_0[sum_idx+corner_idx][0] = 0.75 + random.uniform(0, 0.15)
        #         else:
        #             self.pos_0[sum_idx+corner_idx][0] = 0.5 + random.uniform(-0.05, 0.05)
        #             self.pos_0[sum_idx+corner_idx][1] = 0.6 + random.uniform(-0.05, 0.2)
        #     sum_idx += gen_people_num

        # example v2: maze scene position
        gen_people_num = self.N - sum_idx
        for corner_idx in range (gen_people_num):
            self.pos_0[sum_idx+corner_idx][0] = random.uniform(0.75, 1)
            self.pos_0[sum_idx+corner_idx][1] = random.uniform(0.35, 0.65)
        print("generate group:",self.batch,self.group)

    def fill_desiredvel(self):
        """
        Set the expected speed for each person,
        expected speed * max_speed_factor = the maximum speed of this person. 
        Use max_speed_factor to adjust the speed of the crowd.
        """
        for i in range(self.N):
            self.desiredvel[i] = (60+random.randint(0, self.speed_range))/100

    def fill_pos0_2batch(self,y,x1,x2):
        """
        Adjust pos_0 to set the initial position of the crowd. 
        In this function we have 2 batches of people. 
        You must first set self.group first"""        
        for i in range(self.N):
            self.pos_0[i][1] = y + random.uniform(-0.05, 0.05)
            if i < self.group[0]:
                self.pos_0[i][0] = x1
            else:
                self.pos_0[i][0] = x2
    
    def fill_pos0_circle(self):
        #circle
        for i in range(self.N):
            self.pos_0[i][0] = math.cos(2*math.pi/self.N*i)*0.4+0.5
            self.pos_0[i][1] = math.sin(2*math.pi/self.N*i)*0.4+0.5
        
    def fill_vel_target(self):
        """
        Tool function.
        Set the destination and initial speed of each group of people according to self.group_target and self.group_vel.
        You must set self.group,self.group_target, and self.group_vel first.
        """
        _vels = []
        _desiredpos = []
        for i in range (self.batch):
            for _ in range (self.group[i]):
                _vels.append(self.group_vel[i])
                _desiredpos.append(self.group_target[i])
        self.vel_0 = np.vstack(np.array(_vels))
        self.desiredpos_0 = np.vstack(np.array(_desiredpos))

    def set_scene(self,value):
        # set scene, value is the name of the scene
        # here are a set of example scene to choose from
        # you can also add your scene using all the tool functions, or write your function
        if value == "circle":
            #By default, everyone's target initial velocity is different
            self.group_target = np.zeros(shape=(self.N, 2), dtype=np.float32)
            self.group_vel = np.zeros(shape=(self.N, 2), dtype=np.float32) 
            self.fill_pos0_circle()
            for i in range(self.N):
                self.group_target[i] = 1-self.pos_0[i]
                self.group_vel[i] = [0.01,0]
            self.fill_group()
            self.fill_vel_target()
        elif value == "marching":
            self.group_target = np.zeros(shape=(self.N, 2), dtype=np.float32)
            self.group_vel = np.zeros(shape=(self.N, 2), dtype=np.float32) #默认每个人的target 初速度都不同
            num_cols = 10  #x cols of agent
            x=0.1 #starting position 
            y=0.5 
            num_per_col = self.N // num_cols 
            remainder = self.N % num_cols
            counter = 0 # people index counter
            for col in range(num_cols):
                num_in_this_col = num_per_col + 1 if col < remainder else num_per_col
                start_y = y - (num_in_this_col - 1) * 0.032 / 2
                for i in range(num_in_this_col):
                    self.pos_0[counter][0] = x
                    self.pos_0[counter][1] = start_y + i * 0.032 # Set Line Spacing
                    self.group_target[counter] = self.pos_0[counter]
                    self.group_target[counter][0] += 0.6 
                    
                    self.group_vel[counter] = [0.01,0]
                    counter += 1
                x += 0.032 # Set Column Spacing
            self.fill_group()
            self.fill_vel_target()

            # gender
            self.gender = [random.choice([0, 1]) for _ in range(self.N)]
        elif value == "GAU":# Gaussian normal distribution
            rng = default_rng(seed=20)
            self.group_target = np.zeros(shape=(self.N, 2), dtype=np.float32)
            self.group_vel = np.zeros(shape=(self.N, 2), dtype=np.float32) 
            mean_x = 0.85  # Mean value for x
            std_dev_x = 0.05  # Standard deviation for x
            x_coordinates = np.random.normal(mean_x, std_dev_x, self.N)
            
            
            mean_y = 0.5  # Mean value for y
            std_dev_y = 0.1  # Standard deviation for y
            y_coordinates = np.random.normal(mean_y, std_dev_y, self.N)

            # Ensure x and y coordinates are within the specified range
            x_coordinates = np.clip(x_coordinates, 0.7, 1)
            y_coordinates = np.clip(y_coordinates, 0.15, 0.85)

            for counter in range(self.N):
                self.pos_0[counter][0] = x_coordinates[counter]
                self.pos_0[counter][1] = y_coordinates[counter]
                self.group_target[counter] = self.pos_0[counter]
                self.group_target[counter][0] -= 0.7+rng.random(dtype=np.float32)/6
                self.group_target[counter][1] += (rng.random(dtype=np.float32)-0.5)/6
                self.group_vel[counter] = [-0.01,0]
            self.fill_group()
            self.fill_vel_target()
        elif value == "random":
            self.group_target = np.zeros(shape=(self.N, 2), dtype=np.float32)
            self.group_vel = np.zeros(shape=(self.N, 2), dtype=np.float32) 
            rng = default_rng(seed=42)
            self.pos_0 = rng.random(size=(self.N, 2), dtype=np.float32)
            # rng = default_rng(seed=18)
            # self.group_target = np.zeros(shape=(self.N, 2), dtype=np.float32)
            for i in range(self.N):
                self.pos_0[i][0] = rng.random()/6
                self.group_target[i] = [1,0.5]
                self.group_vel[i] = [0.01,0]
            self.fill_group()
            self.fill_vel_target()
        elif value == "maze": 
            # agents are spawned in the bottom right corner in this scene
            # self.group_target = np.array([[0.1, 0.5]]) 
            # self.group_vel = np.array([[-0.01, 0.]]) 
            # self.fill_group()
            # self.fill_vel_target()
            # self.fill_pos0_2batch(0.1,0.9,0.9)
            self.shuffle_group([0.1, 0.5])
            self.fill_vel_target()
        elif value == "escape":
            self.shuffle_group([0.5, 0])
            self.fill_vel_target()
        elif value == "long_hall":
            # (startx, starty, endx, endy)
            self.obstacles = [[0, 0, 1, 0],[0, 1, 1, 1],[1,0,1,1],[0,0,0,1],[0.5,0,0.5,0.45],[0.5,1,0.5,0.55]] 
            self.obstacles_x = np.array(self.obstacles)[:,0:2]
            self.obstacles_y = np.array(self.obstacles)[:,2:4]
            self.group_target = np.array([[0.1, 0.5],[0.9, 0.5]]) 
            self.group_vel = np.array([[-0.01, 0.],[0.01,0]]) 
            self.fill_group()
            self.fill_vel_target()
            self.fill_pos0_2batch(0.5,0.9,0.1)  
        elif value == "long_hall_gendergroup":
            self.obstacles = [[0, 0, 1, 0],[0, 1, 1, 1],[1,0,1,1],[0,0,0,1],[0.5,0,0.5,0.45],[0.5,1,0.5,0.55]]
            self.obstacles_x = np.array(self.obstacles)[:,0:2]
            self.obstacles_y = np.array(self.obstacles)[:,2:4]
            self.group_target = np.array([[0.1, 0.5],[0.9, 0.5]]) 
            self.group_vel = np.array([[-0.01, 0.],[0.01,0]]) 
            self.fill_group()
            self.fill_vel_target()
            self.fill_pos0_2batch(0.5,0.9,0.1)   
            self.fill_gender()      
        elif value == "block":
            self.group_target = np.array([[0.5, 0.5],[0.5, 0.5]]) 
            self.group_vel = np.array([[-0.01, 0.],[0.01,0]]) 
            self.batch = self.group_target.shape[0] # how many batch of agent
            self.fill_group()
            self.fill_vel_target()
            self.fill_pos0_2batch(0.9,0.9,0.1)
            self.obstacles = []
            if self.obstacle_setting == 2:
                # convex obstacle
                obstacle1 = []
                obstacle1.append(utils.vec2([0.4,0.75]))
                obstacle1.append(utils.vec2([0.2, 0.75]))
                obstacle1.append(utils.vec2([0.2, 0.55]))
                obstacle1.append(utils.vec2([0.4, 0.55]))
                self.obstacles.append(obstacle1)
                obstacle1 = []
                obstacle1.append(utils.vec2([0.6,0.75]))
                obstacle1.append(utils.vec2([0.6, 0.55]))
                obstacle1.append(utils.vec2([0.8, 0.55]))
                obstacle1.append(utils.vec2([0.8, 0.75]))
                self.obstacles.append(obstacle1)
                obstacle1 = []
                obstacle1.append(utils.vec2([0.2,0.45]))
                obstacle1.append(utils.vec2([0.4, 0.45]))
                obstacle1.append(utils.vec2([0.4, 0.25]))
                obstacle1.append(utils.vec2([0.2, 0.25]))
                self.obstacles.append(obstacle1)
                obstacle1 = []
                obstacle1.append(utils.vec2([0.6,0.45]))
                obstacle1.append(utils.vec2([0.8, 0.45]))
                obstacle1.append(utils.vec2([0.8, 0.25]))
                obstacle1.append(utils.vec2([0.6, 0.25]))
                self.obstacles.append(obstacle1)
                self.init_obstacle_line()
            
    def set(self,key,value):
        # This is the function called when reading the ini file, overwriting various parameters in config.py
        # set the value of a parameter, key is the name of the parameter, value is the value of the parameter
        if key == "n":
            self.N = int(value)
            self.vel_0 = np.zeros(dtype=np.float32, shape=(self.N, 2))
            self.pos_0 = np.zeros(dtype=np.float32,shape=(self.N, 2))
            self.desiredpos_0 = np.zeros(dtype=np.float32, shape=(self.N, 2))
            self.desiredvel = np.zeros(dtype=np.float32, shape=self.N)  # 期望速度大小m/s  
            self.speed_range = 15 #20240408:人群的速度差异值，在60之上加减这个范围的随机数      

            self.group_target = np.zeros(shape=(self.N, 2), dtype=np.float32)
            self.group_vel = np.zeros(shape=(self.N, 2), dtype=np.float32) #默认每个人的target 初速度都不同

            self.batch = self.N
        elif key == "speed_range":
            self.speed_range = float(value)
        elif key == "max_speed_factor":
            self.max_speed_factor = float(value)
        elif key == "draw_triangle":
            self.draw_triangle = int(value)
        elif key == "triangle_size":
            self.triangle_size = float(value)
        elif key == "people_radius":
            self.people_radius = int(value)
            self.triangle_size = self.people_radius / self.WINDOW_WIDTH 
        elif key == "obstacle_setting":
            self.obstacle_setting = int(value)
        elif key == "path": # read img and set it as the background map of the crowd
            self.path = value
            self.read_img()
        elif key == "im": # clear the background map
            self.im = None
        elif key == "window_height":
            self.WINDOW_HEIGHT = int(value)
        elif key == "window_width":
            self.WINDOW_WIDTH = int(value)
        elif key == "expand_size":
            self.expand_size = int(value)
        elif key == "astar":
            self.Astar = int(value)
        elif key == "dynamic_search":
            self.dynamic_search = int(value)
        elif key == "sim_step":
            self.sim_step = float(value)
        elif key == "analysis":
            self.analysis = int(value)
        elif key == "window_size":
            self.window_size = int(value)
            self.pixel_number = self.window_size ** 2
            self.resolution = self.window_size+20
        elif key == "resolution":
            self.resolution == int(value)
        elif key == "search_radius":
            self.search_radius = int(value)
        elif key == "stop_radius":
            self.stop_radius = float(value)
        elif key == "max_neighbour":
            self.max_neighbour = int(value)
        elif key == "desired_factor":
            self.desired_factor = float(value)
        elif key == "goal_threshold":
            self.goal_threshold = float(value)
        elif key == "relaxation_time":
            self.relaxation_time = float(value)
        elif key == "timeHorizon":
            self.timeHorizon = float(value)
        elif key == "timeHorizonObst":
            self.timeHorizonObst = float(value)
        elif key == "orca_constraints":
            self.ORCA_constraints = int(value)
        elif key == "invtimestep":
            self.invTimeStep = float(value)
        elif key == "orca_radius":
            self.ORCA_radius = float(value)
        elif key == "obstacle_orcaradius":
            self.obstacle_orcaradius = float(value)
        elif key == "max_obstacle":
            self.max_obstacle = int(value)

        elif key == "self.dense_sense":
            self.dense_sense = int(value)
        elif key =="dense_theta":
            self.dense_theta = float(value)
        elif key =="dense_k":
            self.dense_k = float(value)
        elif key =="dense_p":
            self.dense_p = float(value)
        
        elif key == "use_gender":
            self.use_gender = int(value)
        elif key == "gender_gap":
            self.gender_gap = float(value)
        elif key == "social_radius":
            self.social_radius = float(value)
        elif key == "lambda_importance":
            self.lambda_importance = float(value)
        elif key == "gamma":
            self.gamma = float(value)
        elif key == "n_fij":
            self.n_fij = float(value)
        elif key == "n_prime":
            self.n_prime = float(value)
        elif key == "fij_factor":
            self.fij_factor = float(value)
        elif key == "shoulder_radius":
            self.shoulder_radius = float(value)
        elif key == "sigma":
            self.sigma = float(value)
        elif key == "obstacle_threshold":
            self.obstacle_threshold = float(value)
        elif key == "obstacle_factor":
            self.obstacle_factor = float(value)

        elif key == "flocking":
            self.flocking = int(value)
        elif key == "flocking_radius":
            self.flocking_radius = float(value)
        elif key == "alignment_factor":
            self.alignment_factor = float(value)
        elif key == "separation_factor":
            self.separation_factor = float(value)
        elif key == "cohesion_factor":
            self.cohesion_factor = float(value)

        elif key == "steering_force":
            self.steering_force = int(value)
        elif key == "flee_factor":
            self.flee_factor = float(value)
        elif key == "seek_factor":
            self.seek_factor = float(value)
        elif key == "arrival_factor":
            self.arrival_factor = float(value)
        elif key == "behind_distance":
            self.behind_distance = float(value)
        elif key == "slowing_distance":
            self.slowing_distance = float(value)
        elif key == "leader_following_factor":
            self.leader_following_factor = float(value)
        elif key =="leader_index":
            self.leader_index = int(value)
        elif key == "oa_factor":
            self.oa_factor = float(value)
        elif key == "oa_forsee_factor":
            self.oa_forsee_factor = float(value)

        elif key == "state_machine":
            self.state_machine = int(value)
        elif key == "state_num":
            self.state_num = int(value)
        elif key == "state_target":
            self.state_target = json.loads(value)
        elif key == "state_velfactor":
            self.state_velfactor = json.loads(value)