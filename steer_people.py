import taichi as ti
from pre_astar import AStar
from steer_scene import Scene
import utils
import numpy as np
import  time
@ti.data_oriented
class Steer_People:
    def __init__(self, config):
        self.config = config
        self.scene = Scene(config)
        print("map init begin")
        self.astar = AStar(config.Astar,self.scene)
        print("map init done")

        self.vel = ti.Vector.field(2, ti.f32)
        self.pos = ti.Vector.field(2, ti.f32)
        self.forces = ti.Vector.field(2, ti.f32)
        ti.root.dense(ti.i, config.N).place(self.vel, self.pos, self.forces) # AoS snode
        self.belong_batch = ti.field(ti.i8,self.config.N) 
        self.desiredpos = ti.Vector.field(2, ti.f32)
        self.max_vel = ti.field(ti.f32)
        ti.root.dense(ti.ij,(config.state_num,config.N)).place(self.max_vel, self.desiredpos)
        self.state = ti.field(ti.i8,self.config.N) # The current state of the state machine. initially is 0

        self.fill_parm() 
        self.init_state()

        self.color_list = ti.field(ti.i32,self.config.N)
        self.triangle_X = ti.Vector.field(2, ti.f32) # three vertices of a triangle
        self.triangle_Y = ti.Vector.field(2, ti.f32)
        self.triangle_Z = ti.Vector.field(2, ti.f32)
        ti.root.dense(ti.i, self.config.N).place(self.triangle_X,self.triangle_Y,self.triangle_Z)

        self.ArrivalRate = 0
        self.Collision = ti.field(ti.i8,1)
        self.StartTime = time.time()
        self.has_arrived = ti.field(ti.i8,self.config.N)
        self.TotalTime = 0.0

    def init_state(self):
        """Fill desiredpos and maxvel, which change as the state machine changes""" 
        if self.config.state_machine == 0:
            self.desiredpos.from_numpy(np.array([self.config.desiredpos_0]))
            self.max_vel.from_numpy(
                np.array([self.config.desiredvel * self.config.max_speed_factor]))
        else:
            _desiredpos = np.array([self.get_desiredpos(self.config.state_target[0])])
            _maxvel = np.array([self.config.desiredvel * self.config.state_velfactor[0]])
            for i in range(1,self.config.state_num):
                t = np.array([self.get_desiredpos(self.config.state_target[i])])
                _desiredpos = np.concatenate((_desiredpos,t),axis = 0)
                t = np.array([self.config.desiredvel * self.config.state_velfactor[i]])
                _maxvel = np.concatenate((_maxvel,t),axis = 0)
            self.desiredpos.from_numpy(_desiredpos)
            self.max_vel.from_numpy(_maxvel)

    def fill_parm(self):
        """
        The function called during init.
        Implements the conversion from numpy to taichi field. 
        All data access after initialization uses taichi field.
        """
        self.vel.from_numpy(self.config.vel_0)
        self.pos.from_numpy(self.config.pos_0)

        group = []
        for batch in range (self.config.batch):
            group.append([batch] * self.config.group[batch])
        self.belong_batch.from_numpy(np.hstack(np.array(group)))

    def get_desiredpos(self,group_target):
        """
        Get the target point from group target.
        Group target is a list.
        Each element is a target point.
        It must correspond to the batch and group in config.py
        """
        _desiredpos = []
        for i in range (self.config.batch):
            for _ in range (self.config.group[i]):
                _desiredpos.append(group_target[i])
        x = np.vstack(np.array(_desiredpos))
        return x

    def render(self, gui):
        """now is fixed to taichi gui rendering instead of ggui rendering"""
        if self.config.im is None:
            gui.clear(0xffffff)
            if self.config.obstacle_setting:
                gui.lines(begin=self.config.obstacles_x,end=self.config.obstacles_y, radius=2, color=0xff0000)
        else:
            gui.set_image(self.config.im)
        
        people_centers = self.pos.to_numpy()
        sum = 0
        
        for i in range (self.config.batch): #render people in batch
            # change color each batch
            _color = 0x00ff00 + 0x0000ff * i
            gui.circles(people_centers[sum:sum+self.config.group[i],:],color= _color,radius=self.config.people_radius)
            sum += self.config.group[i]

        # specify leader(in black)
        if self.config.leader_following_factor != 0 and self.config.steering_force == 1:
            gui.circle(self.pos[self.config.leader_index],color= 0x000000,radius=self.config.people_radius)

        # draw triangles
        self.make_triangle()
        if self.config.draw_triangle == 1:
            hex_color = self.color_list.to_numpy()
            triangle_X = self.triangle_X.to_numpy()
            triangle_Y = self.triangle_Y.to_numpy()
            triangle_Z = self.triangle_Z.to_numpy()
            gui.triangles(a=triangle_X, b=triangle_Y, c=triangle_Z,color=hex_color)


    @ti.kernel
    def make_triangle(self):
        """
        Calculate the vertices and colors of the triangle. 
        The color of the triangle represents the magnitude of the force. 
        The direction is the direction of the resultant force at this moment.
        """
        for i in range(self.config.N):
            direction,value = utils.normalize(self.forces[i])
            hex_color = utils.convert_data(value)  
            self.color_list[i] = hex_color
            X = self.pos[i] + direction * self.config.triangle_size * 1.2
            self.triangle_X[i] = X
            per_direction = ti.Vector([direction[1],-direction[0]])
            X_back = self.pos[i] - direction * self.config.triangle_size * 0.8
            self.triangle_Y[i] = X_back + per_direction * self.config.triangle_size *0.6
            self.triangle_Z[i] = X_back - per_direction * self.config.triangle_size *0.6


    @ti.func
    def compute_steering_force(self,i):
        if self.config.steering_force == 1:
            flee_target = self.pos[self.config.leader_index] # flee force for leader_following
            follower_target = self.calculate_follow_position()
            
            # steering forces seek: pass through the target, and then turn back to approach again
            seek_acc = utils.limit(
                utils.set_mag((self.desiredpos[self.state[i],i]-self.pos[i]), self.max_vel[self.state[i],i]) - self.vel[i],
                self.max_vel[self.state[i],i]) 

            #flee forces:
            flee_acc = utils.limit(
                utils.set_mag((self.pos[i]-self.desiredpos[self.state[i],i]), self.max_vel[self.state[i],i]) - self.vel[i],
                self.max_vel[self.state[i],i]) 

            #arrival forces：
            target_offset = self.desiredpos[self.state[i],i]-self.pos[i]
            distance = target_offset.norm()
            norm_desired_speed = utils.set_mag(target_offset,self.max_vel[self.state[i],i])
            desired_speed = norm_desired_speed if distance >= self.config.slowing_distance else norm_desired_speed * distance / self.config.slowing_distance            
            arrival_acc = (desired_speed - self.vel[i])

            #leader following forces:
            follower_acc = ti.Vector([0,0])
            if (self.pos[i]-flee_target).norm() < self.config.behind_distance :
                #flee forces:
                follower_acc = utils.limit(
                    utils.set_mag((self.pos[i]-flee_target), self.max_vel[self.state[i],i]) - self.vel[i],
                    self.max_vel[self.state[i],i]) * self.config.flee_factor
            else: 
                target_offset = follower_target-self.pos[i]
                distance = target_offset.norm()
                norm_desired_speed = utils.set_mag(target_offset,self.max_vel[self.state[i],i])
                desired_speed = norm_desired_speed if distance >= self.config.slowing_distance else norm_desired_speed * distance / self.config.slowing_distance
                follower_acc = (desired_speed - self.vel[i])*15

            if self.config.leader_following_factor != 0:
                if i == self.config.leader_index:
                    self.forces[i] += arrival_acc * self.config.arrival_factor
                else:
                    # for followers
                    self.forces[i] += follower_acc * self.config.leader_following_factor
            else:
                self.forces[i] += arrival_acc * self.config.arrival_factor
                self.forces[i] += flee_acc * self.config.flee_factor
                self.forces[i] += seek_acc * self.config.seek_factor  


    @ti.func
    def cut_val(self,i):
        if self.pos[i][0] >= 1:
            self.pos[i][0] = 0.999
        elif self.pos[i][0] <0:
            self.pos[i][0] = 0
        if self.pos[i][1] >= 1:
            self.pos[i][1] = 0.999
        elif self.pos[i][1] <0:
            self.pos[i][1] = 0

    @ti.kernel
    def update_grid_static(self):
        """
        Dynamic Update-Static Memory Allocation
        """
        # count = 0
        for i in range(self.config.N):
            self.cut_val(i)
            grid_idx = ti.floor(self.pos[i] * self.config.window_size, int)
            self.scene.grid_count[grid_idx] += 1
        #     count +=1
        # assert(count == self.config.N)

        for i in range(self.config.window_size):
            sum = 0
            for j in range(self.config.window_size):
                sum += self.scene.grid_count[i, j]
            self.scene.column_sum[i] = sum

        self.scene.prefix_sum[0, 0] = 0

        ti.loop_config(serialize=True)
        for i in range(1, self.config.window_size):
            self.scene.prefix_sum[i, 0] = self.scene.prefix_sum[i - 1, 0] + self.scene.column_sum[i - 1]

        for i ,j in self.scene.prefix_sum:
            if j == 0:
                self.scene.prefix_sum[i, j] += self.scene.grid_count[i, j]
            else:
                self.scene.prefix_sum[i, j] = self.scene.prefix_sum[i, j - 1] + self.scene.grid_count[i, j]

            linear_idx = i * self.config.window_size + j

            self.scene.list_head[linear_idx] = self.scene.prefix_sum[i, j] - self.scene.grid_count[i, j]
            self.scene.list_cur[linear_idx] = self.scene.list_head[linear_idx]
            self.scene.list_tail[linear_idx] = self.scene.prefix_sum[i, j]

        for i in range(self.config.N):
            grid_idx = ti.floor(self.pos[i] * self.config.window_size, int)
            linear_idx = grid_idx[0] * self.config.window_size + grid_idx[1]
            grain_location = ti.atomic_add(self.scene.list_cur[linear_idx], 1)
            self.scene.particle_id[grain_location] = i

    @ti.kernel
    def update_grid_dynamic(self):
        """
        Dynamic Update-Dynamic Memory Allocation
        """
        for i in range(self.config.N):
            self.cut_val(i)
            grid_idx = ti.floor(self.pos[i] * self.config.window_size, int)
            index = grid_idx[0]*self.config.window_size + grid_idx[1]
            self.scene.grid_count[index] += 1
            ti.append(self.scene.grid_matrix.parent(),index,i)

    def update_grid(self):
        """
        Neighborhood Update+Search 
        Static Memory Allocate (https://zhuanlan.zhihu.com/p/563182093)
        or
        Dynamic Memory Allocate using the dynamic node.
        """
        self.scene.grid_count.fill(0)
        if self.config.dynamic_search == 0:
            self.update_grid_static()
            self.search_grid_static()
        else:
            self.scene.block.deactivate_all()
            self.update_grid_dynamic()
            self.search_grid_dynamic()

    @ti.kernel
    def search_grid_dynamic(self):
        """
        Neighborhood search-Dynamic memory allocation
        Map gridding. For each grid, update fij and obstacle force
        Time complexity O(N) N is the number of grids in the map
        The repulsion between people, the repulsion of obstacles, and the destination force are all calculated in the neighborhood search
        """
        for grid in range(self.config.pixel_number):
            grid_x = ti.floor(grid / self.config.window_size)
            grid_y = grid % self.config.window_size
            x_begin = ti.max(grid_x - self.config.search_radius, 0)
            x_end = ti.min(grid_x + self.config.search_radius+1, self.config.window_size)
            y_begin = ti.max(grid_y - self.config.search_radius, 0)
            y_end = ti.min(grid_y + self.config.search_radius+1, self.config.window_size)

            for i in range(self.scene.grid_count[grid]):
                current_people_index = self.scene.grid_matrix[grid,i]
                self.compute_steering_force(current_people_index)

                flocking_count = 0
                alignment_force = ti.Vector([0.0, 0.0])
                separation_force = ti.Vector([0.0, 0.0])
                cohesion_force = ti.Vector([0.0, 0.0])
                obstacle_count = 0
                v = self.vel[current_people_index] * (self.vel[current_people_index].norm() / self.max_vel[self.state[current_people_index],current_people_index])
                ahead = self.pos[current_people_index]
                ahead += v
                halfahead = self.pos[current_people_index]
                halfahead += v/0.1
                
                for index_i in range(x_begin, x_end):
                    for index_j in range(y_begin, y_end):
                        index = int(index_i * self.config.window_size + index_j)

                        #f_ob  
                        if self.scene.obstacle_exist[index_i,index_j] == 1 and obstacle_count <= self.config.max_neighbour:
                            self.compute_obstacle_avoidance(current_people_index,index_i,index_j,ahead,halfahead)
                            obstacle_count += 1

                        for people_index in range(self.scene.grid_count[index]): 
                            other_people_index = self.scene.grid_matrix[index,people_index]
                            # flocking
                            dist = (self.pos[current_people_index] - self.pos[other_people_index]).norm()
                            if current_people_index != other_people_index and dist < self.config.flocking_radius and flocking_count < self.config.max_neighbour:
                                alignment_force += self.vel[other_people_index]
                                separation_force += (self.pos[current_people_index] - self.pos[other_people_index]) / dist
                                cohesion_force += self.pos[other_people_index]
                                flocking_count += 1

                if flocking_count > 0:
                    self.compute_flocking_force(current_people_index,alignment_force,separation_force,cohesion_force,flocking_count)
                
    @ti.kernel
    def search_grid_static(self):
        """
        Neighborhood search-Static memory allocation
        Map gridding. For each grid, update fij and obstacle force
        Time complexity O(N) N is the number of grids in the map
        The repulsion between people, the repulsion of obstacles, and the destination force are all calculated in the neighborhood search
        """
        for i in range (self.config.N):
            self.compute_steering_force(i)
            grid_index = ti.floor(self.pos[i]*self.config.window_size,int)
            x_begin = ti.max(grid_index[0] - self.config.search_radius, 0)
            x_end = ti.min(grid_index[0] + self.config.search_radius+1, self.config.window_size)
            y_begin = ti.max(grid_index[1] - self.config.search_radius, 0)
            y_end = ti.min(grid_index[1] + self.config.search_radius+1, self.config.window_size)
            flocking_count = 0
            alignment_force = ti.Vector([0.0, 0.0])
            separation_force = ti.Vector([0.0, 0.0])
            cohesion_force = ti.Vector([0.0, 0.0])
            obstacle_count = 0
            v = self.vel[i] * (self.vel[i].norm() / self.max_vel[self.state[i],i])
            ahead = self.pos[i]
            ahead += v
            halfahead = self.pos[i]
            halfahead += v/0.1

            for index_i in range(x_begin, x_end):
                for index_j in range(y_begin, y_end):
                    search_index = int(index_i * self.config.window_size + index_j)

                    # f_ob
                    if self.scene.obstacle_exist[index_i,index_j] == 1 and obstacle_count <= self.config.max_neighbour:
                        self.compute_obstacle_avoidance(i,index_i,index_j,ahead,halfahead)
                        obstacle_count += 1
                    
                    for p_idx in range(self.scene.list_head[search_index],self.scene.list_tail[search_index]):
                        j = self.scene.particle_id[p_idx]
                        
                        #flocking
                        dist = (self.pos[i] - self.pos[j]).norm()

                        if i != j and dist < self.config.flocking_radius and flocking_count < self.config.max_neighbour:
                            alignment_force += self.vel[j]
                            separation_force += (self.pos[i] - self.pos[j]) / dist
                            cohesion_force += self.pos[j]
                            flocking_count += 1
        
            if flocking_count > 0:
                self.compute_flocking_force(i,alignment_force,separation_force,cohesion_force,flocking_count)
            
                        
    @ti.func
    def compute_flocking_force(self,i,alignment_force,separation_force,cohesion_force,flocking_count):
        """
        Calculate flocking force
        i represents the agent index
        alignment_force, separation_force, cohesion_force are the forces updated in the neighborhood search
        flocking_count is the number of neighbors
        """
        alignment = utils.limit(
            utils.set_mag((alignment_force / flocking_count), self.max_vel[self.state[i],i]) - self.vel[i],
            self.max_vel[self.state[i],i])
        separation = utils.limit(
            utils.set_mag((separation_force / flocking_count), self.max_vel[self.state[i],i]) - self.vel[i],
            self.max_vel[self.state[i],i]) 
        cohesion = utils.limit(
            utils.set_mag(((cohesion_force / flocking_count) - self.pos[i]), self.max_vel[self.state[i],i]) -
            self.vel[i], self.max_vel[self.state[i],i]) 

        self.forces[i] += alignment * self.config.alignment_factor
        self.forces[i] += separation * self.config.separation_factor
        self.forces[i] += cohesion * self.config.cohesion_factor

    @ti.func
    def compute_obstacle_avoidance(self,current_people_index,index_i,index_j,ahead,halfahead):
        """
        Calculate the repulsion between a pedestrian (numbered as current_people_index) and the obstacle under the current grid
        The obstacle position take the center of the obstacle grid
        current_people_index: agent index
        index_i, index_j: current obstacle index (make sure this coordinate is an obstacle)
        Use flee to simulate obstacle repulsion
        """ 
        avoid_acc = ti.Vector([0.0, 0.0])

        # reading obstacle info 
        distance = ahead - self.scene.grid_pos[index_i,index_j]
        subdistance = halfahead - self.scene.grid_pos[index_i,index_j]
        direct_dist = (self.pos[current_people_index] - self.scene.grid_pos[index_i,index_j]).norm()
        
        if distance.norm()<=self.config.oa_radius or subdistance.norm()<=self.config.oa_radius or direct_dist <= self.config.oa_radius:
            avoid_acc += utils.limit(utils.set_mag(distance, self.config.oa_factor),self.max_vel[self.state[current_people_index],current_people_index])
                
        self.forces[current_people_index] += avoid_acc*self.config.oa_factor

    def compute(self):
        """
        simulator will call this function
        """
        self.forces.fill(0)
        self.update_grid()       
        if self.config.analysis == 1:
            self.analysis()
            self.collision_detect() 

    @ti.kernel
    def update(self):
        for i in range(self.config.N):
            # compute new_vel from force
            new_vel = self.vel[i] + self.config.sim_step * self.forces[i]                
            new_vel = utils.capped(new_vel, self.max_vel[self.state[i],i])                            

            # If agent are close enough to the target, change the state
            destination_vector = self.desiredpos[self.state[i],i] - self.pos[i]
            _, dist = utils.normalize(destination_vector)
            if dist < self.config.stop_radius:
                if self.state[i] < self.config.state_num-1:# haven't reach final state
                    self.state[i] += 1
                else: # final state achieved, stop
                    new_vel = ti.Vector([0.0,0.0])

            self.vel[i] = new_vel
            self.pos[i] += new_vel * self.config.sim_step

    @ti.kernel
    def print_id(self,mouse_x:ti.f32,mouse_y:ti.f32):
        mouse = ti.Vector([mouse_x,mouse_y])
        for i in range(self.config.N):
            diff = self.pos[i] - mouse
            _,dist = utils.normalize(diff)
            if (dist < 0.05):
                print("selecting people:",self.pos[i],i)

    @ti.func
    def calculate_follow_position(self):
        # assumes that only leader will trriger this function, so we take vel[index] and pos[index]
        tv = -self.vel[self.config.leader_index]
        tv = tv / tv.norm() * self.config.behind_distance
        behind = self.pos[self.config.leader_index] + tv
        return behind

    @ti.kernel
    def collision_detect(self):
        # check if people collide with each other
        for i in range(self.config.N):
            for j in range(i+1,self.config.N):
                diff = self.pos[i] - self.pos[j]
                _,dist = utils.normalize(diff)
                if dist < self.config.people_radius:
                    ti.atomic_add(self.Collision, 1)
    
    def analysis(self):
        # check if people arrive at destination
        for i in range(self.config.N):
            if self.has_arrived[i] == 0:
                destination_vector = self.desiredpos[self.state[i],i] - self.pos[i]
                dist = destination_vector.norm()
                if dist < self.config.stop_radius+self.config.people_radius:
                    t = time.time()
                    self.has_arrived[i] = 1
                    self.TotalTime += t - self.StartTime
                    self.ArrivalRate += 1
