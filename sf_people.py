import taichi as ti
from pre_astar import AStar
from sf_scene import Scene
import utils
import numpy as np
import time

@ti.data_oriented
class SF_People:
    def __init__(self, config):
        self.config = config
        self.scene = Scene(config)
        print("map init begin")
        self.astar = AStar(config.Astar,self.scene,self.config.astar_file)
        print("map init done")

        self.vel = ti.Vector.field(2, ti.f32)
        self.pos = ti.Vector.field(2, ti.f32)
        self.forces = ti.Vector.field(2, ti.f32)
        self.desiredpos = ti.Vector.field(2, ti.f32)
        self.max_vel = ti.field(ti.f32)
        ti.root.dense(ti.i, config.N).place(
            self.vel, self.pos,self.max_vel, self.desiredpos, self.forces) # AoS snode
        self.belong_batch = ti.field(ti.i8,self.config.N) 
        self.gender = ti.field(ti.i8,self.config.N) 
        self.group_count = ti.field(ti.i8,self.config.batch) # how many agents do each group have

        self.fill_parm()

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

        self.count_idx = ti.field(ti.i8,1)
        # self.leader_index[0] = self.config.leader_index 
        self.leader_index = ti.field(ti.i8,1)

    def fill_parm(self):
        """
        The function called during init.
        Implements the conversion from numpy to taichi field. 
        All data access after initialization uses taichi field.
        """
        self.vel.from_numpy(self.config.vel_0)
        self.pos.from_numpy(self.config.pos_0)
        self.desiredpos.from_numpy(self.config.desiredpos_0)
        self.max_vel.from_numpy(
            self.config.desiredvel * self.config.max_speed_factor)
        self.group_count.from_numpy(np.array(self.config.group))
        # fill that each agent belongs to which group
        group = []
        for batch in range (self.config.batch):
            group.append([batch] * self.config.group[batch])
        self.belong_batch.from_numpy(np.concatenate(group))
        # self.belong_batch.from_numpy(np.hstack(np.array(group)))
        if self.config.use_gender == 1:
            self.gender.from_numpy(np.array(self.config.gender))

    def render(self, gui):
        """taichi gui render"""
        if self.config.im is None:
            gui.clear(0xffffff)
            if self.config.obstacle_setting:
                gui.lines(begin=self.config.obstacles_x,end=self.config.obstacles_y, radius=2, color=0x000000)
        else:
            gui.set_image(self.config.im)
        
        people_centers = self.pos.to_numpy()
        sum = 0
        color_list = [0x8B658B,0x54FF9F,0x00BFFF,0xCDBE70,0xD2691E]    
        color_idx = 0
        
        # render people in batch
        # for i in range (self.config.batch): 
            # if self.config.use_gender == 0:
            #     _color = 0x000011 #large group

            #     if self.config.group[i] <= 5: 
            #         # change color each batch
            #         # _color = 0x0000ff + int((0x00ffff-0x0000ff)/self.config.batch) * i
            #         _color = color_list[color_idx]
            #         color_idx += 1

            #         # draw line between group member
            #         for i_idx in range(self.config.group[i]):
            #             for j_idx in range(i_idx):
            #                 gui.line(self.pos[sum+i_idx], self.pos[sum+j_idx], radius=1.6, color=_color)
                
            # elif self.config.use_gender == 1:
            #     _color =  0x08ff08 if i%2 else 0x0000ff
            # gui.circles(people_centers[sum:sum+self.config.group[i],:],color= _color,radius=self.config.people_radius)
            # sum += self.config.group[i]

        for i in range (self.config.N): 
            # # change color each batch
            # _color = 0x111111 + int((0xBBBBBB-0x111111)/self.config.N) * i

            # # runway.ini-------color base on max_vel max_speed_factor = 0.25 speed_range = 15
            # if self.gender[i] == 0: #blue
            #     _color = 0x0000AA + int((0x0000FF-0x0000AA)* ((self.max_vel[i]-0.15)/0.0375))
            #     _color = _color & 0x0000FF
            # else: #green
            #     _color = 0x00AA00 + int((0x00FF00-0x00AA00)* ((self.max_vel[i]-0.15)/0.0375))
            #     blue_part = _color & 0x0000FF
            #     _color = (blue_part >> 8) << 16 | (blue_part & 0xFF) << 8
            _color = 0x002300
            gui.circle(people_centers[i,:],color= _color,radius=self.config.people_radius)

        # specify leader(in black)
        if self.config.leader_following_factor != 0 and self.config.steering_force == 1:
            gui.circle(self.pos[self.leader_index[0]],color= 0x000000,radius=self.config.people_radius)

        # draw triangles
        self.make_triangle()
        if self.config.draw_triangle == 1:
            hex_color = self.color_list.to_numpy()
            triangle_X = self.triangle_X.to_numpy()
            triangle_Y = self.triangle_Y.to_numpy()
            triangle_Z = self.triangle_Z.to_numpy()
            gui.triangles(a=triangle_X, b=triangle_Y, c=triangle_Z,color=hex_color)
    
    def ggui(self, gui):
        """taichi ggui render"""
        if self.config.im is None:
            if self.config.obstacle_setting:
                gui.lines(begin=self.config.obstacles_x,end=self.config.obstacles_y, radius=2, color=0xff0000)
        else:
            gui.set_image(self.config.im)
        
        gui.circles(self.pos,color=(0,0,1),radius=0.005)
        # specify leader(in black)
        if self.config.leader_following_factor != 0 and self.config.steering_force == 1:
            gui.circle(self.pos[self.leader_index[0]],color= 0x000000,radius=self.config.people_radius)


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
    def get_agent_desired_force(self,i) -> ti.Vector:
        """
        Returns the ideal speed/force of agent i
        """
        desired_force = ti.Vector([0.0, 0.0])
        # Non-A*: 
        # defined as the distance to the goal
        direction, dist = utils.normalize(self.desiredpos[i]-self.pos[i])
        if self.config.Astar == 1: #A*
            _pos2 = ti.floor(self.pos[i] * self.config.window_size , int)
            _pos = int(_pos2[0] * self.config.window_size + _pos2[1])
            # for escape scene test
            # direction = self.astar.map[self.belong_batch[i],_pos]
            direction = self.astar.map[0,_pos]
        if dist > self.config.goal_threshold:
            desired_force = direction * self.max_vel[i] - self.vel[i]
        else:
            desired_force = -1.0 * self.vel[i]
        return desired_force / self.config.relaxation_time * self.config.desired_factor
    
    @ti.kernel
    def compute_desired_force(self):
        for i in range(self.config.N):
            self.forces[i] += self.get_agent_desired_force(i)
            
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
        Neighborhood Update-Static Memory Allocation
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
        Neighborhood Update-Dynamic Memory Allocation
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

        Among them, static allocation of memory has a significant advantage when the number of people is large.
        """
        self.scene.grid_count.fill(0)
        if self.config.dynamic_search == 0:
            self.update_grid_static()
            self.count_idx[0] = 0 
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
                self.forces[current_people_index] += self.get_agent_desired_force(i)

                flocking_count = 0
                alignment_force = ti.Vector([0.0, 0.0])
                separation_force = ti.Vector([0.0, 0.0])
                cohesion_force = ti.Vector([0.0, 0.0])
                neighbour_count = 0
                obstacle_count = 0
                for index_i in range(x_begin, x_end):
                    for index_j in range(y_begin, y_end):
                        index = int(index_i * self.config.window_size + index_j)

                        #social force f_ob
                        if self.scene.obstacle_exist[index_i,index_j] == 1 and obstacle_count <= self.config.max_neighbour:
                            self.compute_obstacle_force(current_people_index,index_i,index_j)
                            obstacle_count += 1

                        for people_index in range(self.scene.grid_count[index]): 
                            other_people_index = self.scene.grid_matrix[index,people_index]
                            if current_people_index < other_people_index and neighbour_count <= self.config.max_neighbour:                               
                                #social force f_ij
                                self.compute_fij_force(current_people_index,other_people_index)
                                neighbour_count += 1
                            
                            # flocking
                            if self.config.flocking == 1:
                                dist = (self.pos[current_people_index] - self.pos[other_people_index]).norm()
                                if current_people_index != other_people_index and dist < self.config.flocking_radius and flocking_count < self.config.max_neighbour:
                                    alignment_force += self.vel[other_people_index]
                                    separation_force += (self.pos[current_people_index] - self.pos[other_people_index]) / dist
                                    cohesion_force += self.pos[other_people_index]
                                    flocking_count += 1

                if self.config.flocking == 1 and flocking_count > 0:
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
            grid_index = ti.floor(self.pos[i]*self.config.window_size,int)
            x_begin = ti.max(grid_index[0] - self.config.search_radius, 0)
            x_end = ti.min(grid_index[0] + self.config.search_radius+1, self.config.window_size)
            y_begin = ti.max(grid_index[1] - self.config.search_radius, 0)
            y_end = ti.min(grid_index[1] + self.config.search_radius+1, self.config.window_size)
            flocking_count = 0
            alignment_force = ti.Vector([0.0, 0.0])
            separation_force = ti.Vector([0.0, 0.0])
            cohesion_force = ti.Vector([0.0, 0.0])
            neighbour_count = 0
            obstacle_count = 0

            # small group 20240513 v1 using alignment and cohension. only added here

            # group_count = self.group_count[int(self.belong_batch[i])]
            # if group_count <= 5:
            #     sum_dist = 0.0
            #     for member_idx in range(i-group_count,i+group_count): # search for all group member
            #         if member_idx >= self.config.N or member_idx <= 0 or i == member_idx:
            #             continue
            #         if self.belong_batch[member_idx] == self.belong_batch[i]:
            #             cohesion_force += self.pos[member_idx]
            #             alignment_force += self.vel[member_idx]
            #             dist = (self.pos[i] - self.pos[member_idx]).norm()
            #             sum_dist += dist
            #     # priority: group cohension
            #     flocking_factor = 1 if sum_dist/(group_count-1) < self.config.flocking_radius else 0.3
            #     print("group:",i,group_count,flocking_factor,cohesion_force)
            #     self.forces[i] += self.get_agent_desired_force(i) * flocking_factor
            #     self.compute_flocking_force(i,alignment_force,separation_force,cohesion_force,group_count)

            # # small group 20240515 v2 using leader following. only added here
            # group_count = self.group_count[int(self.belong_batch[i])]
            # if group_count <= 5: # in group
            #     # encounter the first agent in group                    
            #     if self.count_idx[0]==i: 
            #         self.count_idx[0] += group_count
            #         self.leader_index[0] = i
            #         agt_dist = 2.0
            #         dist_threshold = 0.1
            #         # the agent most close to desired pos will be the leader
            #         for idx in range(i,self.count_idx[0]):
            #             tmp_dist = (self.desiredpos[idx]-self.pos[idx]).norm()
            #             if tmp_dist+dist_threshold < agt_dist:
            #                 agt_dist = tmp_dist
            #                 self.leader_index[0] = idx

            #     # compute leader following force
            #     if i == self.leader_index[0]:# for leader
            #         self.forces[i] += self.get_agent_desired_force(i)
            #     else:# for followers:
            #         follower_acc = ti.Vector([0.0,0.0])
            #         follower_target = self.calculate_follow_position()
            #         target_offset = follower_target-self.pos[i]
            #         distance = target_offset.norm()
            #         norm_desired_speed = utils.set_mag(target_offset,self.max_vel[i])
            #         desired_speed = norm_desired_speed if distance >= self.config.slowing_distance else norm_desired_speed * distance / self.config.slowing_distance
            #         follower_acc = (desired_speed - self.vel[i])*15
            #         self.forces[i] += follower_acc * self.config.leader_following_factor
                     
            # else:
            #     self.forces[i] += self.get_agent_desired_force(i)
            self.forces[i] += self.get_agent_desired_force(i)


            for index_i in range(x_begin, x_end):
                for index_j in range(y_begin, y_end):
                    search_index = int(index_i * self.config.window_size + index_j)

                    # social force f_ob
                    if self.scene.obstacle_exist[index_i,index_j] == 1 and obstacle_count <= self.config.max_neighbour:
                        self.compute_obstacle_force(i,index_i,index_j)
                        obstacle_count += 1
                    
                    for p_idx in range(self.scene.list_head[search_index],self.scene.list_tail[search_index]):
                        j = self.scene.particle_id[p_idx]

                        if i < j and neighbour_count <= self.config.max_neighbour:
                            #social force f_ij
                            self.compute_fij_force(i,j)
                            neighbour_count += 1 
                        
                        #flocking
                        if self.config.flocking == 1:
                            dist = (self.pos[i] - self.pos[j]).norm()

                            if i != j and dist < self.config.flocking_radius and flocking_count < self.config.max_neighbour:
                                alignment_force += self.vel[j]
                                separation_force += (self.pos[i] - self.pos[j]) / dist
                                cohesion_force += self.pos[j]
                                flocking_count += 1
            
            if self.config.flocking == 1 and flocking_count > 0:
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
            utils.set_mag((alignment_force / flocking_count), self.max_vel[i]) - self.vel[i],
            self.max_vel[i])
        separation = utils.limit(
            utils.set_mag((separation_force / flocking_count), self.max_vel[i]) - self.vel[i],
            self.max_vel[i]) 
        cohesion = utils.limit(
            utils.set_mag(((cohesion_force / flocking_count) - self.pos[i]), self.max_vel[i]) -
            self.vel[i], self.max_vel[i]) 

        self.forces[i] += alignment * self.config.alignment_factor
        self.forces[i] += separation * self.config.separation_factor
        self.forces[i] += cohesion * self.config.cohesion_factor

    @ti.func
    def compute_fij_force(self,current_people_index,other_people_index):
        """
        Calculate the social force between this individual and all other people in the current scene
        Add force to self.forces and truncate the whole
        current_people_index: agent index
        other_people_index: neighbor index to be calculated
        """
        pos_diff = self.pos[current_people_index]-self.pos[other_people_index]
        diff_direction, diff_length = utils.normalize(pos_diff)
        if diff_length < self.config.social_radius:
            vel_diff = self.vel[other_people_index]-self.vel[current_people_index]

            # compute interaction direction t_ij
            interaction_vec = self.config.lambda_importance * vel_diff + diff_direction
            interaction_direction, interaction_length = utils.normalize(
                interaction_vec)

            # compute angle theta (between interaction and position difference vector)
            theta = utils.vector_angles(
                interaction_direction) - utils.vector_angles(diff_direction)
            # compute model parameter B = gamma * ||D||
            B = self.config.gamma * interaction_length

            force_velocity_amount = ti.exp(
                -1.0 * diff_length / B - (self.config.n_prime * B * theta)**2)
            sign_theta = 0.0
            if theta > 0:
                sign_theta = 1.0
            elif theta < 0:
                sign_theta = -1.0
            force_angle_amount = -sign_theta * \
                ti.exp(-1.0 * diff_length / B -(self.config.n_fij * B * theta)**2)

            force_velocity = force_velocity_amount * interaction_direction
            force_angle = ti.Vector([0.0, 0.0])
            force_angle[0] = -force_angle_amount * interaction_direction[1]
            force_angle[1] = force_angle_amount * interaction_direction[0]
            fij = force_velocity + force_angle

            if self.config.use_gender == 1 and self.gender[current_people_index] != self.gender[other_people_index]:
                fij *= self.config.gender_gap
            
            self.forces[current_people_index] += fij*self.config.fij_factor
            self.forces[other_people_index] -= fij*self.config.fij_factor

    @ti.func
    def compute_obstacle_force(self,current_people_index,index_i,index_j):
        """
        Calculate the repulsion between a pedestrian (numbered as current_people_index) and the obstacle under the current grid
        The obstacle position: the center of the grid
        current_people_index: agent index
        index_i,index_j: current obstacle index (make sure this coordinate is an obstacle)
        """        
        diff = self.pos[current_people_index] - self.scene.grid_pos[index_i,index_j]
        directions, dist = utils.normalize(diff)
        dist += -self.config.shoulder_radius
        if dist < self.config.obstacle_threshold:               
            directions = directions * ti.exp(-dist / self.config.sigma)
            self.forces[current_people_index] += directions*self.config.obstacle_factor

    @ti.kernel
    def compute_steering_force(self):
        flee_target = self.pos[self.config.leader_index] # flee force for leader_following
        follower_target = self.calculate_follow_position()
        for i in range(self.config.N):
            # steering forces seek: pass through the target, and then turn back to approach again
            seek_acc = utils.limit(
                utils.set_mag((self.desiredpos[i]-self.pos[i]), self.max_vel[i]) - self.vel[i],
                self.max_vel[i]) 

            #flee forces:
            flee_acc = utils.limit(
                utils.set_mag((self.pos[i]-self.desiredpos[i]), self.max_vel[i]) - self.vel[i],
                self.max_vel[i]) 

            #arrival forces：
            target_offset = self.desiredpos[i]-self.pos[i]
            distance = target_offset.norm()
            norm_desired_speed = utils.set_mag(target_offset,self.max_vel[i])
            desired_speed = norm_desired_speed if distance >= self.config.slowing_distance else norm_desired_speed * distance / self.config.slowing_distance            
            arrival_acc = (desired_speed - self.vel[i])

            #leader following forces:
            follower_acc = ti.Vector([0,0])
            if (self.pos[i]-flee_target).norm() < self.config.behind_distance :
                #flee forces:
                follower_acc = utils.limit(
                    utils.set_mag((self.pos[i]-flee_target), self.max_vel[i]) - self.vel[i],
                    self.max_vel[i]) * self.config.flee_factor
            else: 
                target_offset = follower_target-self.pos[i]
                distance = target_offset.norm()
                norm_desired_speed = utils.set_mag(target_offset,self.max_vel[i])
                desired_speed = norm_desired_speed if distance >= self.config.slowing_distance else norm_desired_speed * distance / self.config.slowing_distance
                follower_acc = (desired_speed - self.vel[i])*15

            if self.config.leader_following_factor != 0:
                if i == self.config.leader_index:
                    self.forces[i] += arrival_acc * self.config.arrival_factor
                else:
                    #对于跟随者
                    self.forces[i] += follower_acc * self.config.leader_following_factor
            else:
                self.forces[i] += arrival_acc * self.config.arrival_factor
                self.forces[i] += flee_acc * self.config.flee_factor
                self.forces[i] += seek_acc * self.config.seek_factor  

    def compute(self):
        """
        simulator will call this function
        """
        self.forces.fill(0)
        self.update_grid()      

        # additional add steering force
        if self.config.steering_force == 1:
            self.compute_steering_force()

        if self.config.analysis == 1:
            self.analysis()
            self.collision_detect()

    @ti.kernel
    def update(self):
        for i in range(self.config.N):
            # social force: compute new_vel from force
            new_vel = self.vel[i] + self.config.sim_step * self.forces[i]                
            new_vel = utils.capped(new_vel, self.max_vel[i])                            

            # if close enough to target，stop
            destination_vector = self.desiredpos[i] - self.pos[i]
            _, dist = utils.normalize(destination_vector)
            if dist < self.config.stop_radius:
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
        tv = -self.vel[int(self.leader_index[0])] #20240515
        tv = tv / tv.norm() * self.config.behind_distance
        behind = self.pos[int(self.leader_index[0])] + tv
        # tv = -self.vel[self.config.leader_index] 
        # tv = tv / tv.norm() * self.config.behind_distance
        # behind = self.pos[self.config.leader_index] + tv
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
                destination_vector = self.desiredpos[i] - self.pos[i]
                dist = destination_vector.norm()
                if dist < self.config.stop_radius+self.config.people_radius:
                    t = time.time()
                    self.has_arrived[i] = 1
                    self.TotalTime += t - self.StartTime
                    self.ArrivalRate += 1

                                    
