import taichi as ti
from orca_scene import Scene
import utils
import numpy as np
import time
@ti.data_oriented
class ORCA_People:
    def __init__(self, config):
        self.config = config
        self.scene = Scene(config)

        self.vel = ti.Vector.field(2, ti.f32)
        self.pos = ti.Vector.field(2, ti.f32)
        self.new_velocity = ti.Vector.field(2, ti.f32)
        self.forces = ti.Vector.field(2, ti.f32)
        self.desiredpos = ti.Vector.field(2, ti.f32)
        self.max_vel = ti.field(ti.f32)
        ti.root.dense(ti.i, config.N).place(
            self.vel, self.pos,self.new_velocity,self.max_vel, self.desiredpos, self.forces) # AoS snode
        # Which group does each person belong to?
        self.belong_batch = ti.field(ti.i8,self.config.N) 

        self.fill_parm()
        self.ArrivalRate = 0
        self.Collision = ti.field(ti.i8,1)
        self.StartTime = time.time()
        self.has_arrived = ti.field(ti.i8,self.config.N)
        self.TotalTime = 0.0

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
        # fill that each agent belongs to which group
        group = []
        for batch in range (self.config.batch):
            group.append([batch] * self.config.group[batch])
        self.belong_batch.from_numpy(np.hstack(np.array(group)))
        self.expand()

    @ti.kernel
    def expand(self):
        for i in range (self.config.N):
            self.pos[i] = utils.expand(self.pos[i],self.config.WINDOW_WIDTH/self.config.expand_size,self.config.WINDOW_HEIGHT/self.config.expand_size)
            self.desiredpos[i] = utils.expand(self.desiredpos[i],self.config.WINDOW_WIDTH/self.config.expand_size,self.config.WINDOW_HEIGHT/self.config.expand_size)
            
    def render(self, gui):
        """now is fixed to taichi gui rendering instead of ggui rendering"""
        # if self.config.im is None:
        gui.clear(0xffffff)
        if self.config.obstacle_setting:
            gui.lines(begin=self.config.obstacles_x,end=self.config.obstacles_y, radius=2, color=0xff0000)
        # else:
        #     gui.set_image(self.config.im)
        
        people_centers = self.pos.to_numpy()
        for i in range(self.config.N):
            people_centers[i][0] = people_centers[i][0]/(self.config.WINDOW_WIDTH/self.config.expand_size)
            people_centers[i][1] = people_centers[i][1]/(self.config.WINDOW_HEIGHT/self.config.expand_size)
        sum = 0
        for i in range (self.config.batch):
            # change color each batch
            _color = 0x00ff00 + 0x0000ff * i
            gui.circles(people_centers[sum:sum+self.config.group[i],:],color= _color,radius=self.config.people_radius)
            sum += self.config.group[i]
    @ti.func
    def get_agent_desired_force(self,i) -> ti.Vector:
        """
        Returns the ideal speed/force of agent i
        """
        return self.desiredpos[i]-self.pos[i]
    
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

    @ti.func
    def compute_ORCA_obstacle(self,current_people_index,obstaclNo,ORCA_lines,ob_count):
        """
        Returns whether it is necessary to add this ORCA LINE
        """
        obstacle1No = obstaclNo
        obstacle2No = self.scene.obstacles[obstaclNo].next

        relativePosition1 = self.scene.obstacles[obstacle1No].point - self.pos[current_people_index]
        relativePosition2 = self.scene.obstacles[obstacle2No].point - self.pos[current_people_index]

        line = utils.Line([0.0, 0.0, 0.0, 0.0])
        required = False

        invTimeHorizonObst = 1.0 / self.config.timeHorizonObst
        cur_vel = self.vel[current_people_index]
        # Check if velocity obstacle of obstacle is already taken care of by previously constructed obstacle ORCA lines.
        alreadyCovered = False
    
        for i in range(ob_count):
            det1 = utils.det(invTimeHorizonObst * relativePosition1 - ORCA_lines[i,2:],ORCA_lines[i,0:2])
            det2 = utils.det(invTimeHorizonObst * relativePosition2 - ORCA_lines[i,2:], ORCA_lines[i,0:2])
            if (det1 - invTimeHorizonObst * self.config.obstacle_orcaradius >= -utils.EPSILON) and (det2 - invTimeHorizonObst * self.config.obstacle_orcaradius >= -utils.EPSILON):
                alreadyCovered = True
                break
            
        if not alreadyCovered:
            # Not yet covered. Check for collisions.
            distSq1 = utils.abs_sq(relativePosition1,relativePosition1)
            distSq2 = utils.abs_sq(relativePosition2,relativePosition2)

            radiusSq = self.config.obstacle_orcaradius * self.config.obstacle_orcaradius

            obstacleVector = self.scene.obstacles[obstacle2No].point - self.scene.obstacles[obstacle1No].point
            s = utils.abs_sq(-relativePosition1, obstacleVector) / utils.abs_sq(obstacleVector,obstacleVector)
            temp = -relativePosition1 - s * obstacleVector
            distSqLine = utils.abs_sq(temp, temp)

            to_break = False

            if s < 0.0 and distSq1 <= radiusSq:
                # Collision with left vertex. Ignore if non-convex.
                to_break = True
                if self.scene.obstacles[obstacle1No].convex:
                    line[0:2],_ = utils.normalize(utils.vec2([-relativePosition1[1], relativePosition1[0]]))
                    print(current_people_index,obstaclNo,"Collision with left vertex",line)
                    required = True
            elif s > 1.0 and distSq2 <= radiusSq and to_break == False:
                # Collision with right vertex. Ignore if non-convex or if it will be taken care of by neighboring obstacle.
                to_break = True
                if self.scene.obstacles[obstacle2No].convex and utils.det(relativePosition2, self.scene.obstacles[obstacle2No].direction) >= 0.0:
                    line[0:2],_ = utils.normalize(utils.vec2([-relativePosition2[1], relativePosition2[0]]))
                    print(current_people_index,obstaclNo,"Collision with right vertex",line)
                    required = True
            elif s >= 0.0 and s < 1.0 and distSqLine <= radiusSq and to_break == False:
                # Collision with obstacle segment.
                line[0:2] = -self.scene.obstacles[obstacle1No].direction
                print(current_people_index,obstaclNo,"Collision with segment",line)
                to_break = True
                required = True
            elif to_break == False:
                # No collision. Compute legs. When obliquely viewed, both legs can come from a single vertex. Legs extend cut-off line when non-convex vertex.
                leftLegDirection = utils.vec2([0.0,0.0])
                rightLegDirection = utils.vec2([0.0,0.0])
                flag = False

                if s < 0.0 and distSqLine <= radiusSq:
                    # Obstacle viewed obliquely so that left vertex defines velocity obstacle.
                    if not self.scene.obstacles[obstacle1No].convex:
                        # Ignore obstacle.
                        flag = True

                    obstacle2No = obstacle1No

                    leg1 = ti.sqrt(distSq1 - radiusSq)
                    leftLegDirection = utils.vec2([relativePosition1[0] * leg1 - relativePosition1[1] * self.config.obstacle_orcaradius, relativePosition1[0] * self.config.obstacle_orcaradius + relativePosition1[1] * leg1]) / distSq1
                    rightLegDirection = utils.vec2([relativePosition1[0] * leg1 + relativePosition1[1] * self.config.obstacle_orcaradius, -relativePosition1[0] * self.config.obstacle_orcaradius + relativePosition1[1] * leg1]) / distSq1
                elif s > 1.0 and distSqLine <= radiusSq:
                    # Obstacle viewed obliquely so that right vertex defines velocity obstacle.
                    if not self.scene.obstacles[obstacle2No].convex:
                        # Ignore obstacle.
                        flag = True

                    obstacle1No = obstacle2No

                    leg2 = ti.sqrt(distSq2 - radiusSq)
                    leftLegDirection = utils.vec2([relativePosition2[0] * leg2 - relativePosition2[1] * self.config.obstacle_orcaradius, relativePosition2[0] * self.config.obstacle_orcaradius + relativePosition2[1] * leg2]) / distSq2
                    rightLegDirection = utils.vec2([relativePosition2[0] * leg2 + relativePosition2[1] * self.config.obstacle_orcaradius, -relativePosition2[0] * self.config.obstacle_orcaradius + relativePosition2[1] * leg2]) / distSq2
                else:
                    # Usual situation.
                    if self.scene.obstacles[obstacle1No].convex:
                        leg1 = ti.sqrt(distSq1 - radiusSq)
                        leftLegDirection = utils.vec2([relativePosition1[0] * leg1 - relativePosition1[1] * self.config.obstacle_orcaradius, relativePosition1[0] * self.config.obstacle_orcaradius + relativePosition1[1] * leg1]) / distSq1
                    else:
                        # Left vertex non-convex left leg extends cut-off line.
                        leftLegDirection = -self.scene.obstacles[obstacle1No].direction

                    if self.scene.obstacles[obstacle2No].convex:
                        leg2 = ti.sqrt(distSq2 - radiusSq)
                        rightLegDirection = utils.vec2([relativePosition2[0] * leg2 + relativePosition2[1] * self.config.obstacle_orcaradius, -relativePosition2[0] * self.config.obstacle_orcaradius + relativePosition2[1] * leg2]) / distSq2
                    else:
                        # Right vertex non-convex right leg extends cut-off line.
                        rightLegDirection = self.scene.obstacles[obstacle1No].direction

                if flag == False:
                    # Legs can never point into neighboring edge when convex vertex, take cutoff-line of neighboring edge instead. If velocity projected on "foreign" leg, no constraint is added.

                    leftNeighborNo = self.scene.obstacles[obstacle1No].previous

                    isLeftLegForeign = False
                    isRightLegForeign = False

                    leftnei_direction = self.scene.obstacles[leftNeighborNo].direction
                    if self.scene.obstacles[obstacle1No].convex and utils.det(leftLegDirection, -leftnei_direction) >= 0.0:
                        # Left leg points into obstacle.
                        leftLegDirection = -leftnei_direction
                        isLeftLegForeign = True

                    ob2_direction = self.scene.obstacles[obstacle2No].direction
                    if self.scene.obstacles[obstacle2No].convex and utils.det(rightLegDirection, ob2_direction) <= 0.0:
                        # Right leg points into obstacle.
                        rightLegDirection = ob2_direction
                        isRightLegForeign = True

                    # Compute cut-off centers.
                    left = self.scene.obstacles[obstacle1No].point - self.pos[current_people_index]
                    right = self.scene.obstacles[obstacle2No].point - self.pos[current_people_index]
                    leftCutOff = invTimeHorizonObst * left
                    rightCutOff = invTimeHorizonObst * right
                    cutOffVector = rightCutOff - leftCutOff

                    # Project current velocity on velocity obstacle.

                    # Check if current velocity is projected on cutoff circles.
                    t = 0.5 if obstacle1No == obstacle2No else utils.abs_sq((cur_vel - leftCutOff), cutOffVector) / utils.abs_sq(cutOffVector,cutOffVector)
                    tLeft = utils.abs_sq((cur_vel - leftCutOff) , leftLegDirection)
                    tRight = utils.abs_sq((cur_vel - rightCutOff) , rightLegDirection)

                    if (t < 0.0 and tLeft < 0.0) or (obstacle1No == obstacle2No and tLeft < 0.0 and tRight < 0.0):
                        # Project on left cut-off circle.
                        unitW,_ = utils.normalize(self.vel[current_people_index] - leftCutOff)
                        line[0:2] = utils.vec2([unitW.y, -unitW.x])
                        line[2:] = leftCutOff + self.config.obstacle_orcaradius * invTimeHorizonObst * unitW
                        required = True
                    elif t > 1.0 and tRight < 0.0:
                        # Project on right cut-off circle.
                        unitW,_ = utils.normalize(self.vel[current_people_index] - rightCutOff)
                        line[0:2] = utils.vec2([unitW.y, -unitW.x])
                        line[2:] = rightCutOff + self.config.obstacle_orcaradius * invTimeHorizonObst * unitW
                        required = True
                    else:
                        # Project on left leg, right leg, or cut-off line, whichever is closest to velocity.
                        distSqCutoff = utils.INF if t < 0.0 or t > 1.0 or obstacle1No == obstacle2No else utils.abs_sq(cur_vel - (leftCutOff + t * cutOffVector),cur_vel - (leftCutOff + t * cutOffVector))
                        distSqLeft = utils.INF if tLeft < 0.0 else utils.abs_sq(cur_vel - (leftCutOff + tLeft * leftLegDirection),cur_vel - (leftCutOff + tLeft * leftLegDirection))
                        distSqRight = utils.INF if tRight < 0.0 else utils.abs_sq(cur_vel - (rightCutOff + tRight * rightLegDirection),cur_vel - (rightCutOff + tRight * rightLegDirection))

                        if distSqCutoff <= distSqLeft and distSqCutoff <= distSqRight:
                            # Project on cut-off line.
                            line[0:2] = -self.scene.obstacles[obstacle1No].direction
                            line[2:] = leftCutOff + self.config.obstacle_orcaradius * invTimeHorizonObst * utils.vec2([-line[1], line[0]])
                            required = True
                        else:
                            if distSqLeft <= distSqRight:
                                # Project on left leg.
                                if not isLeftLegForeign:
                                    line[0:2] = leftLegDirection
                                    line[2:] = leftCutOff + self.config.obstacle_orcaradius * invTimeHorizonObst * utils.vec2([-line[1], line[0]])
                                    required = True
                            else:
                                # Project on right leg.
                                if not isRightLegForeign:
                                    line[0:2] = -rightLegDirection
                                    line[2:] = rightCutOff + self.config.obstacle_orcaradius * invTimeHorizonObst * utils.vec2([-line[1], line[0]])
                                    required = True

        return required,line
    @ti.func
    def compute_ORCA_agent(self,current_people_index,other_people_index):
        """
        return an ORCA LINE 
        """
        invTimeHorizon = 1.0 / self.config.timeHorizon

        relativePosition = self.pos[other_people_index] - self.pos[current_people_index]
        relativeVelocity = self.vel[other_people_index] - self.vel[current_people_index]

        _,dist = utils.normalize(relativePosition)

        combinedRadius = self.config.ORCA_radius

        direction = utils.vec2([0.0,0.0])
        u = utils.vec2([0.0,0.0])

        if dist > combinedRadius:    
            # No collision.
            w = relativeVelocity - invTimeHorizon * relativePosition
            # Vector from cutoff center to relative velocity.
            unitW,wLength = utils.normalize(w)
            dotProduct1 = utils.abs_sq(w,relativePosition)

            if dotProduct1 < 0.0 and dotProduct1 > combinedRadius * wLength:
                # Project on cut-off circle.
                direction = utils.vec2([unitW.y, -unitW.x])
                u = (combinedRadius * invTimeHorizon - wLength) * unitW
            else:
                # Project on legs.
                leg = dist - combinedRadius

                if utils.det(relativePosition, w) > 0.0:
                    # Project on left leg.
                    direction = utils.vec2([relativePosition.x * leg - relativePosition.y * combinedRadius, relativePosition.x * combinedRadius + relativePosition.y * leg]) / (dist*dist)
                else:
                    # Project on right leg.
                    direction = -utils.vec2([relativePosition.x * leg + relativePosition.y * combinedRadius, -relativePosition.x * combinedRadius + relativePosition.y * leg]) / (dist*dist)

                dotProduct2 = utils.abs_sq(relativeVelocity,direction)
                u = dotProduct2 * direction - relativeVelocity
        else:
            # print(current_people_index,other_people_index,dist)
            # Collision. Project on cut-off circle of time timeStep.
            # Vector from cutoff center to relative velocity.
            w = relativeVelocity - self.config.invTimeStep * relativePosition
            unitW,wLength = utils.normalize(w)
            
            direction = utils.vec2([unitW.y, -unitW.x])
            u = (combinedRadius * self.config.invTimeStep - wLength) * unitW

        point = self.vel[current_people_index] + 0.5 * u
        line = utils.Line([direction[0],direction[1],point[0],point[1]])
        return line
    @ti.func
    def linear_program1(self, ORCA_lines,lineNo, radius, optVelocity, directionOpt,result):
        """
        Solves a one-dimensional linear program on a specified line subject to linear constraints defined by lines and a circular constraint.
        lineNo (int): The specified line constraint.
        optVelocity: The optimization velocity.
        directionOpt (bool): True if the direction should be optimized.
        Returns:
            bool: True if successful.
            result : the result of calculation
        """
        success = False

        dotProduct = utils.abs_sq(ORCA_lines[lineNo,2:],ORCA_lines[lineNo,0:2])
        discriminant = dotProduct*dotProduct + radius*radius - utils.abs_sq(ORCA_lines[lineNo,2:],ORCA_lines[lineNo,2:])

        if not discriminant < 0.0:
            # if discriminant < 0.0, Max speed circle fully invalidates line lineNo.
            sqrtDiscriminant = ti.sqrt(discriminant)
            tLeft = -dotProduct - sqrtDiscriminant
            tRight = -dotProduct + sqrtDiscriminant

            for i in range(lineNo):
                denominator = utils.det(ORCA_lines[lineNo,0:2], ORCA_lines[i,0:2])
                numerator = utils.det(ORCA_lines[i,0:2], ORCA_lines[lineNo,2:] - ORCA_lines[i,2:])

                if ti.abs(denominator) <= utils.EPSILON:
                    # Lines lineNo and i are (almost) parallel.
                    if numerator < 0.0:
                        success = False
                        result = utils.vec2([0.0,0.0])
                        break
                    continue

                t = numerator / denominator

                if denominator >= 0.0:
                    # Line i bounds line lineNo on the right.
                    tRight = ti.min(tRight, t)
                else:
                    # Line i bounds line lineNo on the left.
                    tLeft = ti.max(tLeft, t)

                if tLeft > tRight:
                    success = False
                    result = utils.vec2([0.0,0.0])
                    break

            if directionOpt:
                # Optimize direction.
                if utils.abs_sq(optVelocity ,ORCA_lines[lineNo,0:2]) > 0.0:
                    # Take right extreme.
                    result = ORCA_lines[lineNo,2:] + tRight * ORCA_lines[lineNo,0:2]
                else:
                    # Take left extreme.
                    result = ORCA_lines[lineNo,2:] + tLeft * ORCA_lines[lineNo,0:2]
            else:
                # Optimize closest point.
                t = utils.abs_sq(ORCA_lines[lineNo,0:2] ,(optVelocity - ORCA_lines[lineNo,2:]))

                if t < tLeft:
                    result = ORCA_lines[lineNo,2:] + tLeft * ORCA_lines[lineNo,0:2]
                elif t > tRight:
                    result = ORCA_lines[lineNo,2:] + tRight * ORCA_lines[lineNo,0:2]
                else:
                    result = ORCA_lines[lineNo,2:] + t * ORCA_lines[lineNo,0:2]

        success = True
        return success, result
    @ti.func
    def linear_program2(self,ORCA_lines,current_people_index,count_lines,optVelocity,directionOpt,result):
        """
        Solves a two-dimensional linear program subject to linear constraints defined by lines and a circular constraint.
        directionOpt (bool): True if the direction should be optimized.
        Returns the number of the line it fails on, and the number of lines if successful.
        """
        radius = self.max_vel[current_people_index]*self.config.window_size#约等于没有限制radius
        line_num = count_lines

        if directionOpt:
            # Optimize direction. Note that the optimization velocity is of unit length in this case.
            result = optVelocity * radius
        elif utils.abs_sq(optVelocity,optVelocity) > radius*radius:
            # Optimize closest point and outside circle.
            result,_ = utils.normalize(optVelocity)
            result = result * radius
        else:
            # Optimize closest point and inside circle.
            result = optVelocity

        for i in range(count_lines):
            if utils.det(ORCA_lines[i,0:2], ORCA_lines[i,2:]- result) > 0.0:
                # Result does not satisfy constraint i. Compute new optimal result.
                tempResult = result
                success, result = self.linear_program1(ORCA_lines,i, radius, optVelocity, directionOpt,result)
                if not success:
                    result = tempResult
                    line_num = i
                    break

        return line_num,result
    @ti.func
    def linear_program3(self,ORCA_lines,current_people_index,lines_to_solve, beginLine,result,numObstLines):
        """
        Solves a two-dimensional linear program subject to linear constraints defined by lines and a circular constraint.

        numObstLines (int): Count of obstacle lines.
        beginLine (int): The line on which the 2-d linear program failed.
        Returns the result of the linear program
        """
        distance = 0.0

        for i in range(beginLine, lines_to_solve):
            if utils.det(ORCA_lines[i,0:2], ORCA_lines[i,2:] - result) > distance:
                # Result does not satisfy constraint of line i.
                projLines = ti.Matrix([[0] * 4 for _ in range(self.config.ORCA_constraints)], ti.f32)
                count_projLines = 0

                for ii in range(numObstLines):
                    projLines[count_projLines,:] = ORCA_lines[ii,:]
                    count_projLines += 1

                for j in range(numObstLines, i):
                    line = utils.Line([0.0,0.0,0.0,0.0])
                    determinant = utils.det(ORCA_lines[i,0:2], ORCA_lines[j,0:2])

                    if ti.abs(determinant) <= utils.EPSILON:
                        # Line i and line j are parallel.
                        if utils.abs_sq(ORCA_lines[i,0:2],ORCA_lines[j,0:2]) > 0.0:
                            # Line i and line j point in the same direction.
                            continue
                        else:
                            # Line i and line j point in opposite direction.
                            line[2:] = 0.5 * (ORCA_lines[i,2:] + ORCA_lines[j,2:])
                    else:
                        pp = utils.det(ORCA_lines[j,0:2], ORCA_lines[i,2:] - ORCA_lines[j,2:])
                        line[2:] = ORCA_lines[i,2:] + (pp / determinant) * ORCA_lines[i,0:2]

                    line[0:2],_ = utils.normalize(ORCA_lines[j,0:2] - ORCA_lines[i,0:2])
                    projLines[count_projLines,:] = line
                    count_projLines += 1

                tempResult = result
                lineFail, result = self.linear_program2(projLines,current_people_index, count_projLines, utils.vec2([-ORCA_lines[i,1], ORCA_lines[i,0]]), True,result)
                if lineFail < count_projLines:
                    """
                    This should in principle not happen. The result is by definition already in the feasible region of this linear program. If it fails, it is due to small floating point error, and the current result is kept.
                    """
                    result = tempResult

                distance = utils.det(ORCA_lines[i,0:2], ORCA_lines[i,2:] - result)
        return result
    
    @ti.kernel
    def real_compute(self):
        for i in range(self.config.N):
            # ORCA-line process, get ORCA_new_vel for each agent
            neighbour_count = 0
            obstacle_count = 0
            dense_length = 0.0
            ORCA_lines = ti.Matrix([[0] * 4 for _ in range(self.config.ORCA_constraints)], ti.f32)
            # compute each obstacle's ORCA
            for obstacleNo in range(self.scene.obstacles_number):             
                required, orca_line = self.compute_ORCA_obstacle(i, obstacleNo, ORCA_lines, obstacle_count)
                if required == True:
                    #print(i,obstacleNo,orca_line)
                    ORCA_lines[obstacle_count,:] = orca_line
                    obstacle_count += 1
                if obstacle_count >= self.config.max_neighbour:
                    break
            for j in range(self.config.N):
                if i!=j and neighbour_count <= self.config.max_neighbour:
                    pos_diff = self.pos[i]-self.pos[j]
                    _, diff_length = utils.normalize(pos_diff)
                    if diff_length < (self.config.ORCA_radius/2):   
                        ORCA_lines[obstacle_count+neighbour_count,:] = self.compute_ORCA_agent(i,j)                  
                        dense_length += diff_length
                        neighbour_count += 1
            lines_to_solve = neighbour_count + obstacle_count
            optVelocity = self.get_agent_desired_force(i)
            lineFail,self.new_velocity[i] = self.linear_program2(ORCA_lines,i,lines_to_solve,optVelocity,False,self.new_velocity[i])
            if lineFail < lines_to_solve:
                self.new_velocity[i] = self.linear_program3(ORCA_lines,i,lines_to_solve, lineFail,self.new_velocity[i],obstacle_count)
            if self.config.dense_sense == 1:# use the density filter
                local_density = ti.exp(dense_length/(2*self.config.dense_theta*self.config.dense_theta))/ti.sqrt(2*3.1415*self.config.dense_theta)
                max_speed = self.max_vel[i]/(1+ti.exp(self.config.dense_k*(self.config.dense_p-local_density)))
                _,v_len = utils.normalize(self.new_velocity[i])
                if v_len > max_speed:
                    self.new_velocity[i] = utils.capped(self.new_velocity[i],max_speed)
                

    def compute(self):
        """
        simulator will call this function first
        """
        self.real_compute()
        if self.config.analysis == 1:
            self.analysis()
            self.collision_detect()

    @ti.kernel
    def update(self):
        for i in range(self.config.N):
            new_vel = self.new_velocity[i]
            new_vel = utils.limit(new_vel,self.max_vel[i])

            # if close enough to target，stop
            destination_vector = self.desiredpos[i] - self.pos[i]
            _, dist = utils.normalize(destination_vector)
            if dist < self.config.stop_radius:
                new_vel = ti.Vector([0.0,0.0])

            self.vel[i] = utils.expand(new_vel,self.config.WINDOW_WIDTH,self.config.WINDOW_HEIGHT)
            self.pos[i] += new_vel * self.config.sim_step


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