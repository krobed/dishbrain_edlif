import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

class CustomCircleCarEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,n_sensors,sensor_angle,time,stimulation=2.0, radious=4.0, n_obstacles=3):
        super(CustomCircleCarEnv, self).__init__()
        self.time = time
        # Environment boundaries
        self.radious = radious
        self.n_sensors=n_sensors
        self.sensors_angle = sensor_angle
        self.stimulation = stimulation
        # Define action and observation space
        # Two actions: [velocity_left_wheel, velocity_right_wheel]
        self.action_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([5.0, 5.0]), 
            dtype=np.float32)
        
        # Observation space: [x_position, y_position, orientation]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.pi]), 
            high=np.array([radious, radious, np.pi]), 
            dtype=np.float32)
        
        # Initial state
        self.trajectory = np.array([],dtype=np.float64)
        self.state = np.array([None]*3)
        self.reset()

        # Rendering attributes
        self.fig, self.ax = None, None
        self.crash = []
        # Circle obstacles
        #self.obstacles = [(3, 3), (7, 7), (5, 3), (3, 4), (1, 2), (2, 1), (10, 6), (6, 10), (8, 8), (9, 9), (8, 5), (7, 7)]
        self.obstacles = []
        if n_obstacles ==3:
            self.obstacles = [(self.radious*2.5/4,self.radious*6/4), (self.radious*6/4,self.radious*5/4), (self.radious*3/4,self.radious/2)]
        else:
            if n_obstacles%4 ==0:
                for i in range(n_obstacles//4):
                    self.obstacles=np.random.randint(17,self.radious,2)
                    self.obstacles.append([np.random.randint(self.radious,self.radious*2-17),np.random.randint(17,self.radious)])
                    self.obstacles.append([np.random.randint(17,self.radious),np.random.randint(self.radious,self.radious*2-17)])
                    self.obstacles.append([np.random.randint(self.radious,self.radious*2-17),np.random.randint(self.radious,self.radious*2-17)])
            elif n_obstacles%2==0:
                for i in range(n_obstacles//2):
                    self.obstacles.append([np.random.randint(self.radious,self.radious*2-17),np.random.randint(17,self.radious*2-17)])
                    self.obstacles.append([np.random.randint(17,self.radious),np.random.randint(17,self.radious*2-17)])
            else:
                self.obstacles=np.random.randint(17,self.radious*2-17,(n_obstacles,2))

        # for o in range(n_obstacles):
        #     if np.linalg.norm([self.obstacles[o][0]-self.radious, self.obstacles[o][1]-self.radious])>40:
        #         self.obstacles[o] = [np.random.randint(17,self.radious*2 -17),np.random.randint(17,self.radious*2 -17)]

        # print(self.obstacles)
        #self.obstacles = [(9, 5)]
        #self.obstacles = [(3, 3), (7, 7), (5, 3), (18, 17), (3, 18), (18, 3)]
        #self.obstacles = None

        # List to store the trajectory of the robot
     

        self.key_action_map = {
            'w': [10.0, 10.0],    # Forward
            's': [-10.0, -10.0],  # Backward
            'a': [-10.0, 10.0],   # Left turn
            'd': [10.0, -10.0]    # Right turn
        }

    def is_facing_edge(self,threshold_angle=np.pi/4):
        """
        Determines if the robot is facing the edge of a circle.
        
        Args:
            x, y: robot's current position
            theta: robot's orientation in radians
            R: radius of the circle
            threshold_angle: allowed angular deviation (in radians) from directly facing outward

        Returns:
            True if the robot is facing the edge, False otherwise
        """
        x,y,theta = self.state
        x -= self.radious
        y -= self.radious
        # If robot is at the center, direction is undefined
        distance_from_center = np.hypot(x, y)
        if distance_from_center == 0:
            return False

        # Normalize outward vector from center to robot
        outward_x = x / distance_from_center
        outward_y = y / distance_from_center

        # Robot's facing direction
        facing_x = np.cos(theta)
        facing_y = np.sin(theta)

        # Angle between facing direction and outward direction
        dot_product = facing_x * outward_x + facing_y * outward_y
        dot_product = max(min(dot_product, 1), -1)  # Clamp for safety
        angle = np.arccos(dot_product)

        return angle < threshold_angle

    def step(self, action):
        self.velocity_left_wheel, self.velocity_right_wheel = action
        
        # Update car position and orientation based on wheel velocities
        #current_state = self.state.copy()
        self.state = self._update_state(
            self.velocity_left_wheel, 
            self.velocity_right_wheel, 'foward')

        # If collision, revert to the previous state
        #if self._is_collision(self.state):
        #    self.state = current_state

        # Append current position to the trajectory
        self.trajectory = np.append(self.trajectory,[[self.state[0], self.state[1]]], axis=0)
        

        # Compute reward (customize this as per your environment dynamics)
        distance = self._compute_distance_sensor()
        fr_reward = []
        for d in distance:
            fr_reward.append(self._encode_distance(d))
        
        # Check if the episode is done
        done = self._is_done(self.state)

        # Additional info (optional)
        info = {}

        return self.state, fr_reward, done, info
    
    def reset(self):
        # Reset the state to the initial state
        if self.state.all() !=None:
            self.crash.append([self.trajectory[-1][0],self.trajectory[-1][1]])
            for _ in range(5):
                _,_,done,_ = self.step((-(0.5//self.time),-(0.5//self.time)))
            if done:
                self.trajectory = self.trajectory[:-5]
                
                self.state[:2] = [self.trajectory[-int(2//self.time)][0],self.trajectory[-int(2//self.time)][1]]
                self.trajectory = np.append(self.trajectory,[[self.state[0], self.state[1]]], axis=0)
                
        else:
            self.state = self._initial_state()
            self.trajectory = np.array([self.state[:2]],dtype= np.float64)
        return self.state

    def render(self, mode='human', close=False):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6,6))
            self.ax.set_xlim(0, self.radious*2)
            self.ax.set_ylim(0, self.radious*2)
            self.ax.set_aspect('equal')
            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
            
        self.ax.clear()
        self.ax.set_xlim(0, self.radious*2)
        self.ax.set_ylim(0, self.radious*2)
        if self.crash!=[]:
            for i in range(len(self.crash)):
                if i==0:
                    self.ax.scatter(self.crash[i][0],self.crash[i][1],c='r', hatch='o', label=f'Crash')
                    
                else:
                    self.ax.scatter(self.crash[i][0],self.crash[i][1],c='r', hatch='o')
                    
        init= self.trajectory[0][0], self.trajectory[0][1]
        self.ax.scatter(init[0],init[1],facecolors='none', edgecolors='g', hatch='O', label='Init')
        self.ax.scatter(self.state[0], self.state[1], c='r',marker='x',label= 'Car')
        
        # Draw maze boundaries
        self.ax.add_patch(
            Circle(
                (self.radious, self.radious), 
                self.radious, 
                fill=None, 
                edgecolor='black'))

        # Draw obstacles (example circles)
        if len(self.obstacles)>0:
            for obstacle in self.obstacles:
                self.ax.add_patch(Circle((obstacle[0],obstacle[1]), 0.035, fill=None,color='red'))

        # Draw the car
        car_x, car_y, car_phi = self.state
        car = Circle(
            (car_x, car_y), 
            0.035,
            #angle=np.degrees(car_phi), 
            color='blue')
        # Rotate the car based on the orientation
        t = plt.gca().transData
        rot = plt.matplotlib.transforms.Affine2D().rotate_around(
            car_x, car_y, car_phi)
        car.set_transform(rot + t)
        # Add the car to the plot
        # self.ax.add_patch(car)
        sensors = self.sensors_angle
        sensors_place = []
        self.ax.set_axis_off()
        # Get sensor position
        for i in range(self.n_sensors):
            car_s_x =  car_x + 0.035*np.cos(car_phi + sensors[i])
            car_s_y =  car_y + 0.035*np.sin(car_phi + sensors[i])
            sensors_place.append([car_s_x,car_s_y])
        self.sensors = sensors_place
        # plt.scatter(car_x,car_y, label = 'centre')
        # Print sensors
        # for i in sensors_place:
        #     plt.scatter(i[0],i[1], c='r', marker='.')

        if len(self.trajectory) > 1:
            trajectory = np.array(self.trajectory)
            self.ax.plot(trajectory[:, 0], trajectory[:, 1], color='blue', label='Trajectory')
            
        self.ax.legend(bbox_to_anchor=(0.85,0.9))
        # plt.draw()
        # plt.pause(0.001)
        
    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def _initial_state(self):
        # Define the initial state: [x_position, y_position, orientation]
        # return np.array([5.0, 5.0, 0.0])  # Example initial distance sensor reading, position, and orientation
        # return np.array([np.random.randint(15,65),np.random.randint(15,65),0.0]).tolist()
        return np.array([self.radious,self.radious, 0.0]).tolist() 

    def _update_state(self, velocity_left_wheel, velocity_right_wheel, direction):
        # Radius of the wheels
        r = 0.01
        # Distance between the wheels
        a = 0.07
        # Time step
        dt = self.time
        # Current state
        x, y, phi = self.state
        # Compute linear and angular velocities
        v = r * (velocity_right_wheel + velocity_left_wheel) / 2
        omega = r * (velocity_right_wheel - velocity_left_wheel) / (2*a)

        # Update state using the kinematic model
        if direction=='backward':
            v = -v

        dx = v * np.cos(phi) * dt 
        dy = v * np.sin(phi) * dt 
        dphi = omega * dt
        new_x = x + dx
        new_y = y + dy
        new_phi = phi + dphi

        # Clip to maze boundaries
        new_x = np.clip(new_x, 0, self.radious*2)
        new_y = np.clip(new_y, 0, self.radious*2)
        new_phi = (new_phi + np.pi) % (2 * np.pi) - np.pi  # Keep phi in the range [-pi, pi]

        return np.array([new_x, new_y, new_phi])

    def ray_circle_intersection(self, x0, y0, dx, dy, cx, cy, r):
        fx = x0 - cx
        fy = y0 - cy
        a = dx**2 + dy**2
        b = 2 * (fx * dx + fy * dy)
        c = fx**2 + fy**2 - r**2
        disc = b**2 - 4 * a * c
        if disc < 0:
            return None  # no intersection
        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        t_values = [t for t in [t1, t2] if t >= 0]
        if not t_values:
            return None
        return min(t_values)

    def compute_sensor_distances(self):
        distances = []
        x_r, y_r, r_robot = self.state[0],self.state[1], 0.035
        for theta in self.sensors_angle:
            x_s = x_r + r_robot * np.cos(theta+ self.state[2])
            y_s = y_r + r_robot * np.sin(theta+ self.state[2])
            dx = np.cos(theta+ self.state[2])
            dy = np.sin(theta+ self.state[2])

            # Environment boundary
            d_env = self.ray_circle_intersection(x_s, y_s, dx, dy,self.radious,self.radious,self.radious)
            min_dist = float('inf') if d_env is None else d_env

            # Obstacles
            r_o = 0.035
            for (x_o, y_o) in self.obstacles:
                d_obs = self.ray_circle_intersection(x_s, y_s, dx, dy, x_o, y_o, r_o)
                if d_obs is not None:
                    min_dist = min(min_dist, d_obs)

            distances.append(min_dist)
        return distances

    def _compute_distance_sensor(self):
        # Calculate the distance sensor reading based on the current state
        # Here, it can be the distance to the nearest obstacle
        # n_sensors = []
        # 
        return self.compute_sensor_distances()

    def _encode_distance(self, distance):
        if distance<=0.05:
            return 4 + (self.stimulation-4)*(0.05-distance)/0.05
        else:
            return 0

    def _is_collision(self, state):
        car_x, car_y, _ = state
        if len(self.obstacles)>0:
            for ox, oy in self.obstacles:
                if np.linalg.norm([car_x - ox, car_y - oy]) < 0.07:
                    return True
        return False

    def _is_done(self, state):
        # If crash with any obstacle or the outer wall, the episode is done
        car_x, car_y, _ = state
        if self.radious - 0.035 - np.linalg.norm([car_x - self.radious, car_y - self.radious]) <= 0:
            return True
        if self._is_collision(state):
            return True
        return False

    def _on_key_press(self, event):
        if event.key in self.key_action_map:
            action = self.key_action_map[event.key]
            self.step(action)
            self.render()