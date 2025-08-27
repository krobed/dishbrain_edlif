import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# -----------------------------------------------------------------------------------------------
# This code was made to test the response of the robot with different stimulations.
# To achieve that its necessary to set a firing rate for each sensor (this is made at fr_test.py).
# Test have been made with a non zero firing rate for 1 sensor, the response is faster
# as the firing rate increases.
# REMEMBER TO CHANGE __init__.py.
# -----------------------------------------------------------------------------------------------



class CustomCircleCarEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width=20.0, height=20.0):
        super(CustomCircleCarEnv, self).__init__()

        # Environment boundaries
        self.width = width
        self.height = height
        
        # Define action and observation space
        # Two actions: [velocity_left_wheel, velocity_right_wheel]
        self.action_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([5.0, 5.0]), 
            dtype=np.float32)
        
        # Observation space: [x_position, y_position, orientation]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.pi]), 
            high=np.array([40.0, 40.0, np.pi]), 
            dtype=np.float32)
        
        # Initial state
        self.state = None
        self.reset()

        # Rendering attributes
        self.fig, self.ax = None, None

        # Circle obstacles
        #self.obstacles = [(3, 3), (7, 7), (5, 3), (3, 4), (1, 2), (2, 1), (10, 6), (6, 10), (8, 8), (9, 9), (8, 5), (7, 7)]
        #self.obstacles = [(25,60), (60,50), (30,20)]
        #self.obstacles = [(9, 5)]
        #self.obstacles = [(3, 3), (7, 7), (5, 3), (18, 17), (3, 18), (18, 3)]
        self.obstacles = None

        # List to store the trajectory of the robot
        self.trajectory = []

        self.key_action_map = {
            'w': [10.0, 10.0],    # Forward
            's': [-10.0, -10.0],  # Backward
            'a': [-10.0, 10.0],   # Left turn
            'd': [10.0, -10.0]    # Right turn
        }

    def step(self, action,n_sensors):
        self.velocity_left_wheel, self.velocity_right_wheel = action
        self.n_sensors=n_sensors
        # Update car position and orientation based on wheel velocities
        #current_state = self.state.copy()
        self.state = self._update_state(
            self.velocity_left_wheel, 
            self.velocity_right_wheel)

        # If collision, revert to the previous state
        #if self._is_collision(self.state):
        #    self.state = current_state

        # Append current position to the trajectory
        self.trajectory.append(self.state[:2])

        # Compute reward (customize this as per your environment dynamics)
        distance = self._compute_distance_sensor(self.n_sensors)
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
        self.state = self._initial_state()
        self.trajectory = [self.state[:2]]
        return self.state

    def render(self, mode='human', close=False):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(0, 80)
            self.ax.set_ylim(0, 80)
            self.ax.set_aspect('equal')

            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        self.ax.clear()
        self.ax.set_xlim(0, 80)
        self.ax.set_ylim(0, 80)

        # Draw maze boundaries
        self.ax.add_patch(
            Circle(
                (40, 40), 
                40, 
                fill=None, 
                edgecolor='black'))

        # Draw obstacles (example circles)
        

        # Draw the car
        car_x, car_y, car_phi = self.state
        car = Circle(
            (car_x, car_y), 
            3.5,
            #angle=np.degrees(car_phi), 
            color='blue')
        # Rotate the car based on the orientation
        t = plt.gca().transData
        rot = plt.matplotlib.transforms.Affine2D().rotate_around(
            car_x, car_y, car_phi)
        car.set_transform(rot + t)
        # Add the car to the plot
        self.ax.add_patch(car)
        sensors = self.sensors_angle
        sensors_place = []

        for i in range(self.n_sensors):
            car_s_x =  car_x + 3.5*np.cos(car_phi + sensors[i])
            car_s_y =  car_y + 3.5*np.sin(car_phi + sensors[i])
            sensors_place.append([car_s_x,car_s_y])
        self.sensors = sensors_place
        plt.scatter(car_x,car_y, label = 'centre')
        for i in sensors_place:
            plt.scatter(i[0],i[1], c='r', marker='.')
         # Add text with the car's position
        self.ax.text(
            0.5, 1.05, 
            f'Car Position: ({car_x:.2f}, {car_y:.2f})', 
            horizontalalignment='center', 
            verticalalignment='center', 
            transform=self.ax.transAxes, 
            fontsize=12, 
            color='black')

        if len(self.trajectory) > 1:
            trajectory = np.array(self.trajectory)
            self.ax.plot(trajectory[:, 0], trajectory[:, 1], color='blue')

        plt.draw()
        plt.pause(0.001)

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def _initial_state(self):
        # Define the initial state: [x_position, y_position, orientation]
        #return np.array([5.0, 5.0, 0.0])  # Example initial distance sensor reading, position, and orientation
        return np.array([np.random.randint(15,65),np.random.randint(15,65),0.0]).tolist()
        #return np.array([40.0,40.0, 0.0]).tolist() 

    def _update_state(self, velocity_left_wheel, velocity_right_wheel):
        # Radius of the wheels
        r = 2.0  
        # Distance between the wheels
        a = 7
        # Time step
        dt = 0.1  

        # Current state
        x, y, phi = self.state
        # Compute linear and angular velocities
        v = r * (velocity_right_wheel + velocity_left_wheel) / 2
        omega = r * (velocity_right_wheel - velocity_left_wheel) / (2 * a)

        # Update state using the kinematic model
        dx = v * np.cos(phi) * dt
        dy = v * np.sin(phi) * dt
        dphi = omega * dt
        new_x = x + dx
        new_y = y + dy
        new_phi = phi + dphi

        # Clip to maze boundaries
        new_x = np.clip(new_x, 0, 80)
        new_y = np.clip(new_y, 0, 80)
        new_phi = (new_phi + np.pi) % (2 * np.pi) - np.pi  # Keep phi in the range [-pi, pi]

        return np.array([new_x, new_y, new_phi])

    def _compute_distance_sensor(self, n_sensors):
        # Calculate the distance sensor reading based on the current state
        # Here, it can be the distance to the nearest obstacle
        # n_sensors = []
        # 
        car_x, car_y, phi = self.state
        if n_sensors ==1:
            self.sensors_angle = [0]
        else:
            self.sensors_angle = np.linspace(-np.pi,np.pi,self.n_sensors+1)[:-1]
            
        distances = []
        for i in range(n_sensors):
            distances.append([])
            car_s_x =  car_x + 3.5*np.cos(phi + self.sensors_angle[i])
            car_s_y =  car_y + 3.5*np.sin(phi + self.sensors_angle[i])
           
            car_distance_to_boundary = 40- np.linalg.norm([car_s_x - 40, car_s_y - 40])
            distances[i].append(max(car_distance_to_boundary,0))
        return [np.min(distance) for distance in distances]

    def _encode_distance(self, distance):
        if distance<=5:
            if distance <=0:
                return 5
            elif distance <=3:
                return 3
            else:
                return 2
        else:
            return 0

    def _is_collision(self, state):
        car_x, car_y, _ = state
        
        return False

    def _is_done(self, state):
        # If crash with any obstacle or the outer wall, the episode is done
        car_x, car_y, _ = state
        if (car_x - 40)**2 + (car_y - 40)**2 >= (36.5)**2:
            return True
       
        return False

    def _on_key_press(self, event):
        if event.key in self.key_action_map:
            action = self.key_action_map[event.key]
            self.step(action)
            self.render()