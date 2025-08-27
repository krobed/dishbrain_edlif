import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

class CustomCarEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width=20.0, height=20.0):
        super(CustomCarEnv, self).__init__()

        # Environment boundaries
        self.width = width
        self.height = height
        
        # Define action and observation space
        # Two actions: [velocity_left_wheel, velocity_right_wheel]
        self.action_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32)
        
        # Observation space: [x_position, y_position, orientation]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.pi]), 
            high=np.array([self.width, self.height, np.pi]), 
            dtype=np.float32)
        
        # Initial state
        self.state = None
        self.reset()

        # Rendering attributes
        self.fig, self.ax = None, None

        # Circle obstacles
        #self.obstacles = [(3, 3), (7, 7), (5, 3), (3, 4), (1, 2), (2, 1), (20, 6), (6, 20), (8, 8), (9, 9), (8, 5), (7, 7)]
        #self.obstacles = [(3, 3), (7, 7), (5, 3)]
        #self.obstacles = [(9, 5)]
        self.obstacles = [(3, 3), (7, 7), (5, 3), (18, 17), (3, 18), (18, 3)]
        #self.obstacles = None

        # List to store the trajectory of the robot
        self.trajectory = []

        self.key_action_map = {
            'w': [1.0, 1.0],    # Forward
            's': [-1.0, -1.0],  # Backward
            'a': [-1.0, 1.0],   # Left turn
            'd': [1.0, -1.0]    # Right turn
        }

    def step(self, action):
        self.velocity_left_wheel, self.velocity_right_wheel = action

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
        distance = self._compute_distance_sensor()
        fr_reward = self._encode_distance(distance)
        
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
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            self.ax.set_aspect('equal')

            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)

        # Draw maze boundaries
        self.ax.add_patch(
            Rectangle(
                (0, 0), 
                self.width, 
                self.height, 
                fill=None, 
                edgecolor='black'))

        # Draw obstacles (example circles)
        if self.obstacles:
            for obstacle in self.obstacles:
                self.ax.add_patch(Circle(obstacle, 0.5, color='red'))

        # Draw the car
        car_x, car_y, car_phi = self.state
        car = Rectangle(
            (car_x - 0.5, car_y - 0.5), 
            1, 
            1, 
            color='blue')
        # Rotate the car based on the orientation
        t = plt.gca().transData
        rot = plt.matplotlib.transforms.Affine2D().rotate_around(
            car_x, car_y, car_phi)
        car.set_transform(rot + t)
        # Add the car to the plot
        self.ax.add_patch(car)

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
        return np.random.uniform(2, 8, 2).tolist() + [np.random.uniform(-np.pi, np.pi)]

    def _update_state(self, velocity_left_wheel, velocity_right_wheel):
        # Radius of the wheels
        r = 1.0  
        # Distance between the wheels
        a = 0.5  
        # Time step
        dt = 1.0  

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
        new_x = np.clip(new_x, 0, self.width)
        new_y = np.clip(new_y, 0, self.height)
        new_phi = (new_phi + np.pi) % (2 * np.pi) - np.pi  # Keep phi in the range [-pi, pi]

        return np.array([new_x, new_y, new_phi])

    def _compute_distance_sensor(self):
        # Calculate the distance sensor reading based on the current state
        # Here, it can be the distance to the nearest obstacle
        car_x, car_y, _ = self.state
        distances = []
        if self.obstacles:
            for ox, oy in self.obstacles:
                distances.append(np.linalg.norm([car_x - ox, car_y - oy]))
        car_distance_to_boundary = np.min([car_x, car_y, self.width - car_x, self.height - car_y])
        distances.append(car_distance_to_boundary)
        return np.min(distances)

    def _encode_distance(self, distance):
        return 20.0 / (1 + 20 * distance)

    def _is_collision(self, state):
        car_x, car_y, _ = state
        if self.obstacles:
            for ox, oy in self.obstacles:
                if np.linalg.norm([car_x - ox, car_y - oy]) < 1:
                    return True
        return False

    def _is_done(self, state):
        # If crash with any obstacle or the outer wall, the episode is done
        car_x, car_y, _ = state
        if car_x <= 0 or car_x >= self.width or car_y <= 0 or car_y >= self.height:
            return True
        if self._is_collision(state):
            return True
        return False

    def _on_key_press(self, event):
        if event.key in self.key_action_map:
            action = self.key_action_map[event.key]
            self.step(action)
            self.render()