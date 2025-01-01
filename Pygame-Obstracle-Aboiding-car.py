import pygame
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Initialize Pygame
pygame.init()

# Game Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CAR_WIDTH = 50
CAR_HEIGHT = 100
ROAD_WIDTH = 400
LANE_WIDTH = ROAD_WIDTH // 3
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Initialize Screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Car Racing Game")
clock = pygame.time.Clock()

# Load and scale car images
PLAYER_CAR_IMAGE = pygame.image.load("player_car.png")
PLAYER_CAR_IMAGE = pygame.transform.scale(PLAYER_CAR_IMAGE, (CAR_WIDTH, CAR_HEIGHT))

ENEMY_CAR_IMAGE = pygame.image.load("enemy_car.png")
ENEMY_CAR_IMAGE = pygame.transform.scale(ENEMY_CAR_IMAGE, (CAR_WIDTH, CAR_HEIGHT))

class CarGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.car_x = SCREEN_WIDTH // 2 - CAR_WIDTH // 2
        self.car_y = SCREEN_HEIGHT - CAR_HEIGHT - 10
        self.enemy_x = random.randint(SCREEN_WIDTH // 2 - ROAD_WIDTH // 2, SCREEN_WIDTH // 2 + ROAD_WIDTH // 2 - CAR_WIDTH)
        self.enemy_y = -CAR_HEIGHT
        self.enemy_speed = 10
        self.score = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        return np.array([
            self.car_x / SCREEN_WIDTH,
            self.car_y / SCREEN_HEIGHT,
            self.enemy_x / SCREEN_WIDTH,
            self.enemy_y / SCREEN_HEIGHT,
        ])

    def step(self, action):
        # Actions: 0 = Left, 1 = Stay, 2 = Right
        if action == 0:
            self.car_x -= LANE_WIDTH
        elif action == 2:
            self.car_x += LANE_WIDTH

        # Keep car within road boundaries
        self.car_x = max(SCREEN_WIDTH // 2 - ROAD_WIDTH // 2, min(self.car_x, SCREEN_WIDTH // 2 + ROAD_WIDTH // 2 - CAR_WIDTH))

        # Move enemy car
        self.enemy_y += self.enemy_speed

        # Check for collision
        if (
            self.car_x < self.enemy_x + CAR_WIDTH and
            self.car_x + CAR_WIDTH > self.enemy_x and
            self.car_y < self.enemy_y + CAR_HEIGHT and
            self.car_y + CAR_HEIGHT > self.enemy_y
        ):
            self.done = True
            reward = -100
        elif self.enemy_y > SCREEN_HEIGHT:
            # Enemy car passed
            self.enemy_y = -CAR_HEIGHT
            self.enemy_x = random.randint(SCREEN_WIDTH // 2 - ROAD_WIDTH // 2, SCREEN_WIDTH // 2 + ROAD_WIDTH // 2 - CAR_WIDTH)
            self.score += 1
            reward = 10
        else:
            reward = 0

        return self.get_state(), reward, self.done

    def render(self):
        screen.fill(WHITE)
        pygame.draw.rect(screen, BLACK, [SCREEN_WIDTH // 2 - ROAD_WIDTH // 2, 0, ROAD_WIDTH, SCREEN_HEIGHT])
        screen.blit(PLAYER_CAR_IMAGE, (self.car_x, self.car_y))
        screen.blit(ENEMY_CAR_IMAGE, (self.enemy_x, self.enemy_y))
        pygame.display.flip()

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name + ".weights.h5")

    def save(self, name):
        self.model.save_weights(name + ".weights.h5")

def train_agent(episodes=1000):
    env = CarGame()
    state_size = env.get_state().shape[0]
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break
            env.render()
        agent.replay(batch_size)
        if (e + 1) % 50 == 0:
            agent.save("car_racing_dqn")

    agent.save("car_racing_dqn")

if __name__ == "__main__":
    train_agent()
