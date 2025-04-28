import pygame
from pygame.locals import QUIT, KEYDOWN
from shapely.lib import simplify_preserve_topology
from environment import Environment
from robot import DifferentialDriveRobot
from rnn import SimpleFeedforwardNet
from time import time
import numpy as np
from random import shuffle

# for potential visualization
USE_VISUALIZATION = True

# to pause the execution
PAUSE = False

# Initialize Pygame
pygame.init()

# Set up environment
width, height = 1200, 800  # cm
env = Environment(width, height)

# Simulation time ratio
sim_ratio = 20

# (simulated) time taken for one cycle of the robot executing its algorithm
robot_timestep = 0.1  # in seconds (simulated time)


def spawn(rnn=None):

    if rnn is None:
        rnn = SimpleFeedforwardNet(6, 4, 2)

    return DifferentialDriveRobot(env, width/2-100, height/2-100, 0, rnn, sim_ratio)


ROBOTS = [spawn() for _ in range(10)]


screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Robot Kinematic Simulator")


def main():
    global USE_VISUALIZATION, PAUSE, ROBOTS
    start_time = pygame.time.get_ticks()
    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == pygame.K_h:  # use space key to toggle between visualization and headless
                    USE_VISUALIZATION = not USE_VISUALIZATION
                    print("Visualization is",
                          "on" if USE_VISUALIZATION else "off")
                if event.key == pygame.K_SPACE:
                    PAUSE = not PAUSE

        if not PAUSE:
            # simulate one execution cycle of the robot
            robot_pose = [robot.move(robot_timestep) for robot in ROBOTS]

        if USE_VISUALIZATION:
            screen.fill((0, 0, 0))
            # draw environment
            env.draw(screen)
            # draw robot
            [robot.draw(screen) for robot in ROBOTS]

            # warn the user if collision happened
            if any([robot.collided for robot in ROBOTS]):
                print("Collision!")
                # Draw the animation
                drawBoom()

            pygame.display.flip()
            pygame.display.update()

        sim_time = ((pygame.time.get_ticks() - start_time) / 1000) * sim_ratio

        if sim_time >= 20:

            fitsrt = sorted(ROBOTS, key = lambda r: r.get_score())
            best = fitsrt[0].get_nn()

            if all([robot.get_score() == 0 for robot in ROBOTS]):
                print("Spawning new generation")
                ROBOTS = [spawn() for _ in range(10)]
            else:
                ROBOTS = [spawn(best)] + [spawn(best.clone_and_mutate()) for _ in range(5)] + [spawn(ROBOTS[i].get_nn().clone_and_mutate()) for i in range(1, 4)]
            print(len(ROBOTS))

            # reset screen and time
            screen.fill((0, 0, 0))
            start_time = pygame.time.get_ticks()

    print("total execution time:", (pygame.time.get_ticks() -
          start_time) / 1000, "seconds")  # runtime in seconds

    # Quit Pygame
    pygame.quit()


def drawBoom():
    # pygame.font.Font(self.font, size)
    font = pygame.font.SysFont("comicsansms", 172)
    text_surface = font.render('BOOM', True, (255, 0, 0))
    text_rect = text_surface.get_rect(center=(width/2, height/2))
    screen.blit(text_surface, text_rect)


if __name__ == "__main__":
    main()
