import pygame
from pygame.locals import QUIT, KEYDOWN
from environment import Environment
from robot import DifferentialDriveRobot
from rnn import SimpleFeedforwardNet
from time import strftime
from torch import load
from sys import argv

# for potential visualization
USE_VISUALIZATION = True

# to pause the execution
PAUSE = False

# Initialize Pygame
pygame.init()

# Set up environment
width, height = 800, 800  # cm
env = Environment(width, height)
sim_speed = 1

# (simulated) time taken for one cycle of the robot executing its algorithm
robot_timestep = 0.1  # in seconds (simulated time)

FILE = argv[1]
rnn = load(FILE, weights_only=False)
robot = DifferentialDriveRobot(env, width/2-100, height/2-100, 0, rnn, sim_speed)

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Robot Kinematic Visualiser")

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
            robot_pose = robot.move(robot_timestep)

        if USE_VISUALIZATION:
            screen.fill((0, 0, 0))
            # draw environment
            env.draw(screen)
            # draw robot
            robot.draw(screen)

            # warn the user if collision happened
            if robot.collided:
                # Draw the animation
                drawBoom()

            pygame.display.flip()
            pygame.display.update()


    print("total execution time:", (pygame.time.get_ticks() -
          start_time) / 1000, "seconds")  # runtime in seconds
    print("Robot score:", robot.get_score())
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
