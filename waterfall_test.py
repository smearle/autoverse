# Import pygame and random modules
import pygame
import random
import numpy as np

# Define constants for colors and grid size
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GRID_SIZE = 20
TILE_SIZE = 10

# Initialize pygame and create a window
pygame.init()
screen = pygame.display.set_mode((GRID_SIZE * TILE_SIZE, GRID_SIZE * TILE_SIZE))
pygame.display.set_caption("Water Simulation")

# Create a list of lists to store the grid cells
grid = []
for i in range(GRID_SIZE):
    row = []
    for j in range(GRID_SIZE):
        # Randomly assign empty or wall tiles
        if random.random() < 0.2:
            row.append("wall")
        else:
            row.append("empty")
    grid.append(row)

water_grid = np.zeros((GRID_SIZE, GRID_SIZE), np.uint8)

# Create a clock to control the game loop
clock = pygame.time.Clock()

# Create a boolean variable to indicate if the game is running
running = True

# Start the game loop
while running:
    # Set the frame rate to 10 FPS
    clock.tick(10)

    # Handle events
    for event in pygame.event.get():
        # If the user closes the window, quit the game
        if event.type == pygame.QUIT:
            running = False
        # If the user clicks the mouse, place water on the grid
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Get the mouse position and convert it to grid coordinates
            x, y = pygame.mouse.get_pos()
            i = y // TILE_SIZE
            j = x // TILE_SIZE
            # If the grid cell is empty, place water with mass 7
            if grid[i][j] == "empty":
                grid[i][j] = "water"
                water_grid[i][j] = 7

    # Update the grid
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # If the cell is water, check its mass and spawn new water tiles if possible
            if grid[i][j] == "water":
                mass = water_grid[i][j]
                if mass > 1:
                    # Check the four adjacent cells and place water with mass - 1 if they are empty
                    if i > 0 and grid[i - 1][j] == "empty":
                        grid[i - 1][j] = "water"
                        water_grid[i - 1][j] = mass - 1
                    if i < GRID_SIZE - 1 and grid[i + 1][j] == "empty":
                        grid[i + 1][j] = "water"
                        water_grid[i + 1][j] = mass - 1
                    if j > 0 and grid[i][j - 1] == "empty":
                        grid[i][j - 1] = "water"
                        water_grid[i][j - 1] = mass - 1
                    if j < GRID_SIZE - 1 and grid[i][j + 1] == "empty":
                        grid[i][j + 1] = "water"
                        water_grid[i][j + 1] = mass - 1

    # Draw the grid on the screen
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # Choose the color based on the cell type
            if grid[i][j] == "empty":
                color = WHITE
            elif grid[i][j] == "wall":
                color = BLACK
            elif grid[i][j] == "water":
                color = BLUE
            # Draw a rectangle with the chosen color on the screen
            pygame.draw.rect(screen, color, (j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE))

    # Update the display
    pygame.display.flip()

# Quit pygame and exit the program
pygame.quit()