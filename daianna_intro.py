import shutil
import time
import os
# Set terminal width to 50 characters
columns, lines = shutil.get_terminal_size((50, 20))
shutil.get_terminal_size((50, lines))

def play_intro():
    # Read the contents of vanity.txt
    with open('vanity.txt', 'r') as file:
        ascii_art = file.read()

    # Split the ASCII art into lines
    ascii_lines = ascii_art.strip().split('\n')

    # Print each line with a delay
    os.system('cls' if os.name == 'nt' else 'clear')
    for line in ascii_lines:
        print(line)
        time.sleep(0.01)
    time.sleep(0.5)
    os.system('cls' if os.name == 'nt' else 'clear')