#!/usr/bin/python
import numpy as np
import psutil
import enum
import math
import random
import os
import argparse
from queue import PriorityQueue
import time
import copy
import glob

# 
#   Constants
#

START_NODE = -1
GOAL_NODE = -2
DEMO = 3

NORTH = (0, -1)
EAST = (1, 0)
SOUTH = (0, 1)
WEST = (-1, 0)


#
#   Move objects
#

class MoveType(enum.Enum):
    FORWARD = 0
    ROTATE_LEFT = 1
    ROTATE_RIGHT = 2
    BASH = 3
    DEMOLISH = 4


class Move:

    def __init__(self, move_type, facing, from_pos, to_pos, board_after_move, demos_used):
        self.move_type = move_type
        self.facing = facing
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.board = board_after_move
        self.demos_used = demos_used

    # Return the cost of a move
    def get_cost(self):

        switcher = {
            MoveType.FORWARD: get_val(self.board, self.to_pos),
            MoveType.ROTATE_LEFT: math.ceil(get_val(self.board, self.to_pos) / 2),
            MoveType.ROTATE_RIGHT: math.ceil(get_val(self.board, self.to_pos) / 2),
            MoveType.BASH: 3,
            MoveType.DEMOLISH: 4
        }

        return switcher.get(self.move_type, 0)

    # Return the direction if this move were to turn
    def turn(self, rotation):

        rotate_left = {
            NORTH: WEST,
            EAST: NORTH,
            SOUTH: EAST,
            WEST: SOUTH
        }

        if rotation == "left":
            return rotate_left.get(self.facing, self.facing)

        rotate_right = {
            NORTH: EAST,
            EAST: SOUTH,
            SOUTH: WEST,
            WEST: NORTH
        }

        if rotation == "right":
            return rotate_right.get(self.facing, self.facing)

    # Get string representation for move
    def get_move_string(self):

        switcher = {
            MoveType.FORWARD: "Forward",
            MoveType.ROTATE_LEFT: "Left",
            MoveType.ROTATE_RIGHT: "Right",
            MoveType.BASH: "Bash",
            MoveType.DEMOLISH: "Demolish"
        }

        return switcher.get(self.move_type, "Bad Move")

    def __eq__(self, o):
        return (self.move_type == o.move_type and
                self.facing == o.facing and
                self.from_pos == o.from_pos and
                self.to_pos == o.to_pos and
                self.demos_used == o.demos_used)

    def __lt__(self, o):
        return self.__hash__() < o.__hash__()

    def __hash__(self):
        return hash(hash(self.move_type) + hash(self.facing) + hash(self.from_pos) +
                    hash(self.to_pos) + hash(self.demos_used))


# Get input from user
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate different heuristics using a robot on varying terrain.')
    parser.add_argument('option', help='Can be a [directory], [file], or [integer]')
    parser.add_argument('heuristic', help='Select the heuristic to use [1-6]')
    parser.add_argument('-d', action='store_true', help='Use the demolition move')
    parser.add_argument('-w', type=int, default=10, help='Provide board [width] for random test')
    parser.add_argument('-l', type=int, default=10, help='Provide board [length] for random test')

    args = parser.parse_args()
    return int(args.heuristic), args.option, args.d, args.w, args.l


# Output function
def output(score, actionNum, nodesExpanded, actionsList):
    print("The score is: ", score)
    print("The number of actions are: ", actionNum)
    print("The number of nodes expanded is: ", nodesExpanded)
    print("The series of actions are: ")
    for action in actionsList:
        print(action.get_move_string())


#
#   A-* Functions
#

# Heuristic functions
def getHeuristic(heuristicNum, goalSpace, nextSpace):
    vertical = abs(goalSpace[1] - nextSpace[1])
    horizontal = abs(goalSpace[0] - nextSpace[0])
    switcher = {
        1: heuristic1(),
        2: heuristic2(vertical, horizontal),
        3: heuristic3(vertical, horizontal),
        4: heuristic4(vertical, horizontal),
        5: heuristic5(vertical, horizontal),
        6: heuristic6(vertical, horizontal)
    }

    return switcher.get(heuristicNum, "Invalid heuristic")


# Heuristic #1, always 0
def heuristic1():
    return 0


# Heuristic #2, Min(vertical, horizontal)
def heuristic2(vertical, horizontal):
    return min(vertical, horizontal)


# Heuristic #3, Max(vertical, horizontal)
def heuristic3(vertical, horizontal):
    return max(vertical, horizontal)


# Heuristic #4, vertical + horizontal
def heuristic4(vertical, horizontal):
    return vertical + horizontal


# Heuristic #5, new admissible heuristic
def heuristic5(vertical, horizontal):
    value = heuristic4(vertical, horizontal)
    if vertical > 0 and horizontal > 0:
        value += 1
    return value


# Heuristic #6, non-admissible heuristic
def heuristic6(vertical, horizontal):
    return heuristic5(vertical, horizontal) * 3


# Search for the best path on a board given a heuristic
def astar(board, heuristic, use_demo):
    process = psutil.Process(os.getpid())

    # Try to find start and goal nodes
    try:
        start_pos = get_pos(board, START_NODE)
        goal_pos = get_pos(board, GOAL_NODE)
    except:
        print("Unable to find required node on board. Aborting...")
        return 0, 0, 0, [], process.memory_info(), 0

    start_move = Move(None, NORTH, start_pos, start_pos, board, 0)
    goal_move = 0
    q = PriorityQueue()
    move_order = {}
    cost_so_far = {}
    num_nodes_expanded = 0

    # Load initial values
    q.put((0, start_move))
    move_order[start_move] = None
    cost_so_far[start_move] = 0

    while not q.empty():

        _, current_move = q.get()
        num_nodes_expanded += 1

        if current_move.to_pos == goal_pos:
            goal_move = current_move
            break

        for next_move in get_possible_moves(current_move, use_demo):
            cost = next_move.get_cost() + cost_so_far[current_move]

            # Update total cost of the move
            if next_move not in cost_so_far or cost < cost_so_far[next_move]:
                cost_so_far[next_move] = cost

                # Update priority queue and move order
                priority = cost + getHeuristic(heuristic, goal_pos, next_move.to_pos)
                q.put((priority, next_move))
                move_order[next_move] = current_move

    if not goal_move:
        print("Unable to find valid path on board. Aborting...")
        return 0, 0, 0, [], process.memory_info(), 0

    # Walk back through visited moves
    backtrack_move = goal_move
    demos = goal_move.demos_used
    moves = []

    while backtrack_move != start_move:
        moves.append(backtrack_move)
        backtrack_move = move_order[backtrack_move]

    score = 100 - cost_so_far[goal_move]
    tot_moves = len(moves)
    moves.reverse()

    memory = process.memory_info()

    return score, tot_moves, num_nodes_expanded, moves, memory, demos


# Returns all possible next moves on a board from a starting move
def get_possible_moves(move, use_demo):
    valid_moves = []
    board = move.board

    # Check forward move
    forward_move = (move.to_pos[0] + move.facing[0], move.to_pos[1] + move.facing[1])

    if check_bounds(forward_move, board):
        valid_moves.append(Move(MoveType.FORWARD, move.facing, move.to_pos, forward_move, board, move.demos_used))

    # Check moves invalid after bash
    if move.move_type != MoveType.BASH:

        # Check rotations
        valid_moves.append(
            Move(MoveType.ROTATE_LEFT, move.turn("left"), move.to_pos, move.to_pos, board, move.demos_used))
        valid_moves.append(
            Move(MoveType.ROTATE_RIGHT, move.turn("right"), move.to_pos, move.to_pos, board, move.demos_used))

        # Check bash move
        bash_move = (move.from_pos[0] + move.facing[0] * 2, move.from_pos[1] + move.facing[1] * 2)

        if check_bounds(bash_move, board):
            valid_moves.append(Move(MoveType.BASH, move.facing, move.to_pos, forward_move, board, move.demos_used))

        # Check demo move
        if use_demo:
            demo_board = demolish(board, move.to_pos)
            valid_moves.append(
                Move(MoveType.DEMOLISH, move.facing, move.to_pos, move.to_pos, demo_board, move.demos_used + 1))

    return valid_moves


#
#   Board functions
#

# Returns a tuple representing the position of a value in the board
def get_pos(board, val):
    return np.where(board == val)[1][0], np.where(board == val)[0][0]


# Returns the value at a position in the board
def get_val(board, pos):
    try:
        value = int(board[pos[1], pos[0]])
        if value == START_NODE or value == GOAL_NODE:
            return 1
        else:
            return value
    except:
        return None


# Check if a pos is valid on a board
def check_bounds(pos, board):
    max_x = np.shape(board)[1]
    max_y = np.shape(board)[0]

    if pos[0] >= max_x or pos[0] < 0:
        return False

    if pos[1] >= max_y or pos[1] < 0:
        return False

    return True


# Does a demolish move on a board, returning a new board
def demolish(board, pos):
    demo_board = copy.deepcopy(board)
    max_x = np.shape(board)[1]
    max_y = np.shape(board)[0]
    center_x = pos[1]
    center_y = pos[0]

    x = -1
    y = -1

    # Walk around the position
    while x <= 1:
        while y <= 1:

            # Ignore center node and nodes outside the board
            if ((x != 0 or y != 0) and
                    (center_x + x >= 0 and center_x + x < max_x) and
                    (center_y + y >= 0 and center_y + y < max_y)):

                old_val = board[center_y + y][center_x + x]

                # Ignore start and end nodes
                if old_val > 0:
                    demo_board[center_y + y][center_x + x] = DEMO

            y += 1

        y = -1
        x += 1

    return demo_board


# Render a move on the board in the terminal
def print_current_state(board, move):
    # Render the board:
    print("**************************\n\n")

    print("0------------0")
    for y, col in enumerate(board):
        print("|", end="")
        for x, row in enumerate(col):
            if move.to_pos[0] == x and move.to_pos[1] == y:
                print(" " + "$" + " ", end="")
            elif move.from_pos[0] == x and move.from_pos[1] == y:
                print(" " + "*" + " ", end="")
            elif row == -2:
                print(" " + "G" + " ", end="")
            elif row == -1:
                print(" " + "S" + " ", end="")
            else:
                print(" " + str(row) + " ", end="")
        print("|", end="\n")
    print("0------------0")

    print("Move type: ", move.move_type)
    print("Facing : ", move.facing)
    print("Move cost: ", move.get_cost(board))

    print("\n\n__________________________")


# Create a numpy represented board with random values between 1 and 9
def generate_board(size_x, size_y):
    rand = random.Random()
    board = np.zeros((size_y, size_x), dtype=int)

    for x in range(size_x):
        for y in range(size_y):
            board[y][x] = rand.randint(1, 9)

    goal_pos = (rand.randint(0, size_y - 1), rand.randint(0, size_x - 1))
    start_pos = (rand.randint(0, size_y - 1), rand.randint(0, size_x - 1))

    while start_pos == goal_pos:
        start_pos = (rand.randint(0, size_y - 1), rand.randint(0, size_x - 1))

    board[goal_pos] = -2
    board[start_pos] = -1

    return board


# Process a board text file into a np array
def process_board(filename):
    with open(filename) as f:
        for x, line in enumerate(f.readlines()):
            row = line.split('\t')
            row[-1] = row[-1].rstrip('\n')

            if x == 0:
                # Create properly size array
                arr = ([np.array(row)])
                continue

            if len(row) > 0:
                # Add additional rows to array
                row = np.array(row)
                arr = np.append(arr, [row], axis=0)

    return np.vectorize(map_to_ints)(arr)


# Convert board elements to ints
def map_to_ints(x):
    if x == 'G':
        return GOAL_NODE
    elif x == 'S':
        return START_NODE
    else:
        try:
            return int(x)
        except:
            print("Board is malformed, can only contain ints, 'G' and 'S'")
            return -3  # Error tile


# Writes a board to a file (will overwrite file if it exists)
def save_board(board, fn):
    print(f'Writing board to file {fn}')

    max_x = np.shape(board)[1]
    max_y = np.shape(board)[0]

    with open(fn, "w") as f:

        for y in range(max_y):

            for x in range(max_x):

                val = board[y][x]

                if val == GOAL_NODE:
                    val = "G"
                elif val == START_NODE:
                    val = "S"

                f.write(f'{val}')

                if x != max_x - 1:
                    f.write('\t')
                elif y != max_y - 1:
                    f.write('\n')


#
#   Main script
#

if __name__ == '__main__':

    # Get user input
    given_heuristic, opt, use_demo, w, l = parse_args()

    process = psutil.Process(os.getpid())

    # Directory test
    # Runs astar on all files in a directory. Records average time, memory, and nodes reached.
    if os.path.isdir(opt):
        dr = opt
        timer_total = 0
        mem_total = 0
        total_nodes = 0
        board_num = 0
        total_moves = 0
        board_files = [f for f in os.listdir(dr) if os.path.isfile(os.path.join(dr, f))]

        for board_file in board_files:
            print(f'\n/************** Testing board {board_file}: **************/')
            game_board = process_board(dr + board_file)

            print(f'Board {board_num}:\n{game_board}')
            print(f'Heuristic: {given_heuristic}')

            start = time.time()
            initial_mem = process.memory_info()
            final_score, num_moves, nodes_expanded, all_moves, mem, demos = astar(game_board, given_heuristic, use_demo)
            output(final_score, num_moves, nodes_expanded, all_moves)
            total_nodes += nodes_expanded
            total_moves += num_moves

            mem_used = (mem.rss - initial_mem.rss) / 1000000000
            print("Memory used:", mem_used, "GB")
            mem_total += mem_used
            end = time.time()
            elapsed_time = end - start
            print("Elapsed time is: ", elapsed_time)
            timer_total += elapsed_time

            board_num += 1

        print(f'\n/************** Final Results for Directory {dr} Boards: **************/')
        print("Average memory:", mem_total / len(board_file), "GB")
        print("Average time: ", timer_total / len(board_files))
        print("Average nodes expanded is: ", total_nodes / len(board_files))
        print("Average moves done: ", total_moves / len(board_files))

    # Generate test
    # Runs x tests on boards generated of n x n size
    # Boards are written to the out/ directory
    elif opt.isnumeric():

        # Clear files
        files = glob.glob('./out/*')
        for f in files:
            os.remove(f)

        timer_total = 0
        total_nodes = 0
        mem_total = 0
        total_moves = 0
        number_of_cycles = int(opt)

        for i in range(number_of_cycles):
            game_board = generate_board(int(w), int(l))

            print(f'\n/************** Board {i} **************/\n{game_board}')
            print(f'Heuristic: {given_heuristic}')

            start = time.time()
            initial_mem = process.memory_info()
            final_score, num_moves, nodes_expanded, all_moves, mem, demos = astar(game_board, given_heuristic, use_demo)
            output(final_score, num_moves, nodes_expanded, all_moves)
            total_nodes += nodes_expanded
            total_moves += num_moves

            mem_used = (mem.rss - initial_mem.rss) / 1000000000
            print("Memory used:", mem_used, "GB")
            mem_total += mem_used
            end = time.time()
            elapsed_time = end - start
            print("Elapsed time is: ", elapsed_time)
            timer_total += elapsed_time

            save_board(game_board, f'out/test{i}.txt')

        print(f'\n/************** Final Results for {int(opt)} {l}x{w} Boards: **************/')
        print("Average memory:", mem_total / number_of_cycles, "GB")
        print("Average time: ", timer_total / number_of_cycles)
        print("Average nodes expanded is: ", total_nodes / number_of_cycles)
        print("Average moves done: ", total_moves / int(opt))

    # Single test 
    else:
        start = time.time()
        initial_mem = process.memory_info()

        game_board = process_board(opt)

        print(f'\n/************** Board: {opt} **************/\n{game_board}')
        print(f'Heuristic: {given_heuristic}')

        final_score, total_num_moves, nodes_expanded, all_moves, mem, demos = \
            astar(game_board, given_heuristic, use_demo)
        output(final_score, total_num_moves, nodes_expanded, all_moves)

        mem_used = (mem.rss - initial_mem.rss) / 1000000000
        print("Memory used:", mem_used, "GB")
        end = time.time()
        elapsed_time = end - start
        print("Elapsed time is:", elapsed_time)
