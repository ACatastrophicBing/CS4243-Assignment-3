import astar

for i in range(10):
    astar.save_board(astar.generate_board(125, 125), f'sample_boards/test_board_125_{i}.txt')
