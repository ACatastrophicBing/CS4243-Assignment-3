import astar

if __name__ == '__main__':
    for i in range(10):
        astar.save_board(astar.generate_board(125, 125), f'sample_boards/test_board_190_{i}.txt')
