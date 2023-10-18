import neat
import chess
import os

# Convert a board state to a tensor representation
def board_to_tensor(board):
    # This is a simple placeholder conversion to tensor form; more advanced conversions are available
    return [float(piece) for piece in board.board_fen().replace(' ', '').replace('/', '')]

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Simulate a game of chess using the neural network
        board = chess.Board()
        while not board.is_game_over():
            board_tensor = board_to_tensor(board)
            output = net.activate(board_tensor)
            move = chess.Move.from_uci(chess.SQUARE_NAMES[output.index(max(output))])
            if move in board.legal_moves:
                board.push(move)
        # Set the genome's fitness based on the game result
        result = board.result()
        if result == "1-0":
            genome.fitness = 1.0
        elif result == "0-1":
            genome.fitness = 0.0
        else:
            genome.fitness = 0.5

def run_neat(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create a population
    pop = neat.Population(config)

    # Add reporters to display progress
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(5))

    # Run the NEAT algorithm for up to 30 generations
    winner = pop.run(eval_genomes, 30)

    # Save the best network
    with open('best_chess_net.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

if __name__ == '__main__':
    run_neat('neat_config.txt')
