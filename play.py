"""
Play Go against the RL Agent (or watch AI vs AI)
Interactive CLI interface for 9x9 Go
"""

import sys
import os
import argparse
import numpy as np
import torch

# Ensure utf-8 output on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

from go_env import GoEnv
from network import create_network
from mcts import MCTS
from evaluate import RandomAgent, HeuristicAgent


class HumanAgent:
    """Human player interface with coordinate translation"""

    def __init__(self, color_name="Black"):
        self.color_name = color_name

    def get_action(self, env: GoEnv) -> int:
        legal_mask = env.get_legal_moves_mask()
        legal_actions = np.where(legal_mask > 0.5)[0]

        while True:
            try:
                user_input = input(f"[{self.color_name}'s Turn] Enter move (e.g. E5, 4,4, 'pass', or 'resign'): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nQuitting...")
                return env.action_size - 1

            if not user_input:
                continue

            if user_input.lower() in ["resign", "quit", "q"]:
                print(f"{self.color_name} resigns!")
                # To resign, we return a special indicator or pass
                return -999

            action = env.coord_to_action(user_input)
            if action is None:
                # Try raw action index
                try:
                    action = int(user_input)
                except ValueError:
                    print(f"Invalid format '{user_input}'. Use standard Go coords like 'E5', row/col like '4,4', or 'pass'.")
                    continue

            if action in legal_actions:
                return action
            else:
                row = action // env.board_size
                col = action % env.board_size
                if action == env.action_size - 1:
                    print("Pass is legal.")
                    return action
                elif env.board[row, col] != 0:
                    print(f"Position {env.action_to_coord(action)} is already occupied!")
                elif env.ko_point == (row, col):
                    print("Move violates Ko rule (immediate recapture not allowed)!")
                else:
                    print(f"Move {env.action_to_coord(action)} is illegal (suicide move)!")


def render_board(env: GoEnv, last_action: int = None, scores=None):
    """Render board with coordinates and symbols"""
    cols = "ABCDEFGHJKLMNOPQRST"[:env.board_size]
    symbols = {0: '·', 1: '●', -1: '○'}  # Black = ●, White = ○

    # Star points (hoshi) for 9x9 board: (2,2), (2,6), (6,2), (6,6), (4,4)
    hoshi_points = {(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)}

    print("\n    " + " ".join(cols))
    print("  ┌" + "──" * env.board_size + "┐")
    for r in range(env.board_size):
        row_num = env.board_size - r
        row_str = f"{row_num:2d}│ "
        for c in range(env.board_size):
            action_idx = r * env.board_size + c
            piece = env.board[r, c]
            if piece != 0:
                char = symbols[piece]
            elif (r, c) in hoshi_points:
                char = '+'
            else:
                char = symbols[0]

            # Mark last move
            if last_action is not None and action_idx == last_action:
                row_str += f"[{char}]" if len(row_str) >= 2 and row_str[-1] == ' ' else f"{char}"
            else:
                row_str += f"{char} "
        row_str = row_str.rstrip() + f" │{row_num}"
        print(row_str)
    print("  └" + "──" * env.board_size + "┘")
    print("    " + " ".join(cols))

    # Stats
    black_stones = np.sum(env.board == 1)
    white_stones = np.sum(env.board == -1)
    print(f"\nMove {env.move_count} | Turn: {'● Black' if env.current_player == 1 else '○ White'} | Passes: {env.passes}")
    print(f"Stones on Board: Black (●): {black_stones} | White (○): {white_stones}")
    if scores is not None:
        b_score, w_score = scores
        print(f"Estimated Territory & Area Score: Black {b_score:.1f} vs White {w_score:.1f} (incl. 6.5 komi)")


def play_game(black_agent, white_agent, board_size=9, verbose=True):
    """Run interactive game loop between two agents"""
    env = GoEnv(board_size=board_size)
    env.reset()
    last_action = None

    agents = {1: ("Black", black_agent), -1: ("White", white_agent)}

    if verbose:
        print("\n" + "=" * 55)
        print("          WELCOME TO GO RL AGENT")
        print("=" * 55)
        print("Symbols: Black (●), White (○), Star Point (+), Empty (·)")
        print("Standard coordinates: Columns A-J (skipping I), Rows 9 down to 1")
        render_board(env)

    done = False
    resigned_by = None

    while not done:
        name, agent = agents[env.current_player]
        action = agent.get_action(env)

        if action == -999:
            # Resignation
            resigned_by = env.current_player
            break

        coord_str = env.action_to_coord(action)
        if verbose:
            print(f"==> {name} plays: {coord_str}")

        last_action = action if action != env.action_size - 1 else None
        state, reward, done, info = env.step(action)

        if verbose:
            scores = env.get_scores()
            render_board(env, last_action=last_action, scores=scores)

        if 'illegal_move' in info:
            print(f"Illegal move made by {name}! {name} forfeits the match.")
            resigned_by = env.current_player
            break

    # Final result
    print("\n" + "=" * 55)
    print("                   GAME OVER")
    print("=" * 55)

    if resigned_by is not None:
        winner = -resigned_by
        winner_name = "Black (●)" if winner == 1 else "White (○)"
        loser_name = "White (○)" if winner == 1 else "Black (●)"
        print(f"\n{loser_name} resigned. Winner: {winner_name} by resignation!")
    else:
        b_score, w_score = env.get_scores()
        winner = env.get_winner()
        winner_name = "Black (●)" if winner == 1 else "White (○)"
        margin = abs(b_score - w_score)
        print(f"\nFinal Score:")
        print(f"  ● Black: {b_score:.1f}")
        print(f"  ○ White: {w_score:.1f} (including 6.5 komi)")
        print(f"\nWinner: {winner_name} wins by {margin:.1f} points!")
    print("=" * 55 + "\n")
    return winner


def main():
    parser = argparse.ArgumentParser(description="Play Go against RL Agent")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--color', type=str, choices=['black', 'white'], default='black',
                        help='Color to play as (default: black)')
    parser.add_argument('--mode', type=str, choices=['human_vs_ai', 'ai_vs_ai', 'human_vs_human'],
                        default='human_vs_ai', help='Game mode')
    parser.add_argument('--simulations', type=int, default=100,
                        help='MCTS simulations per move for AI (default: 100)')
    parser.add_argument('--opponent', type=str, choices=['network', 'heuristic', 'random'],
                        default='network', help='Fallback AI if no checkpoint given')
    args = parser.parse_args()

    # Setup AI agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def create_ai_agent(name="AI"):
        if args.checkpoint and os.path.exists(args.checkpoint):
            print(f"Loading checkpoint from: {args.checkpoint} (using {device})...")
            network = create_network(board_size=9)
            try:
                checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(args.checkpoint, map_location=device)
            network.load_state_dict(checkpoint['model_state_dict'])
            network.to(device)
            network.eval()

            class AIAgentWrapper:
                def __init__(self, net, sims):
                    self.net = net
                    self.sims = sims
                def get_action(self, env):
                    mcts = MCTS(self.net, num_simulations=self.sims, temperature=0.05, c_puct=1.5)
                    action, probs = mcts.get_action(env, add_noise=False)
                    return action
            return AIAgentWrapper(network, args.simulations)
        elif args.opponent == 'heuristic':
            print("Using Heuristic Agent (capture & liberty evaluation)...")
            return HeuristicAgent(strategy='capture')
        elif args.opponent == 'random':
            print("Using Random Move Baseline...")
            return RandomAgent()
        else:
            print("Creating fresh Go neural network with MCTS...")
            network = create_network(board_size=9).to(device)
            network.eval()
            class FreshAIAgent:
                def __init__(self, net, sims):
                    self.net = net
                    self.sims = sims
                def get_action(self, env):
                    mcts = MCTS(self.net, num_simulations=self.sims, temperature=0.1, c_puct=1.5)
                    action, _ = mcts.get_action(env, add_noise=False)
                    return action
            return FreshAIAgent(network, args.simulations)

    if args.mode == 'human_vs_ai':
        if args.color == 'black':
            black = HumanAgent("Black")
            white = create_ai_agent("AI White")
        else:
            black = create_ai_agent("AI Black")
            white = HumanAgent("White")
    elif args.mode == 'human_vs_human':
        black = HumanAgent("Black")
        white = HumanAgent("White")
    else:  # ai_vs_ai
        black = create_ai_agent("AI Black")
        white = create_ai_agent("AI White")

    play_game(black, white, board_size=9)


if __name__ == "__main__":
    main()
