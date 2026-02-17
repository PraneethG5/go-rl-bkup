"""
Evaluation Module
Test trained agent against various baselines
"""

import numpy as np
from typing import Tuple
import torch
from tqdm import tqdm


class RandomAgent:
    """Agent that plays random legal moves"""
    
    def get_action(self, env):
        """Get random legal action"""
        legal_mask = env.get_legal_moves_mask()
        legal_actions = np.where(legal_mask > 0.5)[0]
        return np.random.choice(legal_actions)


class HeuristicAgent:
    """Simple heuristic-based agent"""
    
    def __init__(self, strategy='capture'):
        """
        Args:
            strategy: 'capture' or 'liberty'
        """
        self.strategy = strategy
    
    def get_action(self, env):
        """Get action based on heuristics"""
        legal_mask = env.get_legal_moves_mask()
        legal_actions = np.where(legal_mask > 0.5)[0]
        
        if self.strategy == 'capture':
            # Prioritize captures
            best_action = None
            max_captures = -1
            
            for action in legal_actions:
                if action == env.action_size - 1:  # Pass
                    continue
                
                # Try move and count captures
                test_env = env.clone()
                row = action // env.board_size
                col = action % env.board_size
                
                stones_before = np.sum(test_env.board == -test_env.current_player)
                test_env.step(action)
                stones_after = np.sum(test_env.board == test_env.current_player)
                
                captures = stones_before - stones_after
                
                if captures > max_captures:
                    max_captures = captures
                    best_action = action
            
            # If no captures, play randomly
            if best_action is None or max_captures == 0:
                return np.random.choice(legal_actions)
            
            return best_action
        
        elif self.strategy == 'liberty':
            # Maximize liberties of our stones
            best_action = None
            max_liberties = -1
            
            for action in legal_actions:
                if action == env.action_size - 1:  # Pass
                    continue
                
                # Try move and count liberties
                test_env = env.clone()
                row = action // env.board_size
                col = action % env.board_size
                
                test_env.board[row, col] = test_env.current_player
                liberties = test_env._get_liberties(row, col, test_env.board)
                
                if liberties > max_liberties:
                    max_liberties = liberties
                    best_action = action
            
            if best_action is None:
                return np.random.choice(legal_actions)
            
            return best_action
        
        else:
            # Default to random
            return np.random.choice(legal_actions)


class MCTSAgent:
    """Agent using MCTS with neural network"""
    
    def __init__(self, network, num_simulations=200, temperature=0.1):
        self.network = network
        self.num_simulations = num_simulations
        self.temperature = temperature
    
    def get_action(self, env):
        """Get action using MCTS"""
        from mcts import MCTS
        
        mcts = MCTS(
            self.network,
            num_simulations=self.num_simulations,
            temperature=self.temperature,
            c_puct=1.5
        )
        
        action, _ = mcts.get_action(env, add_noise=False)
        return action


def play_game(agent1, agent2, env, verbose=False):
    """
    Play a game between two agents
    
    Args:
        agent1: First agent (plays black)
        agent2: Second agent (plays white)
        env: Go environment
        verbose: Print game progress
    
    Returns:
        winner: 1 if agent1 wins, -1 if agent2 wins, 0 for draw
        game_length: number of moves
        illegal_moves: number of illegal moves
    """
    state = env.reset()
    done = False
    move_count = 0
    illegal_moves = 0
    
    agents = {1: agent1, -1: agent2}
    
    while not done:
        current_agent = agents[env.current_player]
        
        try:
            action = current_agent.get_action(env)
            state, reward, done, info = env.step(action)
            
            if 'illegal_move' in info:
                illegal_moves += 1
                if verbose:
                    print(f"Illegal move by {'Black' if env.current_player == -1 else 'White'}")
                break
            
            move_count += 1
            
            if verbose and move_count % 10 == 0:
                print(f"Move {move_count}")
                env.render()
        
        except Exception as e:
            if verbose:
                print(f"Error during move: {e}")
            illegal_moves += 1
            break
    
    if verbose:
        env.render()
        print(f"Game over! Length: {move_count}, Reward: {reward}")
    
    # Determine winner from reward
    # reward is from perspective of player who just moved (switched)
    if illegal_moves > 0:
        # Player who made illegal move loses
        winner = -env.current_player
    else:
        winner = 1 if reward > 0 else -1
    
    return winner, move_count, illegal_moves


def evaluate_agent(agent, baseline, num_games=20, verbose=False):
    """
    Evaluate agent against baseline
    
    Args:
        agent: Agent to evaluate
        baseline: Baseline agent
        num_games: Number of games to play
        verbose: Print progress
    
    Returns:
        Dictionary with evaluation metrics
    """
    from go_env import GoEnv
    
    wins = 0
    losses = 0
    game_lengths = []
    illegal_moves_total = 0
    
    iterator = tqdm(range(num_games), desc="Evaluating") if verbose else range(num_games)
    
    for game_idx in iterator:
        env = GoEnv(board_size=9)
        
        # Alternate who plays first
        if game_idx % 2 == 0:
            winner, length, illegal = play_game(agent, baseline, env)
            if winner == 1:
                wins += 1
            else:
                losses += 1
        else:
            winner, length, illegal = play_game(baseline, agent, env)
            if winner == -1:
                wins += 1
            else:
                losses += 1
        
        game_lengths.append(length)
        illegal_moves_total += illegal
    
    win_rate = wins / num_games * 100
    avg_length = np.mean(game_lengths)
    
    results = {
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_game_length': avg_length,
        'illegal_moves': illegal_moves_total,
        'num_games': num_games
    }
    
    return results


def evaluate_against_baselines(network, num_games=20, num_simulations=100):
    """
    Evaluate network against multiple baselines
    
    Args:
        network: Trained network
        num_games: Games per baseline
        num_simulations: MCTS simulations per move
    
    Returns:
        Dictionary of results for each baseline
    """
    agent = MCTSAgent(network, num_simulations=num_simulations, temperature=0.1)
    
    results = {}
    
    # Evaluate against random
    print("\n=== Evaluating vs Random Agent ===")
    random_agent = RandomAgent()
    results['random'] = evaluate_agent(agent, random_agent, num_games=num_games, verbose=True)
    print(f"Win rate vs Random: {results['random']['win_rate']:.1f}%")
    
    # Evaluate against capture heuristic
    print("\n=== Evaluating vs Capture Heuristic ===")
    capture_agent = HeuristicAgent(strategy='capture')
    results['capture'] = evaluate_agent(agent, capture_agent, num_games=num_games, verbose=True)
    print(f"Win rate vs Capture: {results['capture']['win_rate']:.1f}%")
    
    # Evaluate against liberty heuristic
    print("\n=== Evaluating vs Liberty Heuristic ===")
    liberty_agent = HeuristicAgent(strategy='liberty')
    results['liberty'] = evaluate_agent(agent, liberty_agent, num_games=num_games, verbose=True)
    print(f"Win rate vs Liberty: {results['liberty']['win_rate']:.1f}%")
    
    return results


def print_evaluation_report(results):
    """Print formatted evaluation report"""
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    
    for baseline, stats in results.items():
        print(f"\n{baseline.upper()}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Games: {stats['wins']}/{stats['num_games']}")
        print(f"  Avg Length: {stats['avg_game_length']:.1f} moves")
        print(f"  Illegal Moves: {stats['illegal_moves']}")


def test_evaluation():
    """Test evaluation system"""
    import sys
    sys.path.append('/home/claude/go_rl_agent')
    from go_env import GoEnv
    from network import create_network
    
    env = GoEnv(board_size=9)
    network = create_network()
    network.eval()
    
    # Test random vs random
    print("Testing Random vs Random...")
    random1 = RandomAgent()
    random2 = RandomAgent()
    
    winner, length, illegal = play_game(random1, random2, env, verbose=True)
    print(f"Winner: {winner}, Length: {length}, Illegal: {illegal}")


if __name__ == "__main__":
    test_evaluation()
