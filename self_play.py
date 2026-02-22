"""
Self-Play Game Generator
Generates training data through self-play with MCTS
"""

import numpy as np
from typing import List, Tuple
import torch
from tqdm import tqdm


class SelfPlayGame:
    """Container for a single self-play game"""
    
    def __init__(self):
        self.states = []  # List of (5, 9, 9) states
        self.mcts_policies = []  # List of (82,) MCTS visit distributions
        self.current_player = []  # List of player indicators
        self.outcome = None  # Final outcome: 1 (win), -1 (loss)
    
    def add_step(self, state, mcts_policy, current_player):
        """Add a step to the game history"""
        self.states.append(state.copy())
        self.mcts_policies.append(mcts_policy.copy())
        self.current_player.append(current_player)
    
    def set_outcome(self, outcome: float):
        """Set final outcome from perspective of final player"""
        self.outcome = outcome
    
    def get_training_samples(self):
        """
        Get training samples with proper value assignment
        Returns list of (state, mcts_policy, value) tuples
        """
        samples = []
        
        for i, (state, policy, player) in enumerate(zip(self.states, self.mcts_policies, self.current_player)):
            # Assign value from perspective of player who made the move
            # outcome is from perspective of player at end of game
            # Need to flip based on whose turn it was
            value = self.outcome if player == self.current_player[-1] else -self.outcome
            
            samples.append((state, policy, value))
        
        return samples
    
    def augment_data(self):
        """
        Augment data with rotations and reflections
        8-fold symmetry: 4 rotations × 2 reflections
        """
        augmented_samples = []
        samples = self.get_training_samples()
        
        for state, policy, value in samples:
            # Original
            augmented_samples.append((state, policy, value))
            
            # Apply all 8 symmetries
            for k in range(1, 4):  # 3 more rotations (90, 180, 270 degrees)
                rot_state = np.rot90(state, k=k, axes=(1, 2))
                rot_policy = self._rotate_policy(policy, k)
                augmented_samples.append((rot_state, rot_policy, value))
            
            # Flip horizontally + 4 rotations
            flip_state = np.flip(state, axis=2)
            flip_policy = self._flip_policy(policy)
            augmented_samples.append((flip_state, flip_policy, value))
            
            for k in range(1, 4):
                rot_flip_state = np.rot90(flip_state, k=k, axes=(1, 2))
                rot_flip_policy = self._rotate_policy(flip_policy, k)
                augmented_samples.append((rot_flip_state, rot_flip_policy, value))
        
        return augmented_samples
    
    def _rotate_policy(self, policy, k):
        """Rotate policy vector (board positions only, not pass)"""
        board_policy = policy[:-1].reshape(9, 9)
        rotated = np.rot90(board_policy, k=k)
        return np.concatenate([rotated.flatten(), [policy[-1]]])  # Keep pass action
    
    def _flip_policy(self, policy):
        """Flip policy horizontally"""
        board_policy = policy[:-1].reshape(9, 9)
        flipped = np.flip(board_policy, axis=1)
        return np.concatenate([flipped.flatten(), [policy[-1]]])


class SelfPlayGenerator:
    """Generate self-play games"""
    
    def __init__(self, network, mcts_config):
        """
        Args:
            network: Neural network
            mcts_config: Dict with MCTS parameters
        """
        self.network = network
        self.mcts_config = mcts_config
    
    def generate_games(self, num_games: int, temperature_schedule=None, verbose=True):
        """
        Generate multiple self-play games
        
        Args:
            num_games: Number of games to generate
            temperature_schedule: Function that returns temperature given move number
            verbose: Show progress bar
        
        Returns:
            List of SelfPlayGame objects
        """
        from go_env import GoEnv
        from mcts import MCTS
        
        games = []
        
        iterator = tqdm(range(num_games), desc="Self-play") if verbose else range(num_games)
        
        for game_idx in iterator:
            game = self.play_single_game(temperature_schedule)
            games.append(game)
        
        return games
    
    def play_single_game(self, temperature_schedule=None):
        """
        Play a single self-play game
        
        Args:
            temperature_schedule: Function that returns temperature given move number
        
        Returns:
            SelfPlayGame object
        """
        from go_env import GoEnv
        from mcts import MCTS
        
        env = GoEnv(board_size=9)
        state = env.reset()
        
        game = SelfPlayGame()
        done = False
        move_count = 0
        
        while not done:
            # Determine temperature for this move
            if temperature_schedule is not None:
                temperature = temperature_schedule(move_count)
            else:
                # Default: high temp for first 15 moves, then low
                temperature = 1.0 if move_count < 15 else 0.1
            
            # Create MCTS with current temperature
            mcts = MCTS(
                self.network,
                num_simulations=self.mcts_config.get('num_simulations', 200),
                c_puct=self.mcts_config.get('c_puct', 1.5),
                temperature=temperature,
                dirichlet_alpha=self.mcts_config.get('dirichlet_alpha', 0.3),
                dirichlet_epsilon=self.mcts_config.get('dirichlet_epsilon', 0.25)
            )
            
            # Get action from MCTS
            action, mcts_policy = mcts.get_action(env, add_noise=True)
            
            # Store state and policy
            game.add_step(state, mcts_policy, env.current_player)
            
            # Execute action
            state, reward, done, info = env.step(action)
            move_count += 1
            
            # Check for illegal move (should not happen with proper masking)
            if done and 'illegal_move' in info:
                print(f"Warning: Illegal move in self-play at move {move_count}")
                reward = -1.0
        
        # Set outcome (reward is from perspective of player who just moved)
        game.set_outcome(reward)
        
        return game
    
    def generate_training_data(self, num_games: int, augment=True, temperature_schedule=None):
        """
        Generate training data from self-play
        
        Returns:
            states: numpy array (N, 5, 9, 9)
            policies: numpy array (N, 82)
            values: numpy array (N,)
        """
        games = self.generate_games(num_games, temperature_schedule=temperature_schedule)
        
        all_states = []
        all_policies = []
        all_values = []
        
        for game in games:
            if augment:
                samples = game.augment_data()
            else:
                samples = game.get_training_samples()
            
            for state, policy, value in samples:
                all_states.append(state)
                all_policies.append(policy)
                all_values.append(value)
        
        return (
            np.array(all_states, dtype=np.float32),
            np.array(all_policies, dtype=np.float32),
            np.array(all_values, dtype=np.float32)
        )


class ReplayBuffer:
    """Replay buffer for storing training data"""
    
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.states = []
        self.policies = []
        self.values = []
    
    def add_data(self, states, policies, values):
        """Add data to buffer"""
        for s, p, v in zip(states, policies, values):
            self.states.append(s)
            self.policies.append(p)
            self.values.append(v)
        
        # Keep only most recent data
        if len(self.states) > self.max_size:
            self.states = self.states[-self.max_size:]
            self.policies = self.policies[-self.max_size:]
            self.values = self.values[-self.max_size:]
    
    def sample(self, batch_size):
        """Sample random batch"""
        indices = np.random.choice(len(self.states), size=min(batch_size, len(self.states)), replace=False)
        
        batch_states = np.array([self.states[i] for i in indices], dtype=np.float32)
        batch_policies = np.array([self.policies[i] for i in indices], dtype=np.float32)
        batch_values = np.array([self.values[i] for i in indices], dtype=np.float32)
        
        return batch_states, batch_policies, batch_values
    
    def __len__(self):
        return len(self.states)
    
    def clear(self):
        """Clear buffer"""
        self.states = []
        self.policies = []
        self.values = []


def test_self_play():
    """Test self-play generation"""
    import sys
    sys.path.append('/home/claude/go_rl_agent')
    from network import create_network
    
    network = create_network()
    network.eval()
    
    mcts_config = {
        'num_simulations': 50,  # Reduced for testing
        'c_puct': 1.5,
        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.25
    }
    
    generator = SelfPlayGenerator(network, mcts_config)
    
    print("Generating self-play game...")
    game = generator.play_single_game()
    
    print(f"Game length: {len(game.states)} moves")
    print(f"Outcome: {game.outcome}")
    
    samples = game.get_training_samples()
    print(f"Training samples: {len(samples)}")
    
    augmented = game.augment_data()
    print(f"Augmented samples: {len(augmented)} (8x augmentation)")


if __name__ == "__main__":
    test_self_play()
