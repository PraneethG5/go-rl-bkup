"""
Main Training Script for Go RL Agent
Orchestrates self-play, training, and evaluation
"""

import torch
import numpy as np
import os
import json
from datetime import datetime
import argparse

from network import create_network, count_parameters
from self_play import SelfPlayGenerator, ReplayBuffer
from ppo_trainer import create_trainer
from evaluate import evaluate_against_baselines, print_evaluation_report, MCTSAgent
from go_env import GoEnv


class TrainingConfig:
    """Configuration for training"""
    
    def __init__(self):
        # Network
        self.board_size = 9
        self.num_channels = 128
        self.num_res_blocks = 8
        
        # MCTS
        self.num_simulations = 200
        self.c_puct = 1.5
        self.dirichlet_alpha = 0.3
        self.dirichlet_epsilon = 0.25
        
        # Self-play
        self.games_per_iteration = 50
        self.augment_data = True
        
        # Training
        self.learning_rate = 0.001
        self.batch_size = 64
        self.training_epochs = 3
        self.value_loss_coef = 1.0
        self.entropy_coef = 0.01
        
        # Replay buffer
        self.buffer_size = 100000
        
        # Training loop
        self.num_iterations = 40
        self.eval_interval = 5  # Evaluate every N iterations
        self.eval_games = 20
        self.eval_simulations = 100
        
        # Checkpointing
        self.save_interval = 5
        self.checkpoint_dir = 'checkpoints'  # Relative path works on all OS
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, path):
        """Save config to JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def temperature_schedule(move_num):
    """Temperature schedule for move selection"""
    if move_num < 15:
        return 1.0  # High exploration early
    elif move_num < 30:
        return 0.5  # Medium exploration mid-game
    else:
        return 0.1  # Low exploration late-game


class Trainer:
    """Main trainer class"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Initialize network
        print("Initializing network...")
        self.network = create_network(
            board_size=config.board_size,
            num_channels=config.num_channels,
            num_res_blocks=config.num_res_blocks
        )
        self.network.to(config.device)
        print(f"Network has {count_parameters(self.network):,} parameters")
        print(f"Using device: {config.device}")
        
        # Initialize components
        self.mcts_config = {
            'num_simulations': config.num_simulations,
            'c_puct': config.c_puct,
            'dirichlet_alpha': config.dirichlet_alpha,
            'dirichlet_epsilon': config.dirichlet_epsilon
        }
        
        self.self_play_generator = SelfPlayGenerator(self.network, self.mcts_config)
        
        self.trainer = create_trainer(
            self.network,
            {
                'learning_rate': config.learning_rate,
                'value_loss_coef': config.value_loss_coef,
                'entropy_coef': config.entropy_coef
            },
            device=config.device
        )
        
        self.replay_buffer = ReplayBuffer(max_size=config.buffer_size)
        
        # Training history
        self.history = {
            'iteration': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'eval_results': []
        }
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        for iteration in range(1, self.config.num_iterations + 1):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}/{self.config.num_iterations}")
            print(f"{'='*60}")
            
            # 1. Generate self-play data
            print(f"\n[1/3] Generating {self.config.games_per_iteration} self-play games...")
            states, policies, values = self.self_play_generator.generate_training_data(
                num_games=self.config.games_per_iteration,
                augment=self.config.augment_data,
                temperature_schedule=temperature_schedule
            )
            
            print(f"Generated {len(states):,} training samples")
            
            # 2. Add to replay buffer
            self.replay_buffer.add_data(states, policies, values)
            print(f"Replay buffer size: {len(self.replay_buffer):,}")
            
            # 3. Train network
            print(f"\n[2/3] Training network for {self.config.training_epochs} epochs...")
            
            # Sample from replay buffer
            if len(self.replay_buffer) > self.config.batch_size:
                train_states, train_policies, train_values = self.replay_buffer.sample(
                    min(len(self.replay_buffer), 10000)  # Use up to 10k samples per iteration
                )
            else:
                train_states, train_policies, train_values = states, policies, values
            
            stats = self.trainer.train_epoch(
                train_states,
                train_policies,
                train_values,
                batch_size=self.config.batch_size,
                epochs=self.config.training_epochs
            )
            
            print(f"  Policy Loss: {stats['policy_loss']:.4f}")
            print(f"  Value Loss: {stats['value_loss']:.4f}")
            print(f"  Entropy: {stats['entropy']:.4f}")
            print(f"  Total Loss: {stats['total_loss']:.4f}")
            
            # Record history
            self.history['iteration'].append(iteration)
            self.history['policy_loss'].append(stats['policy_loss'])
            self.history['value_loss'].append(stats['value_loss'])
            self.history['entropy'].append(stats['entropy'])
            self.history['total_loss'].append(stats['total_loss'])
            
            # 4. Evaluate
            if iteration % self.config.eval_interval == 0:
                print(f"\n[3/3] Evaluating agent...")
                eval_results = evaluate_against_baselines(
                    self.network,
                    num_games=self.config.eval_games,
                    num_simulations=self.config.eval_simulations
                )
                self.history['eval_results'].append({
                    'iteration': iteration,
                    'results': eval_results
                })
                print_evaluation_report(eval_results)
            
            # 5. Save checkpoint
            if iteration % self.config.save_interval == 0:
                self.save_checkpoint(iteration)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        
        # Final evaluation
        if getattr(self.config, 'skip_eval', False) or self.config.eval_games <= 0:
            print("\nSkipping final evaluation (--skip-eval specified)")
        else:
            print("\nRunning final evaluation...")
            final_eval = evaluate_against_baselines(
                self.network,
                num_games=self.config.eval_games,
                num_simulations=self.config.eval_simulations
            )
            print_evaluation_report(final_eval)
        
        # Save final model
        self.save_checkpoint('final')
        
        return self.history
    
    def save_checkpoint(self, iteration):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'model_iter_{iteration}.pt'
        )
        
        checkpoint_data = {
            'iteration': iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history
        }

        # Save model_iter_{iteration}.pt
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Also save as model_latest.pt for convenient reference
        latest_path = os.path.join(self.config.checkpoint_dir, 'model_latest.pt')
        torch.save(checkpoint_data, latest_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['iteration']


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train Go RL Agent')
    parser.add_argument('--iterations', type=int, default=40, help='Number of training iterations')
    parser.add_argument('--games', type=int, default=50, help='Self-play games per iteration')
    parser.add_argument('--simulations', type=int, default=200, help='MCTS simulations per move')
    parser.add_argument('--eval-games', type=int, default=None, help='Games per baseline in evaluation')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation against baselines')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--fast', action='store_true', help='Fast mode (fewer games/simulations)')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    
    # Create checkpoint directory early
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    if args.fast:
        print("FAST MODE: Reducing games and simulations for quick testing")
        config.games_per_iteration = 20
        config.num_simulations = 100
        config.eval_simulations = 50
        config.eval_games = 10
        config.num_iterations = 20
    else:
        config.num_iterations = args.iterations
        config.games_per_iteration = args.games
        config.num_simulations = args.simulations

    if args.eval_games is not None:
        config.eval_games = args.eval_games
    config.skip_eval = args.skip_eval
    
    # Save config
    config.save(os.path.join(config.checkpoint_dir, 'config.json'))
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        start_iter = trainer.load_checkpoint(args.resume)
        print(f"Resuming from iteration {start_iter}")
    
    # Train
    history = trainer.train()
    
    # Save training history
    history_path = os.path.join(config.checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining history saved to: {history_path}")

    # Generate training curves plot
    try:
        from plot_training import plot_training_history
        curves_path = os.path.join(config.checkpoint_dir, 'training_curves.png')
        plot_training_history(history_path, curves_path)
    except Exception as e:
        print(f"Could not generate training plot: {e}")


if __name__ == "__main__":
    main()
