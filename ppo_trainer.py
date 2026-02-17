"""
PPO Trainer for Go Network
Trains policy and value networks using self-play data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


class GoDataset(Dataset):
    """Dataset for Go training data"""
    
    def __init__(self, states, policies, values):
        self.states = torch.FloatTensor(states)
        self.policies = torch.FloatTensor(policies)
        self.values = torch.FloatTensor(values)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


class PPOTrainer:
    """PPO-style trainer for Go network"""
    
    def __init__(self, network, learning_rate=0.001, value_loss_coef=1.0, 
                 entropy_coef=0.01, clip_epsilon=0.2, device='cpu'):
        """
        Args:
            network: Go neural network
            learning_rate: Learning rate for optimizer
            value_loss_coef: Weight for value loss
            entropy_coef: Weight for entropy bonus
            clip_epsilon: PPO clipping parameter
            device: torch device
        """
        self.network = network.to(device)
        self.device = device
        
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.clip_epsilon = clip_epsilon
        
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
        }
    
    def train_epoch(self, states, target_policies, target_values, batch_size=64, epochs=1):
        """
        Train for one epoch on the given data
        
        Args:
            states: (N, 5, 9, 9) numpy array
            target_policies: (N, 82) numpy array - MCTS visit distributions
            target_values: (N,) numpy array - game outcomes
            batch_size: Batch size
            epochs: Number of epochs to train
        
        Returns:
            Dictionary of training statistics
        """
        dataset = GoDataset(states, target_policies, target_values)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.network.train()
        
        epoch_stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0,
            'batches': 0
        }
        
        for epoch in range(epochs):
            for batch_states, batch_target_policies, batch_target_values in dataloader:
                batch_states = batch_states.to(self.device)
                batch_target_policies = batch_target_policies.to(self.device)
                batch_target_values = batch_target_values.to(self.device).unsqueeze(1)
                
                # Forward pass
                policy_logits, value_pred = self.network(batch_states)
                
                # Policy loss (cross-entropy with MCTS policy)
                policy_probs = F.softmax(policy_logits, dim=1)
                policy_loss = -torch.sum(batch_target_policies * F.log_softmax(policy_logits, dim=1), dim=1).mean()
                
                # Value loss (MSE)
                value_loss = F.mse_loss(value_pred, batch_target_values)
                
                # Entropy bonus (encourage exploration)
                entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-8), dim=1).mean()
                
                # Total loss
                total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Track statistics
                epoch_stats['policy_loss'] += policy_loss.item()
                epoch_stats['value_loss'] += value_loss.item()
                epoch_stats['entropy'] += entropy.item()
                epoch_stats['total_loss'] += total_loss.item()
                epoch_stats['batches'] += 1
        
        # Average statistics
        for key in ['policy_loss', 'value_loss', 'entropy', 'total_loss']:
            epoch_stats[key] /= epoch_stats['batches']
            self.training_stats[key].append(epoch_stats[key])
        
        return epoch_stats
    
    def get_stats(self):
        """Get training statistics"""
        return self.training_stats


class PPOTrainerWithClipping(PPOTrainer):
    """
    PPO Trainer with actual PPO clipping (not just cross-entropy)
    More faithful to PPO algorithm
    """
    
    def train_epoch(self, states, target_policies, target_values, batch_size=64, epochs=1):
        """Train with PPO clipping objective"""
        dataset = GoDataset(states, target_policies, target_values)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.network.train()
        
        epoch_stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0,
            'batches': 0
        }
        
        for epoch in range(epochs):
            for batch_states, batch_target_policies, batch_target_values in dataloader:
                batch_states = batch_states.to(self.device)
                batch_target_policies = batch_target_policies.to(self.device)
                batch_target_values = batch_target_values.to(self.device).unsqueeze(1)
                
                # Get old policy probabilities (detached)
                with torch.no_grad():
                    old_policy_logits, _ = self.network(batch_states)
                    old_policy_probs = F.softmax(old_policy_logits, dim=1)
                
                # Forward pass
                policy_logits, value_pred = self.network(batch_states)
                policy_probs = F.softmax(policy_logits, dim=1)
                
                # PPO clipped policy loss
                # Ratio of new policy to old policy
                ratio = policy_probs / (old_policy_probs + 1e-8)
                
                # Weighted by target policy (MCTS distribution)
                weighted_ratio = ratio * batch_target_policies
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_target_policies
                
                policy_loss = -torch.min(weighted_ratio, clipped_ratio).sum(dim=1).mean()
                
                # Value loss (MSE)
                value_loss = F.mse_loss(value_pred, batch_target_values)
                
                # Entropy bonus
                entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-8), dim=1).mean()
                
                # Total loss
                total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Track statistics
                epoch_stats['policy_loss'] += policy_loss.item()
                epoch_stats['value_loss'] += value_loss.item()
                epoch_stats['entropy'] += entropy.item()
                epoch_stats['total_loss'] += total_loss.item()
                epoch_stats['batches'] += 1
        
        # Average statistics
        for key in ['policy_loss', 'value_loss', 'entropy', 'total_loss']:
            epoch_stats[key] /= epoch_stats['batches']
            self.training_stats[key].append(epoch_stats[key])
        
        return epoch_stats


def create_trainer(network, config, device='cpu'):
    """
    Factory function to create trainer
    
    Args:
        network: Go network
        config: Dict with training parameters
        device: torch device
    """
    return PPOTrainer(
        network,
        learning_rate=config.get('learning_rate', 0.001),
        value_loss_coef=config.get('value_loss_coef', 1.0),
        entropy_coef=config.get('entropy_coef', 0.01),
        clip_epsilon=config.get('clip_epsilon', 0.2),
        device=device
    )


def test_trainer():
    """Test the trainer"""
    import sys
    sys.path.append('/home/claude/go_rl_agent')
    from network import create_network
    
    # Create network
    network = create_network()
    
    # Create trainer
    config = {
        'learning_rate': 0.001,
        'value_loss_coef': 1.0,
        'entropy_coef': 0.01
    }
    trainer = create_trainer(network, config)
    
    # Create dummy data
    batch_size = 32
    states = np.random.randn(batch_size, 5, 9, 9).astype(np.float32)
    policies = np.random.rand(batch_size, 82).astype(np.float32)
    policies = policies / policies.sum(axis=1, keepdims=True)  # Normalize
    values = np.random.randn(batch_size).astype(np.float32)
    
    # Train
    print("Training for one epoch...")
    stats = trainer.train_epoch(states, policies, values, batch_size=16, epochs=1)
    
    print(f"Policy loss: {stats['policy_loss']:.4f}")
    print(f"Value loss: {stats['value_loss']:.4f}")
    print(f"Entropy: {stats['entropy']:.4f}")
    print(f"Total loss: {stats['total_loss']:.4f}")


if __name__ == "__main__":
    test_trainer()
