import numpy as np
import torch

class Node:
    def __init__(self, prior):
        self.P = prior
        self.N = 0
        self.W = 0
        self.Q = 0
        self.children = {}

class MCTS:
    def __init__(self, net, sims=200, c_puct=1.5):
        self.net = net
        self.sims = sims
        self.c_puct = c_puct

    def run(self, env):
        root = Node(0)
        state = torch.tensor(env._get_state()).unsqueeze(0)

        with torch.no_grad():
            policy, _ = self.net(state)
            policy = torch.softmax(policy, dim=1).numpy()[0]
            legal = env.legal_moves()
            policy *= legal
            policy /= policy.sum()

        for a in np.where(legal)[0]:
            root.children[a] = Node(policy[a])

        for _ in range(self.sims):
            self._simulate(env, root)

        visits = np.zeros(82)
        for a, child in root.children.items():
            visits[a] = child.N
        return visits / visits.sum()

    def _simulate(self, env, node):
        if not node.children:
            return 0

        best_a, best_u = None, -1e9
        for a, child in node.children.items():
            u = child.Q + self.c_puct * child.P * np.sqrt(node.N + 1) / (1 + child.N)
            if u > best_u:
                best_u = u
                best_a = a

        next_env = env.__class__()
        next_env.__dict__ = env.__dict__.copy()
        _, _, done = next_env.step(best_a)

        if done:
            value = next_env._winner()
        else:
            state = torch.tensor(next_env._get_state()).unsqueeze(0)
            with torch.no_grad():
                _, value = self.net(state)
            value = value.item()

        child = node.children[best_a]
        child.N += 1
        child.W += value
        child.Q = child.W / child.N
        node.N += 1

        return -value
