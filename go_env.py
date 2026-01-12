import numpy as np

BOARD_SIZE = 9
MAX_MOVES = 200
PASS_MOVE = 81

class GoEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.prev_board = None
        self.player = 1
        self.move_count = 0
        self.passes = 0
        return self._get_state()

    def _in_bounds(self, x, y):
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

    def _neighbors(self, x, y):
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if self._in_bounds(nx, ny):
                yield nx, ny

    def _get_group(self, x, y, visited):
        color = self.board[x,y]
        stack = [(x,y)]
        group = []
        liberties = set()

        while stack:
            cx, cy = stack.pop()
            if (cx,cy) in visited:
                continue
            visited.add((cx,cy))
            group.append((cx,cy))
            for nx, ny in self._neighbors(cx, cy):
                if self.board[nx,ny] == 0:
                    liberties.add((nx,ny))
                elif self.board[nx,ny] == color:
                    stack.append((nx,ny))
        return group, liberties

    def _remove_dead(self, opponent):
        removed = False
        visited = set()
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if self.board[x,y] == opponent and (x,y) not in visited:
                    group, libs = self._get_group(x,y,visited)
                    if len(libs) == 0:
                        for gx, gy in group:
                            self.board[gx,gy] = 0
                        removed = True
        return removed

    def legal_moves(self):
        mask = np.zeros(BOARD_SIZE*BOARD_SIZE+1, dtype=np.int8)
        for i in range(BOARD_SIZE*BOARD_SIZE):
            x, y = divmod(i, BOARD_SIZE)
            if self.board[x,y] != 0:
                continue
            backup = self.board.copy()
            self.board[x,y] = self.player
            removed = self._remove_dead(-self.player)
            if not removed:
                visited = set()
                _, libs = self._get_group(x,y,visited)
                if len(libs) == 0:
                    self.board = backup
                    continue
            if self.prev_board is not None and np.array_equal(self.board, self.prev_board):
                self.board = backup
                continue
            mask[i] = 1
            self.board = backup
        mask[PASS_MOVE] = 1
        return mask

    def step(self, action):
        self.prev_board = self.board.copy()
        reward = 0
        done = False

        if action == PASS_MOVE:
            self.passes += 1
        else:
            x, y = divmod(action, BOARD_SIZE)
            self.board[x,y] = self.player
            self._remove_dead(-self.player)
            self.passes = 0

        self.move_count += 1
        if self.passes >= 2 or self.move_count >= MAX_MOVES:
            done = True
            reward = self._winner()

        self.player *= -1
        return self._get_state(), reward, done

    def _winner(self):
        black = np.sum(self.board == 1)
        white = np.sum(self.board == -1)
        return 1 if black > white else -1

    def _get_state(self):
        state = np.zeros((5, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        state[0] = (self.board == 1)
        state[1] = (self.board == -1)
        state[2].fill(1 if self.player == 1 else 0)
        state[3].flat[:BOARD_SIZE*BOARD_SIZE+1] = self.legal_moves()[:-1]
        if self.prev_board is not None:
            state[4] = self.prev_board == self.player
        return state
