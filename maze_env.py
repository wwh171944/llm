# E:\llm_project\maze_env.py
"""
迷宫环境：为多智能体走迷宫讨论提供状态管理。
网格符号：S 起点、G 终点、# 墙、. 通路
渲染符号：* 当前位置、o 已走路径
"""

# 方向 → (行增量, 列增量)
DIR_DELTA = {
    "上": (-1, 0),
    "下": (1, 0),
    "左": (0, -1),
    "右": (0, 1),
}

# 别名归一化（箭头/英文 → 中文方向）
DIR_ALIAS = {
    "↑": "上", "↓": "下", "←": "左", "→": "右",
    "u": "上", "d": "下", "l": "左", "r": "右",
    "U": "上", "D": "下", "L": "左", "R": "右",
}

# 默认 5×5 迷宫（最短路径 8 步，含分支与决策点）
DEFAULT_MAZE = [
    "S . . . .",
    ". # # # .",
    ". . . # .",
    "# . . # .",
    ". . . . G",
]


class MazeEnv:
    def __init__(self, grid):
        self.grid = [row.split() for row in grid]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.start = self._find("S")
        self.goal = self._find("G")
        self.pos = self.start
        self.path = [self.pos]

    def _find(self, ch):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == ch:
                    return (r, c)
        raise ValueError(f"迷宫中找不到符号 {ch}")

    def in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_wall(self, pos):
        return self.grid[pos[0]][pos[1]] == "#"

    def can_move(self, pos, direction):
        direction = DIR_ALIAS.get(direction, direction)
        if direction not in DIR_DELTA:
            return False, pos
        dr, dc = DIR_DELTA[direction]
        new = (pos[0] + dr, pos[1] + dc)
        if not self.in_bounds(new) or self.is_wall(new):
            return False, pos
        return True, new

    def legal_moves(self, pos=None):
        pos = pos or self.pos
        return [d for d in ["上", "下", "左", "右"] if self.can_move(pos, d)[0]]

    def step(self, direction):
        """尝试走一步，成功返回 True 并更新位置与路径"""
        ok, new = self.can_move(self.pos, direction)
        if ok:
            self.pos = new
            self.path.append(new)
        return ok

    def is_goal(self, pos=None):
        return (pos or self.pos) == self.goal

    def render(self, pos=None, mark_path=True):
        """渲染迷宫文本：* 当前位置、G 终点、# 墙、o 走过、. 空地"""
        pos = pos or self.pos
        path_set = set(self.path)
        header = "   " + " ".join(str(c) for c in range(self.cols))
        lines = [header]
        for r in range(self.rows):
            cells = []
            for c in range(self.cols):
                cell = self.grid[r][c]
                if (r, c) == pos:
                    cells.append("*")
                elif cell == "#":
                    cells.append("#")
                elif cell == "G":
                    cells.append("G")
                elif (r, c) == self.start:
                    cells.append("S")
                elif mark_path and (r, c) in path_set:
                    cells.append("o")
                else:
                    cells.append(".")
            lines.append(f"{r} " + " ".join(cells))
        return "\n".join(lines)
