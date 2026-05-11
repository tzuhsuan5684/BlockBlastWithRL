"""
demo/play.py — Pygame visualizer for BlockBlastEnv.

Usage:
    python demo/play.py                                                   # greedy agent
    python demo/play.py --agent random                                    # random agent
    python demo/play.py --agent greedy                                    # greedy agent
    python demo/play.py --agent ppo --checkpoint checkpoints/<file>.pt    # trained PPO
    python demo/play.py --fps 10                                          # slow down
    python demo/play.py --fps 0                                           # manual step: SPACE / ENTER / →
"""

import sys
import argparse
import time
import numpy as np

import pygame

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from env.block_blast_env import BlockBlastEnv, N_ACTIONS
from env.shapes import SHAPES
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent

# ── Layout constants ────────────────────────────────────────────────────────
CELL       = 56          # px per board cell
BOARD_SIZE = 8
MARGIN     = 20          # outer margin
PANEL_W    = 220         # right-side info panel width

BOARD_PX   = CELL * BOARD_SIZE
WIN_W      = MARGIN + BOARD_PX + MARGIN + PANEL_W + MARGIN
WIN_H      = MARGIN + BOARD_PX + MARGIN

PIECE_CELL = 28          # px per cell in piece previews

# ── Colors ──────────────────────────────────────────────────────────────────
BG          = (18,  18,  28)
GRID_LINE   = (40,  40,  60)
CELL_EMPTY  = (28,  28,  42)
CELL_FILLED = (72, 149, 239)   # blue
CELL_FLASH  = (255, 220,  60)  # yellow flash on cleared lines
PIECE_COLORS = [
    (239, 108,  72),   # piece 0 — orange-red
    (76,  201, 120),   # piece 1 — green
    (188,  90, 230),   # piece 2 — purple
]
TEXT_COLOR  = (220, 220, 240)
DIM_COLOR   = (90,  90, 110)
PANEL_BG    = (24,  24,  38)


def draw_board(surf, board: np.ndarray, flash_rows=(), flash_cols=()):
    ox = MARGIN
    oy = MARGIN
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            rect = pygame.Rect(ox + c * CELL, oy + r * CELL, CELL - 2, CELL - 2)
            if r in flash_rows or c in flash_cols:
                color = CELL_FLASH
            elif board[r, c] == 1.0:
                color = CELL_FILLED
            else:
                color = CELL_EMPTY
            pygame.draw.rect(surf, color, rect, border_radius=4)


def draw_grid(surf):
    ox = MARGIN
    oy = MARGIN
    for i in range(BOARD_SIZE + 1):
        pygame.draw.line(surf, GRID_LINE, (ox + i * CELL, oy), (ox + i * CELL, oy + BOARD_PX))
        pygame.draw.line(surf, GRID_LINE, (ox, oy + i * CELL), (ox + BOARD_PX, oy + i * CELL))


def draw_piece_preview(surf, shape, color, ox, oy, label=""):
    font_sm = pygame.font.SysFont("consolas", 14)
    if label:
        lbl = font_sm.render(label, True, DIM_COLOR)
        surf.blit(lbl, (ox, oy - 18))

    cells = set(shape)
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    h = max(rows) + 1 if rows else 1
    w = max(cols) + 1 if cols else 1

    for r in range(h):
        for c in range(w):
            rect = pygame.Rect(ox + c * PIECE_CELL, oy + r * PIECE_CELL,
                               PIECE_CELL - 3, PIECE_CELL - 3)
            if (r, c) in cells:
                pygame.draw.rect(surf, color, rect, border_radius=3)
            else:
                pygame.draw.rect(surf, CELL_EMPTY, rect, border_radius=3)


def draw_panel(surf, score, steps, ep, piece_ids, font, font_lg):
    ox = MARGIN + BOARD_PX + MARGIN
    oy = MARGIN

    pygame.draw.rect(surf, PANEL_BG,
                     pygame.Rect(ox - 8, oy - 8, PANEL_W, WIN_H - MARGIN), border_radius=8)

    # score / steps / episode
    surf.blit(font_lg.render("SCORE", True, DIM_COLOR), (ox, oy))
    surf.blit(font_lg.render(str(score), True, TEXT_COLOR), (ox, oy + 22))

    surf.blit(font.render(f"Steps : {steps}", True, DIM_COLOR), (ox, oy + 60))
    surf.blit(font.render(f"Episode: {ep}", True, DIM_COLOR), (ox, oy + 80))

    # piece previews
    surf.blit(font.render("NEXT PIECES", True, DIM_COLOR), (ox, oy + 120))
    py = oy + 145
    for i, pid in enumerate(piece_ids):
        if pid == -1:
            surf.blit(font.render(f"#{i}  (used)", True, DIM_COLOR), (ox, py))
            py += 40
        else:
            draw_piece_preview(surf, SHAPES[pid], PIECE_COLORS[i],
                               ox, py, label=f"#{i}")
            rows = [r for r, _ in SHAPES[pid]]
            cols = [c for _, c in SHAPES[pid]]
            h = max(rows) + 1 if rows else 1
            py += h * PIECE_CELL + 24


class PPODemoAgent:
    """Wrap a trained PPO checkpoint with the (obs, mask) -> int interface.

    Uses deterministic argmax over masked logits — same convention as
    agents.ppo.evaluate, so demo behavior matches what Member E sees in
    the JSON deliverable.
    """

    def __init__(self, env, checkpoint_path: str):
        import torch
        from agents.network import BlockBlastActorCritic

        self.env = env
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BlockBlastActorCritic().to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print(f"[ppo] loaded {checkpoint_path}  (device={self.device})")

    def select_action(self, obs, action_mask: np.ndarray) -> int:
        torch = self.torch
        with torch.no_grad():
            board  = torch.from_numpy(obs["board"]).to(self.device).unsqueeze(0).unsqueeze(0)
            pieces = torch.from_numpy(obs["pieces"]).to(self.device).unsqueeze(0)
            logits, _ = self.model(board, pieces)
            mask_t = torch.from_numpy(action_mask).to(self.device).unsqueeze(0)
            logits = logits.masked_fill(~mask_t, float("-inf"))
            return int(logits.argmax(dim=1).item())


def build_agent(name: str, env, checkpoint: str = None):
    if name == "random":
        return RandomAgent(env)
    if name == "ppo":
        if not checkpoint:
            raise SystemExit("--checkpoint is required for --agent ppo")
        return PPODemoAgent(env, checkpoint)
    return GreedyAgent(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="greedy", choices=["random", "greedy", "ppo"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to PPO checkpoint .pt (required for --agent ppo)")
    parser.add_argument("--fps", type=float, default=4,
                        help="frames per second (0 = manual step with SPACE)")
    parser.add_argument("--episodes", type=int, default=0,
                        help="stop after N episodes (0 = run forever)")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Block Blast — Agent Demo")
    clock = pygame.time.Clock()

    font    = pygame.font.SysFont("consolas", 16)
    font_lg = pygame.font.SysFont("consolas", 20, bold=True)

    env   = BlockBlastEnv(reward_mode="dense")
    agent = build_agent(args.agent, env, checkpoint=args.checkpoint)

    obs, info = env.reset()
    mask  = info["action_mask"]
    score = 0
    steps = 0
    ep    = 1

    manual    = (args.fps == 0)
    step_now  = False          # flag for manual mode

    # flash state
    flash_rows: list = []
    flash_cols: list = []
    flash_timer = 0.0

    running = True
    while running:
        dt = clock.tick(60) / 1000.0   # real delta time in seconds

        # ── Events ──────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_RIGHT) and manual:
                    step_now = True

        # ── Decide whether to step ──────────────────────────────────────────
        do_step = step_now if manual else True
        step_now = False

        if do_step and flash_timer <= 0:
            action = agent.select_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
            mask  = info["action_mask"]
            score = info["score"]
            steps = info["steps"]

            lines = info["lines_cleared"]
            if lines > 0:
                flash_rows = [r for r in range(BOARD_SIZE)
                              if np.all(env.board[r, :] == 0.0)]  # already cleared
                flash_cols = [c for c in range(BOARD_SIZE)
                              if np.all(env.board[:, c] == 0.0)]
                # re-detect from pre-clear would need extra state; just flash all 0-rows briefly
                flash_timer = 0.25

            if terminated or truncated:
                print(f"Episode {ep} ended — score={score} steps={steps}")
                ep += 1
                if args.episodes and ep > args.episodes:
                    running = False
                obs, info = env.reset()
                mask  = info["action_mask"]
                score = 0
                steps = 0

        # ── Flash timer ─────────────────────────────────────────────────────
        if flash_timer > 0:
            flash_timer -= dt
            if flash_timer <= 0:
                flash_rows = []
                flash_cols = []

        # ── Draw ─────────────────────────────────────────────────────────────
        screen.fill(BG)
        draw_board(screen, env.board, flash_rows, flash_cols)
        draw_grid(screen)
        draw_panel(screen, score, steps, ep,
                   env.piece_shape_ids, font, font_lg)

        # HUD bottom-left
        hint = "SPACE / ENTER / →: step  |  ESC: quit" if manual else f"FPS: {args.fps}  |  ESC: quit"
        screen.blit(font.render(hint, True, DIM_COLOR),
                    (MARGIN, WIN_H - 20))

        pygame.display.flip()

        if not manual:
            time.sleep(max(0, 1.0 / args.fps - dt))

    pygame.quit()


if __name__ == "__main__":
    main()
