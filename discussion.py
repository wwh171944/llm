import argparse
import re
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent import LLMAgent
from maze_env import DEFAULT_MAZE, DIR_ALIAS, DIR_DELTA, MazeEnv
from pc1_session import PC1Session
from utils.log_utils import save_json, start_log

MAX_NEW_TOKENS = 64
STEPS_PER_ROUND = 2
MAX_ROUNDS = 16
PC1_AGENT_NAME = "Agent-A"
PC2_AGENT_NAME = "Agent-B"

REVERSE = {"上": "下", "下": "上", "左": "右", "右": "左"}


def build_maze_messages(env, steps_per_round, path_desc):
    legal = env.legal_moves()
    legal_str = "、".join(legal) if legal else "（无，已在死角）"
    system = (
        "你是走迷宫的导航智能体。你只能输出移动方向，方向仅限「上」「下」「左」「右」"
        "四个字，用空格分隔，最多输出规定步数。不要输出任何解释、坐标、标点或其他文字。"
    )
    user = (
        "迷宫如下（* 是你的当前位置，G 是终点，# 是墙，o 是走过的路，. 是空地）：\n"
        f"{env.render()}\n\n"
        f"你在第 {env.pos[0]} 行、第 {env.pos[1]} 列。\n"
        f"合法移动方向（不会撞墙、不越界）：{legal_str}。\n"
        f"已走路径：{path_desc}。\n"
        "目标是走到 G，请不要走回头路。\n"
        f"现在只输出接下来最多 {steps_per_round} 步方向，例如：右 下。你的输出："
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_moves(text, max_steps):
    moves = []
    for ch in re.findall(r"[上下左右↑↓←→]", text):
        direction = DIR_ALIAS.get(ch, ch)
        if direction in DIR_DELTA:
            moves.append(direction)
            if len(moves) >= max_steps:
                break
    return moves


def pick_move(env, proposed, last_move, visited):
    legal = env.legal_moves()
    if not legal:
        return None
    reverse = REVERSE.get(last_move)

    def key(move):
        _, next_pos = env.can_move(env.pos, move)
        return (
            1 if move == reverse else 0,
            1 if next_pos in visited else 0,
            0 if move in proposed else 1,
        )

    return min(legal, key=key)


def communicate(session, pc1_agent, sender_name, receiver_name, messages, round_idx):
    if sender_name == PC1_AGENT_NAME:
        result = session.run_kv_turn(pc1_agent, messages, MAX_NEW_TOKENS)
        direction = "PC1->PC2"
    else:
        result = session.run_token_turn(pc1_agent, messages, MAX_NEW_TOKENS)
        direction = "PC2->PC1"

    record = {
        "round": round_idx,
        "sender": sender_name,
        "receiver": receiver_name,
        "agent": receiver_name,
        "direction": direction,
        "content": result["answer"],
        "mode": result["mode"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "payload_bytes": result["payload_bytes"],
        "kv_size_kb": round(result["payload_bytes"] / 1024, 2) if result["mode"] == "kv" else 0,
        "prefill_ms": round(result["prefill_ms"], 2),
        "decode_ms": round(result["decode_ms"], 2),
        "tx_ms": result["tx_ms"],
        "e2e_ms": round(result["e2e_ms"], 2),
        "timestamp": datetime.now().isoformat(),
    }
    print(
        f"[Round {round_idx}] {sender_name} -> {receiver_name} | "
        f"{direction} | {result['mode'].upper()} | "
        f"payload={record['payload_bytes']}B | E2E={record['e2e_ms']:.0f}ms"
    )
    print(f"    {receiver_name} 输出：{record['content'][:60]!r}")
    return record


def run_maze(session, pc1_agent):
    env = MazeEnv(DEFAULT_MAZE)
    print("\n" + "=" * 60)
    print("走迷宫任务：PC1 发送 KV-Cache，PC2 发送 Token IDs")
    print(env.render())
    print("=" * 60)

    visited = {env.pos}
    last_move = None
    turns = []

    for round_idx in range(1, MAX_ROUNDS + 1):
        if round_idx % 2:
            sender_name, receiver_name = PC1_AGENT_NAME, PC2_AGENT_NAME
        else:
            sender_name, receiver_name = PC2_AGENT_NAME, PC1_AGENT_NAME

        path_desc = " -> ".join(f"({row},{col})" for row, col in env.path)
        messages = build_maze_messages(env, STEPS_PER_ROUND, path_desc)
        record = communicate(
            session,
            pc1_agent,
            sender_name,
            receiver_name,
            messages,
            round_idx,
        )

        proposed = parse_moves(record["content"], STEPS_PER_ROUND)
        applied = []
        for _ in range(STEPS_PER_ROUND):
            move = pick_move(env, proposed, last_move, visited)
            if move is None or not env.step(move):
                break
            applied.append(move)
            visited.add(env.pos)
            last_move = move
            if env.is_goal():
                break

        record.update({
            "moves_proposed": proposed,
            "moves_applied": applied,
            "position": list(env.pos),
            "maze": env.render(),
            "reached_goal": env.is_goal(),
        })
        turns.append(record)
        print(f"    模型提议：{proposed} -> 实际移动：{applied} -> 位置 {env.pos}")
        print(env.render())

        if env.is_goal():
            print(f"[讨论结束] 第 {round_idx} 轮到达终点 G！")
            break

    return {
        "type": "maze_discussion",
        "model": "Qwen2.5-0.5B-Instruct",
        "task": "走迷宫：从 S 到 G",
        "protocol": {"PC1->PC2": "kv", "PC2->PC1": "token"},
        "steps_per_round": STEPS_PER_ROUND,
        "num_rounds": len(turns),
        "success": env.is_goal(),
        "total_steps": len(env.path) - 1,
        "start": list(env.start),
        "goal": list(env.goal),
        "final_position": list(env.pos),
        "final_answer": f"{'到达终点' if env.is_goal() else '未到达'}，共走 {len(env.path) - 1} 步",
        "agents": [PC1_AGENT_NAME, PC2_AGENT_NAME],
        "turns": turns,
        "timestamp": datetime.now().isoformat(),
    }


def load_pc1_agent(model_path):
    print(f"[PC1] 正在加载模型：{model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return LLMAgent(PC1_AGENT_NAME, model, tokenizer, role_desc="导航者，负责规划路线并给出移动方向")


def main():
    parser = argparse.ArgumentParser(description="PC1 迷宫讨论协调器")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--model-path", default=r"E:\llm_project\models\Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    start_log("discussion")
    pc1_agent = load_pc1_agent(args.model_path)
    session = PC1Session(args.host, args.port, args.timeout)
    try:
        session.wait_for_pc2()
        result = run_maze(session, pc1_agent)
        save_json("discussion", {
            "task_type": "maze",
            "num_tasks": 1,
            "success_rate": float(result["success"]),
            "results": [result],
        })
    finally:
        session.close()


if __name__ == "__main__":
    main()
