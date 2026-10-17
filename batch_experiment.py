import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from comm_utils import iter_kv
from maze_env import DIR_ALIAS, DIR_DELTA, MazeEnv
from mode_selector import find_crossover_bandwidth, kv_tx_ms, select_mode, token_tx_ms
from utils.log_utils import save_json, start_log

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-0.5B-Instruct")
DEFAULT_BANDWIDTHS_MBPS = [1, 10, 100, 1000]

MAZES = [
    {
        "name": "easy_5x5",
        "difficulty": "简单",
        "grid": [
            "S . . . .",
            ". # # # .",
            ". . . # .",
            "# . . # .",
            ". . . . G",
        ],
    },
    {
        "name": "medium_6x6",
        "difficulty": "中等",
        "grid": [
            "S . . . . .",
            "# # # # # .",
            ". . . . # .",
            ". # # . # .",
            ". # . . . .",
            ". . . # # G",
        ],
    },
    {
        "name": "hard_7x7",
        "difficulty": "困难",
        "grid": [
            "S . . . . . .",
            "# # # # # # .",
            ". . . . . . .",
            ". # # # # # #",
            ". . . . . . .",
            "# # # # # # .",
            ". . . . . . G",
        ],
    },
]


def shortest_path(env, start=None):
    start = start or env.pos
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        position, path = queue.popleft()
        if position == env.goal:
            return path
        for direction in DIR_DELTA:
            valid, next_position = env.can_move(position, direction)
            if valid and next_position not in visited:
                visited.add(next_position)
                queue.append((next_position, path + [direction]))
    return None


def build_maze_question(env):
    legal_moves = "、".join(env.legal_moves())
    return (
        "你是走迷宫导航智能体。迷宫中 * 是当前位置，G 是终点，# 是墙，. 是可通行位置。\n"
        f"迷宫：\n{env.render()}\n\n"
        f"当前位置：第 {env.pos[0]} 行、第 {env.pos[1]} 列，合法移动方向：{legal_moves}。\n"
        "请只输出下一步的一个移动方向（上、下、左、右），不要解释。"
    )


def parse_direction(answer):
    for character in answer:
        direction = DIR_ALIAS.get(character, character)
        if direction in DIR_DELTA:
            return direction
    return None


def measure_prefill(model, tokenizer, question):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_tokens = int(inputs.input_ids.shape[1])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.time()
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    kv_bytes = 0
    kv_seq_len = 0
    for key, value in iter_kv(outputs.past_key_values):
        kv_bytes += key.numel() * key.element_size() + value.numel() * value.element_size()
        kv_seq_len = int(key.shape[-2])
    return {
        "past": outputs.past_key_values,
        "first_token": outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True),
        "input_tokens": input_tokens,
        "prefill_ms": (time.time() - started) * 1000,
        "kv_bytes": kv_bytes,
        "kv_seq_len": kv_seq_len,
    }


def manual_decode_kv(model, tokenizer, prefill_data, max_new_tokens):
    cache = DynamicCache()
    for layer_idx, (key, value) in enumerate(iter_kv(prefill_data["past"])):
        cache.update(key.to(model.device), value.to(model.device), layer_idx)
    current_input = prefill_data["first_token"]
    current_attn = torch.ones(
        (1, prefill_data["kv_seq_len"] + 1), dtype=torch.long, device=model.device
    )
    current_pos = torch.tensor([[prefill_data["kv_seq_len"]]], device=model.device)
    generated = []
    if int(current_input.item()) != tokenizer.eos_token_id:
        generated.append(current_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.time()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=current_input,
                past_key_values=cache,
                attention_mask=current_attn,
                position_ids=current_pos,
                use_cache=True,
            )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if int(next_token.item()) == tokenizer.eos_token_id:
            break
        generated.append(next_token)
        cache = outputs.past_key_values
        current_input = next_token
        current_attn = torch.cat(
            [current_attn, torch.ones((1, 1), dtype=torch.long, device=model.device)], dim=-1
        )
        current_pos = current_pos + 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    answer = ""
    if generated:
        answer = tokenizer.decode(torch.cat(generated, dim=-1)[0], skip_special_tokens=True)
    return {
        "answer": answer,
        "decode_ms": (time.time() - started) * 1000,
        "output_tokens": len(generated),
    }


def measure_turn(model, tokenizer, question, max_new_tokens, bandwidth_mbps):
    prefill = measure_prefill(model, tokenizer, question)
    decoded = manual_decode_kv(model, tokenizer, prefill, max_new_tokens)
    output_tokens = decoded["output_tokens"]
    token_e2e_ms = (
        prefill["prefill_ms"]
        + decoded["decode_ms"]
        + token_tx_ms(output_tokens, bandwidth_mbps)
    )
    kv_e2e_ms = (
        kv_tx_ms(prefill["kv_bytes"], bandwidth_mbps)
        + decoded["decode_ms"]
        + token_tx_ms(output_tokens, bandwidth_mbps)
    )
    adaptive_mode, _, _, adaptive_reason = select_mode(
        prefill["prefill_ms"],
        decoded["decode_ms"],
        output_tokens,
        prefill["kv_bytes"],
        bandwidth_mbps,
    )
    return {
        "answer": decoded["answer"],
        "input_tokens": prefill["input_tokens"],
        "output_tokens": output_tokens,
        "prefill_ms": prefill["prefill_ms"],
        "decode_ms": decoded["decode_ms"],
        "kv_bytes": prefill["kv_bytes"],
        "token_e2e_ms": token_e2e_ms,
        "kv_e2e_ms": kv_e2e_ms,
        "adaptive_mode": adaptive_mode,
        "adaptive_reason": adaptive_reason,
        "adaptive_e2e_ms": kv_e2e_ms if adaptive_mode == "kv" else token_e2e_ms,
        "crossover_bandwidth_mbps": find_crossover_bandwidth(
            prefill["prefill_ms"], decoded["decode_ms"], output_tokens, prefill["kv_bytes"]
        ),
    }


def run_maze(model, tokenizer, maze, max_new_tokens, bandwidth_mbps):
    env = MazeEnv(maze["grid"])
    optimal_path = shortest_path(env)
    if optimal_path is None:
        raise ValueError(f"迷宫 {maze['name']} 不连通")

    max_steps = len(optimal_path) * 2
    totals = {"token": 0.0, "kv": 0.0, "adaptive": 0.0}
    mode_counts = {"token": 0, "kv": 0}
    model_progress_moves = 0
    fallback_moves = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_kv_bytes = 0
    total_prefill_ms = 0.0
    total_decode_ms = 0.0

    for _ in range(max_steps):
        if env.is_goal():
            break
        route = shortest_path(env)
        turn = measure_turn(
            model, tokenizer, build_maze_question(env), max_new_tokens, bandwidth_mbps
        )
        proposed = parse_direction(turn["answer"])
        valid, next_position = env.can_move(env.pos, proposed) if proposed else (False, env.pos)
        remaining = shortest_path(env, next_position) if valid else None
        progresses = valid and remaining is not None and len(remaining) < len(route)
        move = proposed if progresses else route[0]
        env.step(move)

        model_progress_moves += int(progresses)
        fallback_moves += int(not progresses)
        totals["token"] += turn["token_e2e_ms"]
        totals["kv"] += turn["kv_e2e_ms"]
        totals["adaptive"] += turn["adaptive_e2e_ms"]
        mode_counts[turn["adaptive_mode"]] += 1
        total_input_tokens += turn["input_tokens"]
        total_output_tokens += turn["output_tokens"]
        total_kv_bytes += turn["kv_bytes"]
        total_prefill_ms += turn["prefill_ms"]
        total_decode_ms += turn["decode_ms"]

    success = env.is_goal()
    steps = len(env.path) - 1
    return {
        "maze_name": maze["name"],
        "difficulty": maze["difficulty"],
        "optimal_steps": len(optimal_path),
        "max_steps": max_steps,
        "completed_steps": steps,
        "path_efficiency": round(len(optimal_path) / steps, 4) if success and steps else 0.0,
        "token_success": success,
        "kv_success": success,
        "adaptive_success": success,
        "token_e2e_ms": round(totals["token"], 2),
        "kv_e2e_ms": round(totals["kv"], 2),
        "adaptive_e2e_ms": round(totals["adaptive"], 2),
        "model_progress_moves": model_progress_moves,
        "safety_fallback_moves": fallback_moves,
        "model_progress_rate": round(model_progress_moves / steps, 4) if steps else 0.0,
        "adaptive_kv_turns": mode_counts["kv"],
        "adaptive_token_turns": mode_counts["token"],
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "kv_bytes": total_kv_bytes,
        "kv_size_kb": round(total_kv_bytes / 1024, 2),
        "prefill_ms": round(total_prefill_ms, 2),
        "decode_ms": round(total_decode_ms, 2),
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description="完整迷宫通关率实验")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--bandwidths", nargs="+", type=float, default=DEFAULT_BANDWIDTHS_MBPS)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--maze-names", nargs="+", choices=[maze["name"] for maze in MAZES])
    return parser.parse_args()


def main():
    args = parse_arguments()
    selected_mazes = [
        maze for maze in MAZES if args.maze_names is None or maze["name"] in args.maze_names
    ]
    os.makedirs(LOG_DIR, exist_ok=True)
    start_log("batch_experiment")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    rows = []
    for bandwidth_mbps in args.bandwidths:
        print(f"\n{'=' * 60}\n带宽：{bandwidth_mbps:g} Mbps\n{'=' * 60}")
        for maze in selected_mazes:
            row = run_maze(model, tokenizer, maze, args.max_new_tokens, bandwidth_mbps)
            row.update({"bandwidth_mbps": bandwidth_mbps, "max_new_tokens": args.max_new_tokens})
            rows.append(row)
            print(
                f"{maze['name']}: {'通关' if row['adaptive_success'] else '失败'}，"
                f"{row['completed_steps']}/{row['optimal_steps']} 步，"
                f"token={row['token_e2e_ms']:.1f}ms，"
                f"kv={row['kv_e2e_ms']:.1f}ms，"
                f"adaptive={row['adaptive_e2e_ms']:.1f}ms"
            )

    results = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(LOG_DIR, f"experiments_{timestamp}.csv")
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = results.groupby("bandwidth_mbps").agg(
        token_success=("token_success", "mean"),
        kv_success=("kv_success", "mean"),
        adaptive_success=("adaptive_success", "mean"),
        token_e2e_ms=("token_e2e_ms", "mean"),
        kv_e2e_ms=("kv_e2e_ms", "mean"),
        adaptive_e2e_ms=("adaptive_e2e_ms", "mean"),
        completed_steps=("completed_steps", "mean"),
        path_efficiency=("path_efficiency", "mean"),
        model_progress_rate=("model_progress_rate", "mean"),
        safety_fallback_moves=("safety_fallback_moves", "sum"),
    ).round(4)
    print("\n========== 完整迷宫实验汇总 ==========")
    print(summary)
    print(f"\nCSV 已保存：{csv_path}")

    save_json("batch_experiment", {
        "experiment_type": "full_maze_completion",
        "num_mazes": len(selected_mazes),
        "bandwidths_mbps": args.bandwidths,
        "max_new_tokens": args.max_new_tokens,
        "csv_path": csv_path,
        "summary": summary.reset_index().to_dict(orient="records"),
        "mazes": [{key: value for key, value in maze.items() if key != "grid"} for maze in selected_mazes],
    })


if __name__ == "__main__":
    main()
