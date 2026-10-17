import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent import LLMAgent
from comm_utils import pack_kv, unpack_kv
from maze_env import DIR_ALIAS, MazeEnv
from pc1_chat_server import build_maze_messages

MODEL_PATH = r"E:\llm_project\models\Qwen2.5-0.5B-Instruct"
LOG_DIR = r"E:\llm_project\logs"
DIRECTION_CASES = [
    ("only_up", "上", ["# G #", "# . #", "# S #"]),
    ("only_down", "下", ["# S #", "# . #", "# G #"]),
    ("only_left", "左", ["# # #", "G . S", "# # #"]),
    ("only_right", "右", ["# # #", "S . G", "# # #"]),
    (
        "open_5x5_goal_down_right",
        "下或右",
        [". . . . .", ". S . . .", ". . . . .", ". . G . .", ". . . . ."],
    ),
]


def parse_direction(answer):
    match = re.search(r"[上下左右↑↓←→]", answer)
    return DIR_ALIAS.get(match.group(), match.group()) if match else None


def load_agent(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return LLMAgent("Direction-Test-Agent", model, tokenizer)


def run_case(agent, name, expected, grid, max_new_tokens):
    env = MazeEnv(grid)
    messages = build_maze_messages(env, agent.name, [])

    token_result = agent.generate_full(messages, max_new_tokens)
    prefill = agent.prefill(messages)
    restored = unpack_kv(pack_kv(prefill["past"], prefill["first_token"]))
    restored["kv_seq_len"] = prefill["kv_seq_len"]
    kv_result = agent.decode_from_kv(restored, max_new_tokens)

    return {
        "case": name,
        "expected_direction": expected,
        "legal_moves": env.legal_moves(),
        "token_answer": token_result["answer"],
        "token_direction": parse_direction(token_result["answer"]),
        "kv_answer": kv_result["answer"],
        "kv_direction": parse_direction(kv_result["answer"]),
        "kv_payload_bytes": len(pack_kv(prefill["past"], prefill["first_token"])),
    }


def main():
    parser = argparse.ArgumentParser(description="模型方向偏置诊断")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    agent = load_agent(args.model_path)
    results = [
        run_case(agent, name, expected, grid, args.max_new_tokens)
        for name, expected, grid in DIRECTION_CASES
    ]
    token_counts = Counter(result["token_direction"] or "无有效方向" for result in results)
    kv_counts = Counter(result["kv_direction"] or "无有效方向" for result in results)

    print("========== 方向偏置诊断 ==========")
    for result in results:
        print(
            f"{result['case']}: 期望={result['expected_direction']} | "
            f"Token={result['token_direction'] or '无'} ({result['token_answer']!r}) | "
            f"KV={result['kv_direction'] or '无'} ({result['kv_answer']!r})"
        )
    print(f"Token 输出分布：{dict(token_counts)}")
    print(f"KV 输出分布：{dict(kv_counts)}")

    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(
        LOG_DIR, f"direction_bias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "model_path": args.model_path,
                "results": results,
                "token_direction_counts": dict(token_counts),
                "kv_direction_counts": dict(kv_counts),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"诊断结果已保存：{path}")


if __name__ == "__main__":
    main()
