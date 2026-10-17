import argparse
import re
import threading
import time
from collections import deque
from datetime import datetime

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent import LLMAgent
from maze_env import DIR_ALIAS, DIR_DELTA, MazeEnv
from pc1_session import PC1Session
from utils.log_utils import save_json, start_log

PC1_AGENT_NAME = "Agent-A"
PC2_AGENT_NAME = "Agent-B"


class DiscussionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    rounds: int = Field(default=4, ge=2, le=8)
    max_new_tokens: int = Field(default=64, ge=16, le=128)


class MazeRequest(BaseModel):
    maze: str = Field(min_length=9, max_length=5000)
    max_rounds: int = Field(default=32, ge=1, le=100)
    bandwidth_mbps: float = Field(default=1000, gt=0, le=100000)


def build_messages(question, history, speaker, stage):
    transcript = "\n".join(
        f"{turn['agent']}：{turn['content']}" for turn in history
    ) or "（尚未开始讨论）"
    role = "严谨分析者" if speaker == PC1_AGENT_NAME else "批判审阅者"
    return [
        {
            "role": "system",
            "content": (
                f"你是 {speaker}，角色是{role}。直接回答问题，不要描述智能体协作过程，"
                "不要说‘我来介绍’、‘接下来’或要求另一位智能体继续。"
                "每次只输出一个简洁观点，最多 80 个汉字。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"问题：{question}\n\n已有讨论：\n{transcript}\n\n"
                f"本轮任务：{stage}\n请直接给出你的观点："
            ),
        },
    ]


def parse_maze(text):
    rows = [line.strip().split() for line in text.strip().splitlines() if line.strip()]
    if len(rows) < 2 or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("迷宫必须至少有两行，且每行格子数量相同")
    if any(cell not in {"S", "G", "#", "."} for row in rows for cell in row):
        raise ValueError("迷宫只允许 S、G、#、. 四种符号，并以空格分隔")
    if sum(cell == "S" for row in rows for cell in row) != 1:
        raise ValueError("迷宫必须且只能有一个起点 S")
    if sum(cell == "G" for row in rows for cell in row) != 1:
        raise ValueError("迷宫必须且只能有一个出口 G")
    return MazeEnv([" ".join(row) for row in rows])


def parse_direction(answer):
    match = re.search(r"[上下左右↑↓←→]", answer)
    return DIR_ALIAS.get(match.group(), match.group()) if match else None


def invalid_move_feedback(env, direction, allowed_moves=None):
    if direction is None:
        return "未识别到方向，当前位置保持不变。"
    if allowed_moves is not None and direction not in allowed_moves:
        return f"{direction}不在当前可选方向中，当前位置保持不变。"
    dr, dc = DIR_DELTA[direction]
    attempted = (env.pos[0] + dr, env.pos[1] + dc)
    if not env.in_bounds(attempted):
        return f"{direction}会越过边界，当前位置保持在({env.pos[0]},{env.pos[1]})。"
    return f"{direction}前方是障碍物，当前位置保持在({env.pos[0]},{env.pos[1]})。"


def direction_token_ids(tokenizer, directions):
    token_ids = []
    for direction in directions:
        encoded = tokenizer.encode(direction, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"方向“{direction}”无法编码为单个 token")
        token_ids.append(encoded[0])
    return token_ids


def direction_to(env, target):
    for direction in env.legal_moves():
        _, next_position = env.can_move(env.pos, direction)
        if next_position == target:
            return direction
    return None


def exploration_moves(env, visited_positions, search_stack):
    unvisited = []
    for direction in env.legal_moves():
        _, next_position = env.can_move(env.pos, direction)
        if next_position not in visited_positions:
            unvisited.append(direction)
    if unvisited:
        return unvisited, False
    if len(search_stack) > 1:
        return [direction_to(env, search_stack[-2])], True
    return [], False


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


def build_maze_messages(
    env, speaker, history, feedback="", proposal=None, allowed_moves=None, backtracking=False
):
    allowed_moves = allowed_moves or env.legal_moves()
    candidates = []
    for direction in allowed_moves:
        _, next_position = env.can_move(env.pos, direction)
        distance = abs(next_position[0] - env.goal[0]) + abs(next_position[1] - env.goal[1])
        candidates.append((direction, next_position, distance))

    min_distance = min(distance for _, _, distance in candidates)
    best_directions = "、".join(
        direction for direction, _, distance in candidates if distance == min_distance
    )
    candidate_text = "；".join(
        f"{direction}->{distance}" for direction, _, distance in candidates
    )
    recent_moves = "、".join(history[-2:]) or "（暂无）"
    phase = (
        "这是提议阶段，请独立选择一个方向。"
        if proposal is None
        else f"这是审核阶段，另一名智能体提议“{proposal or '无效'}”。请核对候选评分后给出最终方向。"
    )
    exploration_instruction = (
        "当前分支没有未访问格子，必须按唯一候选方向回溯。"
        if backtracking
        else "候选方向均通向未访问格子。"
    )
    return [
        {
            "role": "system",
            "content": (
                f"你是 {speaker}，正在与另一名智能体协作走迷宫。{phase}"
                "只能输出一个汉字：上、下、左或右。不要输出理由、标点、坐标或其他文字。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"当前位置：({env.pos[0]},{env.pos[1]})，出口：({env.goal[0]},{env.goal[1]})。\n"
                f"候选评分：{candidate_text}。最小距离={min_distance}，可选方向：{best_directions}。\n"
                f"探索状态：{exploration_instruction}\n"
                f"最近实际执行方向：{recent_moves}。上轮反馈：{feedback or '无'}。\n"
                "仅输出最终方向："
            ),
        },
    ]


def load_agent(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return LLMAgent(PC1_AGENT_NAME, model, tokenizer, role_desc="负责分析问题和给出结论")


def create_app(host, port, timeout_s, model_path):
    app = FastAPI(title="LatentMAS Experimental Chat Coordinator")
    session = PC1Session(host, port, timeout_s)
    pc1_agent = load_agent(model_path)
    connected = threading.Event()
    discussion_lock = threading.Lock()

    def wait_for_pc2():
        try:
            session.wait_for_pc2()
            session.conn.settimeout(None)
            connected.set()
        except OSError as error:
            print(f"[PC1] 等待 PC2 失败：{error}")

    @app.on_event("startup")
    def start_connection_listener():
        threading.Thread(target=wait_for_pc2, daemon=True).start()

    @app.on_event("shutdown")
    def close_connection():
        session.close()

    @app.get("/health")
    def health():
        return {"status": "ready" if connected.is_set() else "waiting_for_pc2"}

    @app.post("/discuss")
    def discuss(request: DiscussionRequest):
        if not connected.is_set():
            raise HTTPException(status_code=503, detail="PC2 尚未连接")

        with discussion_lock:
            history = []
            turns = []
            started = time.time()
            try:
                for round_idx in range(1, request.rounds + 1):
                    if round_idx % 2:
                        sender, receiver = PC1_AGENT_NAME, PC2_AGENT_NAME
                        stage = (
                            "审阅已有观点，指出一处需要补充或修正的地方，并给出更准确的表述。"
                            if history else "给出问题的核心定义、前提或关键推理。"
                        )
                        result = session.run_kv_turn(
                            pc1_agent,
                            build_messages(request.question, history, receiver, stage),
                            request.max_new_tokens,
                        )
                        direction = "PC1->PC2"
                    else:
                        sender, receiver = PC2_AGENT_NAME, PC1_AGENT_NAME
                        stage = (
                            "综合已有观点，给出最终结论或可执行答案，不要重复前文。"
                            if round_idx == request.rounds else
                            "检查上一轮观点的准确性，并补充一个遗漏的关键条件或例子。"
                        )
                        result = session.run_token_turn(
                            pc1_agent,
                            build_messages(request.question, history, receiver, stage),
                            request.max_new_tokens,
                        )
                        direction = "PC2->PC1"

                    turn = {
                        "round": round_idx,
                        "sender": sender,
                        "receiver": receiver,
                        "direction": direction,
                        "mode": result["mode"],
                        "content": result["answer"],
                        "input_tokens": result["input_tokens"],
                        "output_tokens": result["output_tokens"],
                        "payload_bytes": result["payload_bytes"],
                        "prefill_ms": round(result["prefill_ms"], 2),
                        "decode_ms": round(result["decode_ms"], 2),
                        "tx_ms": round(result["tx_ms"], 2),
                        "e2e_ms": round(result["e2e_ms"], 2),
                    }
                    turns.append(turn)
                    history.append({"agent": receiver, "content": result["answer"]})
            except (ConnectionError, OSError, ValueError) as error:
                connected.clear()
                raise HTTPException(status_code=502, detail=f"与 PC2 通信失败：{error}") from error

        result = {
            "type": "experimental_discussion",
            "question": request.question,
            "protocol": {"PC1->PC2": "kv", "PC2->PC1": "token"},
            "num_rounds": len(turns),
            "total_e2e_ms": round((time.time() - started) * 1000, 2),
            "turns": turns,
            "timestamp": datetime.now().isoformat(),
        }
        save_json("experimental_discussion", result)
        return result

    @app.post("/solve-maze")
    def solve_maze(request: MazeRequest):
        if not connected.is_set():
            raise HTTPException(status_code=503, detail="PC2 尚未连接")
        try:
            env = parse_maze(request.maze)
            if shortest_path(env) is None:
                raise ValueError("入口 S 与出口 G 不连通，当前地图无解")
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        with discussion_lock:
            turns = []
            history = []
            feedback = ""
            visited_positions = {env.pos}
            search_stack = [env.pos]
            started = time.time()
            try:
                for round_idx in range(1, request.max_rounds + 1):
                    if env.is_goal():
                        break
                    if round_idx % 2:
                        sender, receiver = PC1_AGENT_NAME, PC2_AGENT_NAME
                    else:
                        sender, receiver = PC2_AGENT_NAME, PC1_AGENT_NAME
                    allowed_moves, backtracking = exploration_moves(
                        env, visited_positions, search_stack
                    )
                    if not allowed_moves:
                        feedback = "所有可探索分支均已访问，无法继续搜索。"
                        break

                    allowed_token_ids = direction_token_ids(
                        pc1_agent.tokenizer, allowed_moves
                    )
                    proposal_result = session.run_adaptive_turn(
                        pc1_agent,
                        build_maze_messages(
                            env, receiver, history, feedback,
                            allowed_moves=allowed_moves, backtracking=backtracking,
                        ),
                        1,
                        request.bandwidth_mbps,
                        allowed_token_ids,
                    )
                    proposal = parse_direction(proposal_result["answer"])
                    result = session.run_adaptive_turn(
                        pc1_agent,
                        build_maze_messages(
                            env, sender, history, feedback, proposal,
                            allowed_moves, backtracking,
                        ),
                        1,
                        request.bandwidth_mbps,
                        allowed_token_ids,
                    )
                    direction = f"提议:{proposal_result['mode'].upper()} / 审核:{result['mode'].upper()}"

                    proposed = parse_direction(result["answer"])
                    valid = proposed in allowed_moves
                    applied = proposed if valid else None
                    if applied:
                        env.step(applied)
                        history.append(applied)
                        if backtracking:
                            search_stack.pop()
                        else:
                            visited_positions.add(env.pos)
                            search_stack.append(env.pos)
                        feedback = ""
                    else:
                        feedback = invalid_move_feedback(env, proposed, allowed_moves)

                    turns.append({
                        "round": round_idx,
                        "sender": sender,
                        "receiver": receiver,
                        "proposer": receiver,
                        "reviewer": sender,
                        "direction": direction,
                        "mode": result["mode"],
                        "content": result["answer"],
                        "proposal_content": proposal_result["answer"],
                        "proposal_move": proposal,
                        "proposed_move": proposed,
                        "applied_move": applied,
                        "move_source": (
                            "model_backtrack" if valid and backtracking
                            else "model" if valid else "model_invalid"
                        ),
                        "position": list(env.pos),
                        "maze": env.render(),
                        "reached_goal": env.is_goal(),
                        "payload_bytes": proposal_result["payload_bytes"] + result["payload_bytes"],
                        "bandwidth_mbps": result["bandwidth_mbps"],
                        "selection_reason": result["selection_reason"],
                        "estimated_token_e2e_ms": round(
                            proposal_result["estimated_token_e2e_ms"] + result["estimated_token_e2e_ms"], 2
                        ),
                        "estimated_kv_e2e_ms": round(
                            proposal_result["estimated_kv_e2e_ms"] + result["estimated_kv_e2e_ms"], 2
                        ),
                        "selection_kv_bytes": proposal_result["selection_kv_bytes"] + result["selection_kv_bytes"],
                        "prefill_ms": round(proposal_result["prefill_ms"] + result["prefill_ms"], 2),
                        "decode_ms": round(proposal_result["decode_ms"] + result["decode_ms"], 2),
                        "tx_ms": round(proposal_result["tx_ms"] + result["tx_ms"], 2),
                        "e2e_ms": round(proposal_result["e2e_ms"] + result["e2e_ms"], 2),
                    })
            except (ConnectionError, OSError, ValueError) as error:
                connected.clear()
                raise HTTPException(status_code=502, detail=f"与 PC2 通信失败：{error}") from error

        response = {
            "type": "experimental_maze",
            "maze_input": request.maze,
            "protocol": "adaptive",
            "bandwidth_mbps": request.bandwidth_mbps,
            "success": env.is_goal(),
            "num_rounds": len(turns),
            "final_position": list(env.pos),
            "total_e2e_ms": round((time.time() - started) * 1000, 2),
            "turns": turns,
            "timestamp": datetime.now().isoformat(),
        }
        save_json("experimental_maze", response)
        return response

    return app


def main():
    parser = argparse.ArgumentParser(description="PC1 实验对话协调服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9003)
    parser.add_argument("--api-port", type=int, default=9010)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--model-path", default=r"E:\llm_project\models\Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    start_log("pc1_chat_server")
    import uvicorn
    uvicorn.run(
        create_app(args.host, args.port, args.timeout, args.model_path),
        host="127.0.0.1",
        port=args.api_port,
    )


if __name__ == "__main__":
    main()
