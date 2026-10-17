import argparse
import socket
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent import LLMAgent, select_next_token
from comm_utils import unpack_kv
from distributed_comm import connect, recv_frame, send_frame


def load_agent(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return LLMAgent("Agent-B", model, tokenizer, role_desc="导航者，负责规划路线并给出移动方向")


def generate_token_ids(agent, messages, max_new_tokens, allowed_token_ids=None):
    inputs = agent.encode(messages)
    input_tokens = int(inputs.input_ids.shape[1])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.time()
    with torch.no_grad():
        if allowed_token_ids:
            output = agent.model(**inputs, use_cache=True)
            token_ids = select_next_token(
                output.logits[:, -1, :], allowed_token_ids
            ).cpu().tolist()[0]
        else:
            output = agent.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            token_ids = output[0][input_tokens:].cpu().tolist()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {
        "token_ids": token_ids,
        "input_tokens": input_tokens,
        "output_tokens": len(token_ids),
        "compute_ms": (time.time() - started) * 1000,
    }


def serve(agent, host, port, timeout_s):
    while True:
        try:
            conn = connect(host, port, timeout_s)
            conn.settimeout(None)
            print(f"[PC2] 已连接 PC1：{host}:{port}")
            while True:
                header, payload = recv_frame(conn)
                message_type = header.get("type")
                request_id = header.get("request_id")

                if message_type == "kv_turn":
                    received = unpack_kv(payload)
                    received["kv_seq_len"] = int(header["kv_seq_len"])
                    result = agent.decode_from_kv(
                        received,
                        int(header["max_new_tokens"]),
                    )
                    send_frame(conn, {
                        "type": "kv_result",
                        "request_id": request_id,
                        "answer": result["answer"],
                        "decode_ms": result["decode_ms"],
                        "output_tokens": result["output_tokens"],
                        "payload_bytes": len(payload),
                    })
                    continue

                if message_type == "token_turn":
                    result = generate_token_ids(
                        agent,
                        header["messages"],
                        int(header["max_new_tokens"]),
                        header.get("allowed_token_ids"),
                    )
                    send_frame(conn, {
                        "type": "token_result",
                        "request_id": request_id,
                        **result,
                    })
                    continue

                if message_type == "shutdown":
                    print("[PC2] 收到关闭请求")
                    return

                raise ValueError(f"未知消息类型：{message_type}")
        except (ConnectionError, OSError, ValueError, KeyError) as error:
            print(f"[PC2] 连接或请求失败：{error}；5 秒后重连")
            time.sleep(5)
        finally:
            try:
                conn.close()
            except UnboundLocalError:
                pass


def main():
    parser = argparse.ArgumentParser(description="PC2 Token/KV 智能体服务")
    parser.add_argument("--pc1-host", required=True)
    parser.add_argument("--pc1-port", type=int, default=9002)
    parser.add_argument("--model-path", default=r"E:\llm_project\models\Qwen2.5-0.5B-Instruct")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    agent = load_agent(args.model_path)
    serve(agent, args.pc1_host, args.pc1_port, args.timeout)


if __name__ == "__main__":
    main()
