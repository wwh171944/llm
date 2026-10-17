import socket
import time
import uuid

from comm_utils import pack_kv
from distributed_comm import recv_frame, send_frame
from mode_selector import select_mode


class PC1Session:
    def __init__(self, host, port, timeout_s=60):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.server = None
        self.conn = None

    def wait_for_pc2(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        self.server.settimeout(self.timeout_s)
        print(f"[PC1] 等待 PC2 连接：{self.host}:{self.port}")
        self.conn, address = self.server.accept()
        self.conn.settimeout(self.timeout_s)
        print(f"[PC1] PC2 已连接：{address[0]}:{address[1]}")

    def close(self):
        if self.conn:
            try:
                send_frame(self.conn, {"type": "shutdown", "request_id": str(uuid.uuid4())})
            except (ConnectionError, OSError):
                pass
            self.conn.close()
            self.conn = None
        if self.server:
            self.server.close()
            self.server = None

    def _receive_result(self, expected_type, request_id):
        header, payload = recv_frame(self.conn)
        if payload:
            raise ValueError("响应不应包含二进制负载")
        if header.get("type") != expected_type or header.get("request_id") != request_id:
            raise ValueError(f"收到不匹配的响应：{header.get('type')}")
        return header

    def _run_prefilled_kv_turn(
        self, prefill, payload, max_new_tokens, allowed_token_ids=None, started=None
    ):
        started = started or time.time()
        request_id = str(uuid.uuid4())
        send_frame(self.conn, {
            "type": "kv_turn",
            "request_id": request_id,
            "max_new_tokens": max_new_tokens,
            "allowed_token_ids": allowed_token_ids or [],
            "input_tokens": prefill["input_tokens"],
            "kv_seq_len": prefill["kv_seq_len"],
        }, payload)
        response = self._receive_result("kv_result", request_id)
        e2e_ms = (time.time() - started) * 1000
        tx_ms = max(0.0, e2e_ms - prefill["prefill_ms"] - response["decode_ms"])
        return {
            "answer": response["answer"],
            "mode": "kv",
            "input_tokens": prefill["input_tokens"],
            "output_tokens": response["output_tokens"],
            "prefill_ms": prefill["prefill_ms"],
            "decode_ms": response["decode_ms"],
            "payload_bytes": len(payload),
            "tx_ms": tx_ms,
            "e2e_ms": e2e_ms,
        }

    def run_kv_turn(self, sender, messages, max_new_tokens, allowed_token_ids=None):
        started = time.time()
        prefill = sender.prefill(messages, allowed_token_ids)
        payload = pack_kv(prefill["past"], prefill["first_token"])
        return self._run_prefilled_kv_turn(
            prefill, payload, max_new_tokens, allowed_token_ids, started
        )

    def run_adaptive_turn(
        self, sender, messages, max_new_tokens, bandwidth_mbps, allowed_token_ids=None
    ):
        started = time.time()
        prefill = sender.prefill(messages, allowed_token_ids)
        payload = pack_kv(prefill["past"], prefill["first_token"])
        mode, token_estimate_ms, kv_estimate_ms, reason = select_mode(
            prefill["prefill_ms"],
            0.0,
            max_new_tokens,
            len(payload),
            bandwidth_mbps,
        )
        if mode == "kv":
            result = self._run_prefilled_kv_turn(
                prefill, payload, max_new_tokens, allowed_token_ids, started
            )
        else:
            result = self.run_token_turn(
                sender, messages, max_new_tokens, allowed_token_ids
            )
            result["e2e_ms"] += prefill["prefill_ms"]
            result["selection_prefill_ms"] = prefill["prefill_ms"]

        result.update({
            "bandwidth_mbps": bandwidth_mbps,
            "selection_reason": reason,
            "estimated_token_e2e_ms": token_estimate_ms,
            "estimated_kv_e2e_ms": kv_estimate_ms,
            "selection_kv_bytes": len(payload),
        })
        return result

    def run_token_turn(self, receiver, messages, max_new_tokens, allowed_token_ids=None):
        started = time.time()
        request_id = str(uuid.uuid4())
        send_frame(self.conn, {
            "type": "token_turn",
            "request_id": request_id,
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "allowed_token_ids": allowed_token_ids or [],
        })
        response = self._receive_result("token_result", request_id)
        token_ids = response["token_ids"]
        answer = receiver.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        e2e_ms = (time.time() - started) * 1000
        tx_ms = max(0.0, e2e_ms - response["compute_ms"])
        return {
            "answer": answer,
            "mode": "token",
            "input_tokens": response["input_tokens"],
            "output_tokens": response["output_tokens"],
            "prefill_ms": response["compute_ms"],
            "decode_ms": 0.0,
            "payload_bytes": len(str(token_ids).encode("utf-8")),
            "tx_ms": tx_ms,
            "e2e_ms": e2e_ms,
        }
