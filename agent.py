# E:\llm_project\agent.py
"""
LLM 智能体封装。
每个 Agent 持有（或共享）一个模型，负责：
  - 构建对话上下文
  - prefill 得到 KV-Cache 和 first_token
  - 从 KV 复用解码 / 完整生成
"""
import time
import torch
from transformers.cache_utils import DynamicCache

from comm_utils import iter_kv


def select_next_token(logits, allowed_token_ids=None):
    if not allowed_token_ids:
        return logits.argmax(dim=-1, keepdim=True)
    allowed = torch.tensor(allowed_token_ids, dtype=torch.long, device=logits.device)
    allowed_logits = logits.index_select(dim=-1, index=allowed)
    best_index = allowed_logits.argmax(dim=-1, keepdim=True)
    return allowed.index_select(0, best_index.reshape(-1)).reshape(best_index.shape)


class LLMAgent:
    def __init__(self, name, model, tokenizer, role_desc="", device=None):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.role_desc = role_desc
        self.device = device or model.device

    # ---------- 上下文构建 ----------
    def build_messages(self, task, history, round_idx=None, stage_hint=""):
        """把任务 + 历史发言拼成当前 agent 要接续的对话上下文"""
        system = (
            "你是一个多智能体协作系统中的成员，正在和其他智能体讨论同一个任务。"
            "请简洁地给出你的观点或结论，不要重复别人的话。"
        )
        if self.role_desc:
            system += f"\n你的角色：{self.role_desc}"

        transcript = f"任务：{task}\n\n讨论记录：\n"
        if not history:
            transcript += "（暂无发言，请你先开始）\n"
        else:
            for turn in history:
                transcript += f"{turn['agent']}：{turn['content']}\n"

        if round_idx is not None:
            transcript += f"\n（当前是第 {round_idx} 轮讨论）\n"
        if stage_hint:
            transcript += f"本轮要求：{stage_hint}\n"
        transcript += f"\n请 {self.name} 发言："

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": transcript},
        ]

    def encode(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    # ---------- prefill（KV 模式的发送方计算）----------
    def prefill(self, messages, allowed_token_ids=None):
        inputs = self.encode(messages)
        input_len = int(inputs.input_ids.shape[1])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_ms = (time.time() - t0) * 1000

        past = outputs.past_key_values
        first_token = select_next_token(outputs.logits[:, -1, :], allowed_token_ids)

        kv_bytes = 0
        kv_seq_len = 0
        for k, v in iter_kv(past):
            kv_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()
            kv_seq_len = int(k.shape[-2])

        return {
            "past": past,
            "first_token": first_token,
            "input_tokens": input_len,
            "prefill_ms": prefill_ms,
            "kv_bytes": kv_bytes,
            "kv_seq_len": kv_seq_len,
        }

    # ---------- KV 复用解码（KV 模式的接收方计算）----------
    def decode_from_kv(self, pf, max_new_tokens):
        moved = tuple(
            (k.to(self.device), v.to(self.device)) for k, v in iter_kv(pf["past"])
        )
        cache = DynamicCache()
        for layer_idx, (key_states, value_states) in enumerate(moved):
            cache.update(key_states, value_states, layer_idx)


        kv_seq_len = pf["kv_seq_len"]
        first_token = pf["first_token"].to(self.device)
        current_input = first_token
        current_attn = torch.ones(
            (1, kv_seq_len + 1), dtype=torch.long, device=self.device
        )
        current_pos = torch.tensor([[kv_seq_len]], dtype=torch.long, device=self.device)

        eos_id = self.tokenizer.eos_token_id
        im_end_id = self.tokenizer.convert_tokens_to_ids("")

        generated = []
        # first_token 是 prefill 产出的第一个生成 token，不能丢弃
        first_tid = int(first_token.item())
        if first_tid != eos_id and not (im_end_id is not None and first_tid == im_end_id):
            generated.append(first_token)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(max(0, max_new_tokens - 1)):
            with torch.no_grad():
                o = self.model(
                    input_ids=current_input,
                    past_key_values=cache,
                    attention_mask=current_attn,
                    position_ids=current_pos,
                    use_cache=True,
                )
            next_token = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tid = int(next_token.item())
            if tid == eos_id or (im_end_id is not None and tid == im_end_id):
                break
            generated.append(next_token)
            cache = o.past_key_values
            current_input = next_token
            current_attn = torch.cat(
                [current_attn, torch.ones((1, 1), dtype=torch.long, device=self.device)],
                dim=-1,
            )
            current_pos = current_pos + 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decode_ms = (time.time() - t0) * 1000

        answer = (
            self.tokenizer.decode(
                torch.cat(generated, dim=-1)[0], skip_special_tokens=True
            )
            if generated
            else ""
        )
        return {
            "answer": answer.strip(),
            "decode_ms": decode_ms,
            "output_tokens": len(generated),
        }

    # ---------- 完整生成（Token 模式的接收方计算）----------
    def generate_full(self, messages, max_new_tokens):
        inputs = self.encode(messages)
        input_len = int(inputs.input_ids.shape[1])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        compute_ms = (time.time() - t0) * 1000

        answer = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        return {
            "answer": answer.strip(),
            "compute_ms": compute_ms,
            "output_tokens": int(out.shape[1] - input_len),
            "input_tokens": input_len,
        }
