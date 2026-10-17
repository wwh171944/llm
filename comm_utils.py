# E:\llm_project\comm_utils.py
import io
import torch


def iter_kv(past):
    """兼容新旧 transformers 的 KV 迭代器，返回 (k, v) 序列"""
    # 情况 A：DynamicCache，有 layers 属性
    if hasattr(past, "layers"):
        for layer in past.layers:
            yield layer.keys, layer.values
    # 情况 B：DynamicCache，有 key_cache / value_cache
    elif hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        for k, v in zip(past.key_cache, past.value_cache):
            yield k, v
    # 情况 C：老式 tuple(tuple(k, v))
    else:
        for layer in past:
            if hasattr(layer, "keys"):
                yield layer.keys, layer.values
            elif isinstance(layer, (tuple, list)) and len(layer) >= 2:
                yield layer[0], layer[1]
            else:
                raise ValueError(f"未知的 KV 结构: {type(layer)}")


def pack_kv(past, first_token, question=None):
    """将 KV-Cache、第一个 token、问题打包成字节流"""
    legacy = tuple((k.detach().cpu(), v.detach().cpu()) for k, v in iter_kv(past))
    data = {
        "past": legacy,
        "first_token": first_token.detach().cpu(),
        "question": question,
    }
    buffer = io.BytesIO()
    torch.save(data, buffer)
    return buffer.getvalue()


def unpack_kv(data):
    """从字节流还原 KV-Cache 等"""
    buffer = io.BytesIO(data)
    return torch.load(buffer, map_location="cpu")


def recv_exact(conn, length):
    """从 socket 精确接收 length 字节"""
    data = b""
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            break
        data += packet
    return data