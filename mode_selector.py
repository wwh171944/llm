# E:\llm_project\mode_selector.py
"""
通信模式自适应选择器。

决策逻辑：
    t_token = prefill_ms + decode_ms + token_tx_ms
    t_kv    = kv_tx_ms + decode_ms
    选较小的那个。

其中：
    token_tx_ms = n_tokens * bits_per_token / bandwidth_bps * 1000
    kv_tx_ms    = kv_bytes * 8 / bandwidth_bps * 1000
"""

VOCAB_BITS = 16  # 假设每个 token 用 16 bit 传输（词表 65536）


def token_tx_ms(output_tokens, bandwidth_mbps, bits_per_token=VOCAB_BITS):
    """Token 模式传输时延（毫秒）"""
    if bandwidth_mbps is None or bandwidth_mbps <= 0:
        return 0.0
    bandwidth_bps = bandwidth_mbps * 1e6
    return output_tokens * bits_per_token / bandwidth_bps * 1000


def kv_tx_ms(kv_bytes, bandwidth_mbps):
    """KV-Cache 模式传输时延（毫秒）"""
    if bandwidth_mbps is None or bandwidth_mbps <= 0:
        return 0.0
    bandwidth_bps = bandwidth_mbps * 1e6
    return kv_bytes * 8 / bandwidth_bps * 1000


def estimate_token_e2e(prefill_ms, decode_ms, output_tokens, bandwidth_mbps):
    """估算 Token 模式 E2E"""
    return prefill_ms + decode_ms + token_tx_ms(output_tokens, bandwidth_mbps)


def estimate_kv_e2e(decode_ms, kv_bytes, bandwidth_mbps):
    """估算 KV 模式 E2E（接收方跳过 prefill）"""
    return kv_tx_ms(kv_bytes, bandwidth_mbps) + decode_ms


def select_mode(prefill_ms, decode_ms, output_tokens, kv_bytes, bandwidth_mbps):
    """
    返回 (mode, t_token, t_kv, reason)
    mode ∈ {"token", "kv"}
    """
    t_token = estimate_token_e2e(prefill_ms, decode_ms, output_tokens, bandwidth_mbps)
    t_kv = estimate_kv_e2e(decode_ms, kv_bytes, bandwidth_mbps)
    if t_kv < t_token:
        return "kv", t_token, t_kv, f"t_kv({t_kv:.1f}) < t_token({t_token:.1f})"
    else:
        return "token", t_token, t_kv, f"t_token({t_token:.1f}) <= t_kv({t_kv:.1f})"


def find_crossover_bandwidth(prefill_ms, decode_ms, output_tokens, kv_bytes):
    """
    二分查找交叉带宽（Mbps），使 t_token == t_kv。
    如果找不到（低带宽下 KV 永远更优或 Token 永远更优），返回 None。
    """
    lo, hi = 0.1, 10000.0
    for _ in range(80):
        mid = (lo + hi) / 2
        t_token = estimate_token_e2e(prefill_ms, decode_ms, output_tokens, mid)
        t_kv = estimate_kv_e2e(decode_ms, kv_bytes, mid)
        if t_kv < t_token:
            # KV 更优 → 当前带宽偏高 → 交叉点在下方
            hi = mid
        else:
            # Token 更优 → 当前带宽偏低 → 交叉点在上方
            lo = mid
    return round((lo + hi) / 2, 3)