import json
import socket

MAX_HEADER_BYTES = 1_048_576
MAX_PAYLOAD_BYTES = 2_147_483_648


def recv_exact(conn, length):
    data = bytearray()
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            raise ConnectionError("对端在消息接收完成前断开连接")
        data.extend(packet)
    return bytes(data)


def send_frame(conn, header, payload=b""):
    header_data = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(header_data) > MAX_HEADER_BYTES:
        raise ValueError("消息头过大")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("消息负载过大")
    conn.sendall(len(header_data).to_bytes(4, "big"))
    conn.sendall(header_data)
    conn.sendall(len(payload).to_bytes(8, "big"))
    conn.sendall(payload)


def recv_frame(conn):
    header_len = int.from_bytes(recv_exact(conn, 4), "big")
    if header_len > MAX_HEADER_BYTES:
        raise ValueError("收到过大的消息头")
    header = json.loads(recv_exact(conn, header_len).decode("utf-8"))

    payload_len = int.from_bytes(recv_exact(conn, 8), "big")
    if payload_len > MAX_PAYLOAD_BYTES:
        raise ValueError("收到过大的消息负载")
    payload = recv_exact(conn, payload_len)
    return header, payload


def connect(host, port, timeout_s=30):
    conn = socket.create_connection((host, port), timeout=timeout_s)
    conn.settimeout(timeout_s)
    return conn
