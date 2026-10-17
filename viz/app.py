import glob
import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="LatentMAS 演示", page_icon="🤖", layout="wide")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
KV_COLOR = "#4C78A8"
TOKEN_COLOR = "#F58518"


@st.cache_data(ttl=2)
def load_latest_discussion(log_dir):
    files = glob.glob(os.path.join(log_dir, "discussion_*.json"))
    if not files:
        return None, None
    path = max(files, key=os.path.getmtime)
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file), path


@st.cache_data(ttl=2)
def load_latest_experiment(log_dir):
    files = glob.glob(os.path.join(log_dir, "experiments_*.csv"))
    if not files:
        return None, None
    path = max(files, key=os.path.getmtime)
    return pd.read_csv(path, encoding="utf-8-sig"), path


def parse_maze_render(text):
    lines = [line for line in text.strip().split("\n") if line.strip()]
    return [line.split()[1:] for line in lines[1:] if len(line.split()) >= 2]


def draw_maze(grid, title):
    color_map = {
        "#": "#2F3437", "*": "#54A24B", "G": "#E4B84B",
        "S": KV_COLOR, "o": "#A9C7E4", ".": "#FFFFFF",
    }
    figure = go.Figure()
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    for row, cells in enumerate(grid):
        for column, cell in enumerate(cells):
            y = rows - row - 1
            figure.add_shape(
                type="rect", x0=column, x1=column + 1, y0=y, y1=y + 1,
                fillcolor=color_map.get(cell, "#FFFFFF"), line=dict(color="#BDBDBD"),
            )
            figure.add_annotation(
                x=column + 0.5, y=y + 0.5, text="" if cell == "." else cell,
                showarrow=False,
                font=dict(size=18, color="white" if cell == "#" else "black"),
            )
    figure.update_layout(
        title=title, height=380, showlegend=False, plot_bgcolor="white",
        xaxis=dict(range=[0, cols], visible=False),
        yaxis=dict(range=[0, rows], visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return figure


def mode_color(mode):
    return KV_COLOR if str(mode).lower() == "kv" else TOKEN_COLOR


st.title("🤖 LatentMAS 多智能体通信演示")
st.caption("蓝色：KV-Cache 通信 ｜ 橙色：Token 通信")

st.header("1. 双机多智能体迷宫讨论")
discussion, discussion_path = load_latest_discussion(LOG_DIR)
if discussion is None:
    st.info("暂无讨论数据。请先运行 discussion.py 和 pc2_agent.py。")
else:
    results = discussion.get("results", [])
    turns = [turn for result in results for turn in result.get("turns", [])]
    protocol = {
        direction: mode.upper()
        for result in results
        for direction, mode in result.get("protocol", {}).items()
    }
    st.caption(
        f"数据来源：{os.path.basename(discussion_path)} ｜ "
        + " ｜ ".join(f"{direction}: {mode}" for direction, mode in protocol.items())
    )

    total_payload = sum(turn.get("payload_bytes", 0) for turn in turns)
    average_e2e = sum(turn.get("e2e_ms", 0) for turn in turns) / len(turns) if turns else 0
    kv_count = sum(turn.get("mode") == "kv" for turn in turns)
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("任务成功率", f"{discussion.get('success_rate', 0) * 100:.0f}%")
    metric2.metric("通信轮次", len(turns))
    metric3.metric("总通信负载", f"{total_payload / 1024:.1f} KB")
    metric4.metric("平均 E2E 时延", f"{average_e2e:.0f} ms", f"KV {kv_count}/{len(turns)} 轮" if turns else None)

    if turns:
        left, right = st.columns([1, 1.2])
        with left:
            selected_round = st.slider("查看讨论轮次", 1, len(turns), len(turns))
            selected = turns[selected_round - 1]
            grid = parse_maze_render(selected.get("maze", ""))
            if grid:
                title = (
                    f"第 {selected_round} 轮 ｜ {selected.get('direction')} ｜ "
                    f"{str(selected.get('mode', '')).upper()} ｜ "
                    f"负载 {selected.get('payload_bytes', 0):,} B"
                )
                st.plotly_chart(draw_maze(grid, title), use_container_width=True)
        with right:
            table = pd.DataFrame([{
                "轮次": turn.get("round"),
                "方向": turn.get("direction"),
                "协议": str(turn.get("mode", "")).upper(),
                "模型输出": turn.get("content", ""),
                "实际移动": " ".join(turn.get("moves_applied", [])) or "-",
                "负载 (B)": turn.get("payload_bytes", 0),
                "E2E (ms)": round(turn.get("e2e_ms", 0), 1),
            } for turn in turns])
            st.subheader("讨论与通信记录")
            st.dataframe(table, hide_index=True, use_container_width=True)

        chart_left, chart_right = st.columns(2)
        rounds = [f"R{turn.get('round')}" for turn in turns]
        colors = [mode_color(turn.get("mode")) for turn in turns]
        with chart_left:
            latency = go.Figure()
            latency.add_trace(go.Bar(x=rounds, y=[turn.get("prefill_ms", 0) for turn in turns], name="Prefill", marker_color=KV_COLOR))
            latency.add_trace(go.Bar(x=rounds, y=[turn.get("tx_ms", 0) for turn in turns], name="传输", marker_color=TOKEN_COLOR))
            latency.add_trace(go.Bar(x=rounds, y=[turn.get("decode_ms", 0) for turn in turns], name="Decode", marker_color="#54A24B"))
            latency.update_layout(title="各轮时延分解", barmode="stack", yaxis_title="毫秒 (ms)", height=320)
            st.plotly_chart(latency, use_container_width=True)
        with chart_right:
            payload = go.Figure(go.Bar(
                x=rounds, y=[turn.get("payload_bytes", 0) for turn in turns], marker_color=colors,
                text=[f"{turn.get('payload_bytes', 0) / 1024:.1f} KB" for turn in turns], textposition="auto",
            ))
            payload.update_layout(title="各轮通信负载", yaxis_title="字节 (B)", height=320)
            st.plotly_chart(payload, use_container_width=True)

st.divider()
st.header("2. 完整迷宫通关率与 Token / KV-Cache 性能对比")


@st.fragment(run_every=5)
def render_experiment():
    experiment, experiment_path = load_latest_experiment(LOG_DIR)
    if experiment is None or experiment.empty:
        st.info("暂无完整迷宫实验数据。请先运行 batch_experiment.py。")
        return

    st.caption(
        f"数据来源：{os.path.basename(experiment_path)} ｜ {len(experiment)} 条完整迷宫记录 ｜ 每 5 秒自动刷新"
    )
    aggregate = experiment.groupby("bandwidth_mbps").agg(
        token_e2e=("token_e2e_ms", "mean"),
        kv_e2e=("kv_e2e_ms", "mean"),
        adaptive_e2e=("adaptive_e2e_ms", "mean"),
    ).reset_index()

    comparison = go.Figure()
    comparison.add_trace(go.Scatter(x=aggregate["bandwidth_mbps"], y=aggregate["token_e2e"], mode="lines+markers", name="Token", line=dict(color=TOKEN_COLOR)))
    comparison.add_trace(go.Scatter(x=aggregate["bandwidth_mbps"], y=aggregate["kv_e2e"], mode="lines+markers", name="KV-Cache", line=dict(color=KV_COLOR)))
    comparison.add_trace(go.Scatter(x=aggregate["bandwidth_mbps"], y=aggregate["adaptive_e2e"], mode="lines+markers", name="自适应", line=dict(color="#54A24B", dash="dot")))
    comparison.update_layout(title="完成一张迷宫的端到端时延", xaxis_title="带宽 (Mbps)", yaxis_title="E2E 时延 (ms)", xaxis_type="log", height=380)
    st.plotly_chart(comparison, use_container_width=True)

    left, right = st.columns(2)
    with left:
        success = experiment.groupby("bandwidth_mbps")[["token_success", "kv_success", "adaptive_success"]].mean().reset_index()
        success_long = success.melt(id_vars="bandwidth_mbps", var_name="策略", value_name="通关率")
        success_chart = px.bar(
            success_long, x="bandwidth_mbps", y="通关率", color="策略", barmode="group", title="完整迷宫通关率",
            labels={"bandwidth_mbps": "带宽 (Mbps)"}, color_discrete_sequence=[TOKEN_COLOR, KV_COLOR, "#54A24B"],
        )
        success_chart.update_layout(height=340, yaxis=dict(range=[0, 1]))
        st.plotly_chart(success_chart, use_container_width=True)
    with right:
        quality = experiment.groupby("difficulty").agg(
            平均步数=("completed_steps", "mean"),
            最短路径步数=("optimal_steps", "mean"),
            路径效率=("path_efficiency", "mean"),
            模型有效移动率=("model_progress_rate", "mean"),
        ).reset_index()
        st.dataframe(quality, use_container_width=True, hide_index=True)

    if {"adaptive_kv_turns", "adaptive_token_turns"}.issubset(experiment.columns):
        modes = experiment.groupby("bandwidth_mbps").agg(
            KV=("adaptive_kv_turns", "sum"),
            Token=("adaptive_token_turns", "sum"),
        ).reset_index().melt(id_vars="bandwidth_mbps", var_name="协议", value_name="决策轮数")
        selection = px.bar(
            modes, x="bandwidth_mbps", y="决策轮数", color="协议", barmode="stack",
            color_discrete_map={"Token": TOKEN_COLOR, "KV": KV_COLOR}, title="完整路径中的自适应协议选择",
        )
        selection.update_layout(height=340)
        st.plotly_chart(selection, use_container_width=True)


render_experiment()
st.caption("LatentMAS · 双机智能体 Token / KV-Cache 通信系统")
