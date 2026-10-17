import json
import urllib.error
import urllib.request

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

KV_COLOR = "#4C78A8"
TOKEN_COLOR = "#F58518"
MAZE_SIZES = [5, 7, 9]

st.set_page_config(page_title="自定义迷宫实验", page_icon="🧭", layout="wide")


def solve_maze(api_url, maze, max_rounds, bandwidth_mbps):
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/solve-maze",
        data=json.dumps({
            "maze": maze,
            "max_rounds": max_rounds,
            "bandwidth_mbps": bandwidth_mbps,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_maze_render(text):
    lines = [line for line in text.strip().split("\n") if line.strip()]
    return [line.split()[1:] for line in lines[1:] if len(line.split()) >= 2]


def draw_maze(grid, title):
    colors = {"#": "#2F3437", "*": "#54A24B", "G": "#E4B84B", "S": KV_COLOR, "o": "#A9C7E4", ".": "#FFFFFF"}
    figure = go.Figure()
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    for row, cells in enumerate(grid):
        for column, cell in enumerate(cells):
            y = rows - row - 1
            figure.add_shape(type="rect", x0=column, x1=column + 1, y0=y, y1=y + 1, fillcolor=colors.get(cell, "#FFFFFF"), line=dict(color="#BDBDBD"))
            figure.add_annotation(x=column + 0.5, y=y + 0.5, text="" if cell == "." else cell, showarrow=False, font=dict(size=18, color="white" if cell == "#" else "black"))
    figure.update_layout(title=title, height=380, showlegend=False, plot_bgcolor="white", xaxis=dict(range=[0, cols], visible=False), yaxis=dict(range=[0, rows], visible=False, scaleanchor="x", scaleratio=1), margin=dict(l=20, r=20, t=45, b=20))
    return figure


st.title("🧭 自定义迷宫双智能体协作实验")
st.caption("选择地图尺寸后，在表格中填入入口 S、出口 G、障碍物 # 与通路 .；每轮会根据链路条件自适应选择 Token 或 KV-Cache。")

with st.sidebar:
    st.subheader("实验连接")
    api_url = st.text_input("PC1 协调服务地址", "http://127.0.0.1:9010")
    max_rounds = st.slider("最大决策轮数", 1, 100, 32)
    bandwidth_mbps = st.select_slider(
        "模拟链路带宽 (Mbps)",
        options=[1, 10, 100, 1000],
        value=1000,
    )
    st.caption("每轮会用实际 KV 负载和设定带宽，自动选择 Token 或 KV-Cache。")

maze_size = st.segmented_control("地图尺寸", MAZE_SIZES, default=5)
grid_key = f"maze_grid_{maze_size}"
if grid_key not in st.session_state:
    st.session_state[grid_key] = pd.DataFrame(
        ".",
        index=range(maze_size),
        columns=[str(column + 1) for column in range(maze_size)],
    )

st.caption("每格只能选择：`.` 通路、`#` 障碍、`S` 入口、`G` 出口。必须且只能设置一个 S 和一个 G。")
column_config = {
    column: st.column_config.SelectboxColumn(
        column,
        options=[".", "#", "S", "G"],
        required=True,
    )
    for column in st.session_state[grid_key].columns
}
edited_grid = st.data_editor(
    st.session_state[grid_key],
    column_config=column_config,
    hide_index=True,
    num_rows="fixed",
    use_container_width=True,
    key=f"maze_editor_{maze_size}",
)

if st.button("发送地图并开始求解", type="primary"):
    values = edited_grid.astype(str).to_numpy().flatten().tolist()
    if values.count("S") != 1 or values.count("G") != 1:
        st.error("请在地图中设置且只设置一个入口 S 与一个出口 G。")
    else:
        maze_input = "\n".join(" ".join(row) for row in edited_grid.astype(str).to_numpy())
        with st.spinner("两个智能体正在协作寻找出口..."):
            try:
                st.session_state["maze_result"] = solve_maze(
                    api_url, maze_input, max_rounds, bandwidth_mbps
                )
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                st.error(f"迷宫服务返回错误：{detail}")
            except (urllib.error.URLError, TimeoutError) as error:
                st.error(f"无法连接 PC1 协调服务：{error}")

result = st.session_state.get("maze_result")
if result:
    turns = result.get("turns", [])
    total_payload = sum(turn.get("payload_bytes", 0) for turn in turns)
    metric1, metric2, metric3, metric4 = st.columns(4)
    kv_turns = sum(turn.get("mode") == "kv" for turn in turns)
    metric1.metric("求解结果", "✅ 到达出口" if result.get("success") else "❌ 未到达")
    metric2.metric("决策轮次", result.get("num_rounds", 0))
    metric3.metric("协议选择", f"KV {kv_turns} / Token {len(turns) - kv_turns}")
    metric4.metric("总时延", f"{result.get('total_e2e_ms', 0):.0f} ms")
    st.caption(f"自适应带宽：{result.get('bandwidth_mbps', bandwidth_mbps):g} Mbps")

    if turns:
        left, right = st.columns([1, 1.2])
        with left:
            selected_round = st.slider("查看求解进度", 1, len(turns), len(turns))
            selected = turns[selected_round - 1]
            grid = parse_maze_render(selected.get("maze", ""))
            if grid:
                title = f"第 {selected_round} 轮 ｜ {selected.get('direction')} ｜ {str(selected.get('mode', '')).upper()} ｜ 执行：{selected.get('applied_move')}"
                st.plotly_chart(draw_maze(grid, title), use_container_width=True)
        with right:
            table = pd.DataFrame([{
                "轮次": turn.get("round"),
                "方向": turn.get("direction"),
                "审核协议": str(turn.get("mode", "")).upper(),
                "提议方向": turn.get("proposal_move") or "无",
                "审核方向": turn.get("proposed_move") or "无",
                "执行移动": turn.get("applied_move") or "未执行",
                "来源": (
                    "模型回溯" if turn.get("move_source") == "model_backtrack"
                    else "模型" if turn.get("move_source") == "model"
                    else "模型输出无效"
                ),
                "预估 Token (ms)": turn.get("estimated_token_e2e_ms", 0),
                "预估 KV (ms)": turn.get("estimated_kv_e2e_ms", 0),
                "选择原因": turn.get("selection_reason", "-"),
                "负载 (B)": turn.get("payload_bytes", 0),
                "E2E (ms)": turn.get("e2e_ms", 0),
            } for turn in turns])
            st.subheader("协作决策记录")
            st.dataframe(table, hide_index=True, use_container_width=True)

        with st.expander("查看每轮智能体原始输出"):
            for turn in turns:
                mode = str(turn.get("mode", "")).upper()
                agent = turn.get("receiver", "智能体")
                st.markdown(
                    f"**第 {turn.get('round')} 轮 · {agent} · {mode}** ｜ "
                    f"{turn.get('direction', '-')}"
                )
                st.code(
                    f"提议（{turn.get('proposer', '-')}）：{turn.get('proposal_content', '（无输出）')}\n"
                    f"审核（{turn.get('reviewer', agent)}）：{turn.get('content', '（无输出）')}",
                    language=None,
                )

        rounds = [f"R{turn.get('round')}" for turn in turns]
        latency = go.Figure()
        latency.add_trace(go.Bar(x=rounds, y=[turn.get("prefill_ms", 0) for turn in turns], name="Prefill", marker_color=KV_COLOR))
        latency.add_trace(go.Bar(x=rounds, y=[turn.get("tx_ms", 0) for turn in turns], name="传输", marker_color=TOKEN_COLOR))
        latency.add_trace(go.Bar(x=rounds, y=[turn.get("decode_ms", 0) for turn in turns], name="Decode", marker_color="#54A24B"))
        latency.update_layout(title="各轮通信时延分解", barmode="stack", yaxis_title="毫秒 (ms)")
        st.plotly_chart(latency, use_container_width=True)
