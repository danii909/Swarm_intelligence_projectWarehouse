from __future__ import annotations

import numpy as np

from ui.constants import (
    AGENT_PALETTE,
    DEFAULT_RADIUS,
    INITIAL_BATTERY,
    STRATEGIES,
)


def build_delivery_curve(history, max_ticks: int) -> list[float]:
    max_ticks = max(1, int(max_ticks))
    curve = np.zeros(max_ticks, dtype=float)
    for snap in history or []:
        tick = int(getattr(snap, "tick", 0))
        delivered = float(getattr(snap, "delivered", 0.0))
        if tick <= 0:
            continue
        idx = min(tick, max_ticks) - 1
        curve[idx] = max(curve[idx], delivered)
    curve = np.maximum.accumulate(curve)
    return curve.tolist()


def style_dark_chart(ax):
    ax.set_facecolor("#0e1117")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.grid(axis="y", color="#2a2a2a", linewidth=0.7, alpha=0.6)


def build_agents(agent_configs: list, num_agents: int):
    from src.agents.agent import Agent
    from src.agents.strategies.frontier import FrontierStrategy
    from src.agents.strategies.greedy import GreedyStrategy
    from src.agents.strategies.sector import SectorStrategy
    from src.agents.strategies.Repulsion import RepulsionStrategy
    from src.agents.strategies.random_walk import RandomWalkStrategy
    from src.agents.strategies.ant_colony_lite import AntColonyLiteStrategy

    factories = {
        0: lambda: FrontierStrategy(),
        1: lambda: GreedyStrategy(),
        2: lambda: SectorStrategy(num_agents=num_agents),
        3: lambda: RepulsionStrategy(),
        4: lambda: RandomWalkStrategy(),
        5: lambda: AntColonyLiteStrategy(),
    }

    agents = []
    for cfg in agent_configs:
        strategy = factories[cfg["strategy_id"]]()
        agents.append(Agent(
            agent_id=cfg["agent_id"],
            strategy=strategy,
            visibility_radius=cfg["radius"],
            comm_radius=cfg.get("comm_radius", 2),
        ))
    return agents


def agent_label_rgb(color_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    luminance = (0.299 * color_rgb[0]) + (0.587 * color_rgb[1]) + (0.114 * color_rgb[2])
    return (255, 255, 255) if luminance < 140 else (0, 0, 0)


def agent_label_hex(color_hex: str) -> str:
    color_hex = color_hex.lstrip("#")
    rgb = tuple(int(color_hex[i:i + 2], 16) for i in (0, 2, 4))
    label_rgb = agent_label_rgb(rgb)
    return "#FFFFFF" if label_rgb == (255, 255, 255) else "#000000"


def render_battery_html(agents, agent_configs) -> str:
    strat_by_id = {cfg["agent_id"]: STRATEGIES[cfg["strategy_id"]][0] for cfg in agent_configs}
    html_parts = []

    for i, agent in enumerate(agents):
        pct = max(agent.battery / INITIAL_BATTERY, 0.0)
        pct_display = pct * 100
        if pct > 0.5:
            bar_color = "#55A868"
        elif pct > 0.2:
            bar_color = "#DD8452"
        else:
            bar_color = "#C44E52"

        agent_color = AGENT_PALETTE[i % len(AGENT_PALETTE)]
        agent_text_color = agent_label_hex(agent_color)
        state_label = agent.state.name.replace("_", " ").title()
        strat_name = strat_by_id.get(agent.id, "?")
        radii_label = f"(v{agent.visibility_radius}, c{agent.comm_radius})"

        html_parts.append(
            f"<div style='margin-bottom:8px; padding:8px; border:1px solid #2e3342; border-radius:8px; background:#161b28;'>"
            f"  <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>"
            f"    <span style='display:inline-flex; align-items:center; gap:6px;'>"
            f"      <span style='background:{agent_color}; color:{agent_text_color}; border:1px solid #000; border-radius:999px; padding:2px 8px; font-size:0.78em; font-weight:bold;'>A{agent.id + 1}</span>"
            f"      <span style='color:#cfd3d8; font-size:0.78em;'>{strat_name}</span>"
            f"      <span style='color:#8ea2b3; font-size:0.72em;'>{radii_label}</span>"
            f"    </span>"
            f"    <span style='color:#999; font-size:0.7em;'>🔋{agent.battery}/{INITIAL_BATTERY} — {state_label}</span>"
            f"  </div>"
            f"  <div style='background:#333; border-radius:4px; height:10px; overflow:hidden; margin-top:2px;'>"
            f"    <div style='background:{bar_color}; width:{pct_display:.1f}%; height:100%; border-radius:4px; transition: width 0.3s ease;'></div>"
            f"  </div>"
            f"</div>"
        )

    html_parts.append("</div>")
    return "".join(html_parts)


def render_status_card_html(title: str, value: str, accent: str) -> str:
    return (
        "<div style='"
        "padding:10px; border:1px solid #2e3342; border-radius:8px; "
        "background:#161b28; min-height:50px; display:flex; flex-direction:column; "
        "justify-content:center; margin-bottom:12px;'>"
        f"<div style='font-size:0.78em; color:#8ea2b3; margin-bottom:4px;'>{title}</div>"
        f"<div style='font-size:1.35em; font-weight:700; color:{accent}; line-height:1.1;'>{value}</div>"
        "</div>"
    )


def build_agents_table_html(rows):
    hdr_cols = list(rows[0].keys()) if rows else []
    parts = ["<table style='width:100%; border-collapse:collapse; font-family:system-ui; font-size:0.92rem;'>"]
    parts.append("<thead><tr>")
    for col_name in hdr_cols:
        parts.append(f"<th style='text-align:left; padding:6px 8px; color:#9ea6af; font-weight:600; border-bottom:1px solid #2e3342;'>{col_name}</th>")
    parts.append("</tr></thead><tbody>")

    for row in rows:
        parts.append("<tr style='border-bottom:1px solid #20242b;'>")
        for col_name in hdr_cols:
            val = row.get(col_name, "")
            if col_name == "Agente":
                try:
                    idx = int(str(val).lstrip("A")) - 1
                except Exception:
                    idx = 0
                agent_color = AGENT_PALETTE[idx % len(AGENT_PALETTE)]
                text_color = agent_label_hex(agent_color)
                parts.append(
                    f"<td style='padding:8px; vertical-align:middle;'>"
                    f"<span style='display:inline-block; background:{agent_color}; color:{text_color}; border:1px solid rgba(0,0,0,0.25); padding:4px 10px; border-radius:999px; font-weight:700;'>{val}</span></td>"
                )
            else:
                parts.append(f"<td style='padding:8px; vertical-align:middle; color:#cfd3d8;'>{val}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def apply_pending_preset_if_any(session_state, strategy_name_options):
    if "_apply_preset" not in session_state:
        return
    preset = session_state.pop("_apply_preset")
    session_state["num_agents_val"] = preset["num_agents"]
    for agent in preset["agents"]:
        agent_id = agent["agent_id"]
        session_state[f"strat_{agent_id}"] = strategy_name_options[agent["strategy_id"]]
        session_state[f"radius_{agent_id}"] = agent["radius"]
        session_state[f"comm_{agent_id}"] = agent.get("comm_radius", 2)


def default_radius_for_strategy(strategy_id: int) -> int:
    return DEFAULT_RADIUS[strategy_id]
