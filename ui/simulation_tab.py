from __future__ import annotations

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ui.constants import (
    DEFAULT_FRAME_DELAY,
    PACKAGE_ICON_DEFAULT_PATH,
    STRATEGIES,
    strategy_ids,
    strategy_name_options,
)
from ui.helpers import (
    apply_pending_preset_if_any,
    build_agents,
    build_agents_table_html,
    build_delivery_curve,
    default_radius_for_strategy,
    render_battery_html,
    render_status_card_html,
    style_dark_chart,
)
from ui.rendering import load_pygame_icon, load_uploaded_pygame_icon, render_frame


SIMULATION_SLIDER_CSS = """
<style>
div[data-testid="stSlider"] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
.element-container:has(div[data-testid="stSlider"]) {
    margin-top: 0rem;
    margin-bottom: -0.4rem;
}
</style>
"""


def _render_agent_config_panel():
    # Carica automaticamente il preset default se è la prima volta
    if "_default_preset_loaded" not in st.session_state:
        try:
            with open("assets/default_preset.json", "r") as f:
                st.session_state["_apply_preset"] = json.load(f)
                st.session_state["_default_preset_loaded"] = True
        except Exception as e:
            st.warning(f"Impossibile caricare il preset default: {e}")
            st.session_state["_default_preset_loaded"] = True
    
    apply_pending_preset_if_any(st.session_state, strategy_name_options)

    run_clicked = st.button("▶ Avvia", type="primary", width="stretch")
    st.markdown("##### Configurazione")

    with st.expander("Parametri generali", expanded=False):
        num_agents = st.slider("N. agenti", min_value=1, max_value=10, value=5, key="num_agents_val")
        max_ticks = st.slider("Tick massimi", min_value=100, max_value=750, value=500, step=50)
        update_every = st.slider("Aggiorna ogni N tick", min_value=1, max_value=20, value=1)
        frame_delay = st.slider(
            "Delay tra frame (secondi)",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_FRAME_DELAY,
            step=0.01,
            help="Piu alto = simulazione visivamente piu lenta.",
        )
        agent_icon_upload = st.file_uploader(
            "Immagine agente personalizzata",
            type=["png", "jpg", "jpeg", "webp"],
            key="agent_icon_upload",
        )
        package_icon_upload = st.file_uploader(
            "Immagine pacco personalizzata",
            type=["png", "jpg", "jpeg", "webp"],
            key="package_icon_upload",
        )

    uploaded_preset = st.file_uploader("📂 Carica preset config. agenti", type=["json"], key="upload_preset_col")
    if uploaded_preset is not None:
        try:
            st.session_state["_apply_preset"] = json.loads(uploaded_preset.read().decode("utf-8"))
        except Exception as exc:
            st.error(f"Errore lettura preset: {exc}")

    agent_configs = []
    for agent_id in range(num_agents):
        default_sid = agent_id % len(STRATEGIES)
        default_r = default_radius_for_strategy(default_sid)
        with st.expander(f"Agente {agent_id + 1}", expanded=False):
            chosen_name = st.selectbox(
                "Strategia",
                options=strategy_name_options,
                index=default_sid,
                key=f"strat_{agent_id}",
            )
            chosen_sid = strategy_ids[strategy_name_options.index(chosen_name)]
            radius = st.slider(" Raggio Visione", min_value=1, max_value=3, value=default_r, key=f"radius_{agent_id}")
            comm_r = st.slider(" Raggio Comunicazione", min_value=1, max_value=2, value=2, key=f"comm_{agent_id}")

        agent_configs.append({
            "agent_id": agent_id,
            "strategy_id": chosen_sid,
            "radius": radius,
            "comm_radius": comm_r,
        })

    return run_clicked, agent_configs, max_ticks, update_every, frame_delay, agent_icon_upload, package_icon_upload


def _render_preview(instance_path, agent_configs, max_ticks, frame_ph, tick_ph, stats_ph, prog_ph, battery_ph, agent_icon_upload, package_icon_upload):
    preview_agents = []
    total_objects_preview = "?"

    if os.path.isfile(instance_path):
        try:
            from src.environment.environment import Environment
            env_preview = Environment.from_json(instance_path)
            preview_agents = build_agents(agent_configs, len(agent_configs))
            total_objects_preview = str(env_preview.total_objects)

            preview_agent_icon = load_uploaded_pygame_icon(agent_icon_upload)
            preview_package_icon = load_uploaded_pygame_icon(package_icon_upload) or load_pygame_icon(PACKAGE_ICON_DEFAULT_PATH)

            preview_png = render_frame(
                0,
                preview_agents,
                env_preview,
                show_fog=False,
                agent_icon_img=preview_agent_icon,
                package_icon_img=preview_package_icon,
            )
            frame_ph.image(preview_png, width="stretch")
        except Exception as preview_exc:
            frame_ph.warning(f"Anteprima non disponibile: {preview_exc}")
    else:
        frame_ph.info("Seleziona un file istanza valido per vedere l'anteprima.")

    tick_ph.markdown(render_status_card_html("Tick", "0", "#4C72B0"), unsafe_allow_html=True)
    stats_ph.markdown(render_status_card_html("Consegnati", f"0 / {total_objects_preview}", "#55A868"), unsafe_allow_html=True)
    prog_ph.progress(0.0, text=f"Tick 0/{max_ticks}")

    if preview_agents:
        battery_ph.markdown(render_battery_html(preview_agents, agent_configs), unsafe_allow_html=True)
    else:
        battery_ph.info("Configura gli agenti e premi Avvia.")


def _run_simulation(instance_path, seed, agent_configs, max_ticks, update_every, frame_delay, frame_ph, tick_ph, stats_ph, prog_ph, battery_ph, agent_icon_upload, package_icon_upload):
    if not os.path.isfile(instance_path):
        st.error(f"File istanza non trovato: `{instance_path}`")
        st.stop()

    from src.environment.environment import Environment
    from src.simulation.simulator import Simulator

    env_obj = Environment.from_json(instance_path)
    built_agents = build_agents(agent_configs, len(agent_configs))
    sim = Simulator(
        env=env_obj,
        agents=built_agents,
        max_ticks=max_ticks,
        seed=seed if seed >= 0 else None,
        verbose=False,
        log_every=1,
    )

    agent_icon_img = load_uploaded_pygame_icon(agent_icon_upload)
    package_icon_img = load_uploaded_pygame_icon(package_icon_upload) or load_pygame_icon(PACKAGE_ICON_DEFAULT_PATH)

    t0 = time.perf_counter()
    try:
        for tick, cur_agents, cur_env in sim.step_gen():
            if tick % update_every == 0 or cur_env.all_delivered:
                png = render_frame(
                    tick,
                    cur_agents,
                    cur_env,
                    show_fog=True,
                    agent_icon_img=agent_icon_img,
                    package_icon_img=package_icon_img,
                )
                frame_ph.image(png, width="stretch")
                prog_ph.progress(min(tick / max_ticks, 1.0), text=f"Tick {tick}/{max_ticks}")
                tick_ph.markdown(render_status_card_html("Tick", str(tick), "#4C72B0"), unsafe_allow_html=True)
                stats_ph.markdown(
                    render_status_card_html("Consegnati", f"{cur_env.delivered} / {cur_env.total_objects}", "#55A868"),
                    unsafe_allow_html=True,
                )
                battery_ph.markdown(render_battery_html(cur_agents, agent_configs), unsafe_allow_html=True)
                if frame_delay > 0:
                    time.sleep(frame_delay)
    except Exception as exc:
        st.error(f"Errore durante la simulazione: {exc}")
        st.exception(exc)
        st.stop()

    elapsed = time.perf_counter() - t0
    summary = sim.metrics.summary()
    delivery_curve = build_delivery_curve(sim.metrics.history, max_ticks)
    st.session_state.setdefault("history_runs", []).append({"summary": summary, "configs": list(agent_configs), "delivery_curve": delivery_curve})
    st.session_state["last_delivery_curve"] = delivery_curve
    st.session_state["last_max_ticks"] = max_ticks
    return elapsed, summary


def _render_simulation_results(summary, elapsed, agent_configs):
    st.divider()
    st.subheader("📊 Risultati")

    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Oggetti consegnati", f"{summary['objects_delivered']} / {summary['total_objects']}")
    r2.metric("Completamento", f"{summary['delivery_rate'] * 100:.1f}%")
    r3.metric("Tick totali", summary["total_ticks"])
    r4.metric("Energia media", f"{summary['average_energy_consumed']:.1f}")
    r5.metric("Tempo CPU", f"{elapsed:.2f}s")

    st.subheader("Dettaglio agenti")
    steps_list = summary.get("agent_steps", [])
    batteries_list = summary.get("agent_final_batteries", [])

    agent_rows = []
    for cfg in agent_configs:
        i = cfg["agent_id"]
        agent_rows.append({
            "Agente": f"A{i + 1}",
            "Strategia": STRATEGIES[cfg["strategy_id"]][0],
            "Raggio vis.": cfg["radius"],
            "Raggio com.": cfg["comm_radius"],
            "Passi": steps_list[i] if i < len(steps_list) else "—",
            "Batteria finale": batteries_list[i] if i < len(batteries_list) else "—",
        })

    st.markdown(build_agents_table_html(agent_rows), unsafe_allow_html=True)

    preset_data = {"name": "preset", "num_agents": len(agent_configs), "agents": agent_configs}
    st.download_button(
        "⬇ Scarica preset corrente",
        data=json.dumps(preset_data, indent=4),
        file_name="preset.json",
        mime="application/json",
    )

    st.divider()
    st.markdown("#### 📈 Curva cumulativa deliveries")
    delivery_curve = st.session_state.get("last_delivery_curve")
    max_ticks = st.session_state.get("last_max_ticks", summary["total_ticks"])
    
    if delivery_curve is not None:
        fig_curve, ax_curve = plt.subplots(figsize=(10, 5), facecolor="#0e1117")
        style_dark_chart(ax_curve)
        x_ticks = np.arange(1, max_ticks + 1)
        curve_arr = np.array(delivery_curve, dtype=float)
        if curve_arr.ndim > 1:
            curve_arr = curve_arr[0]
        ax_curve.plot(x_ticks, curve_arr, linewidth=2.5, color="#00D9FF", label="Deliveries cumulative")
        ax_curve.fill_between(x_ticks, curve_arr, alpha=0.2, color="#00D9FF")
        ax_curve.set_title("", color="white")
        ax_curve.set_xlabel("Tick", color="white")
        ax_curve.set_ylabel("Oggetti consegnati", color="white")
        ax_curve.set_ylim(bottom=0)
        ax_curve.grid(axis="y", color="#2a2a2a", linewidth=0.7, alpha=0.6)

        try:
            ticks = np.concatenate((np.array([1], dtype=int), np.arange(50, max_ticks + 1, 50, dtype=int)))
            ticks = ticks[ticks <= max_ticks]
            ax_curve.set_xticks(ticks)
        except Exception:
            pass
        fig_curve.tight_layout()
        st.pyplot(fig_curve)
        plt.close(fig_curve)
    else:
        st.info("Curva deliveries non disponibile.")

    st.divider()
    
def _render_history_runs():
    history_runs = st.session_state.get("history_runs", [])
    if len(history_runs) <= 1:
        return

    st.divider()
    st.subheader("🕑 Storico simulazioni")

    hist_rows = []
    for idx, run in enumerate(history_runs):
        s = run["summary"]
        cfg_str = ", ".join(
            f"A{c['agent_id'] + 1}:{STRATEGIES[c['strategy_id']][0]}(v{c['radius']} c{c.get('comm_radius', 2)})"
            for c in run["configs"]
        )
        hist_rows.append({
            "Run": idx + 1,
            "Consegnati": f"{s['objects_delivered']}/{s['total_objects']}",
            "Completamento": f"{s['delivery_rate'] * 100:.1f}%",
            "Tick": s["total_ticks"],
            "Energia media": s["average_energy_consumed"],
            "Configurazione": cfg_str,
        })

    st.dataframe(pd.DataFrame(hist_rows), width="stretch", hide_index=True)
    if st.button("🗑 Azzera storico"):
        st.session_state["history_runs"] = []
        st.rerun()


def render_simulation_tab(instance_path: str, seed: int):
    st.markdown(SIMULATION_SLIDER_CSS, unsafe_allow_html=True)
    col_cfg, col_sim, col_status = st.columns([2, 4, 2])

    with col_cfg:
        sim_cfg = _render_agent_config_panel()
    run_clicked, agent_configs, max_ticks, update_every, frame_delay, agent_icon_upload, package_icon_upload = sim_cfg

    with col_sim:
        frame_ph = st.empty()

    with col_status:
        top_left, top_right = st.columns(2)
        with top_left:
            tick_ph = st.empty()
        with top_right:
            stats_ph = st.empty()
        prog_ph = st.empty()
        battery_ph = st.empty()

    if not run_clicked:
        _render_preview(
            instance_path,
            agent_configs,
            max_ticks,
            frame_ph,
            tick_ph,
            stats_ph,
            prog_ph,
            battery_ph,
            agent_icon_upload,
            package_icon_upload,
        )
    else:
        elapsed, summary = _run_simulation(
            instance_path,
            seed,
            agent_configs,
            max_ticks,
            update_every,
            frame_delay,
            frame_ph,
            tick_ph,
            stats_ph,
            prog_ph,
            battery_ph,
            agent_icon_upload,
            package_icon_upload,
        )
        _render_simulation_results(summary, elapsed, agent_configs)

    _render_history_runs()
