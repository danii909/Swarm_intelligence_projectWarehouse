from __future__ import annotations

import io
import itertools
import json
import os
import random as _random
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ui.constants import STRATEGIES, STRATEGY_COLORS
from ui.helpers import build_agents, build_delivery_curve, style_dark_chart


def _render_benchmark_controls():
    left_col, right_col = st.columns([1.6, 1], gap="large")

    with left_col:
        st.markdown("### Configurazione")
        with st.container(border=True):
            st.markdown("#### Parametri generali")
            g1, g2 = st.columns(2)
            with g1:
                bench_num_agents = st.slider("N. agenti", min_value=1, max_value=10, value=5, key="bench_num_agents")
            with g2:
                bench_max_ticks = st.slider("Tick massimi", min_value=100, max_value=750, value=500, step=50, key="bench_max_ticks")

        st.markdown("")
        with st.container(border=True):
            st.markdown("#### Strategie")
            strat_mode = st.segmented_control("Modalità strategia", options=["Casuale", "Fissa"], default="Casuale", key="bench_strat_mode")
            benchmark_strategy_entries = [(sid, name) for sid, (name, _) in STRATEGIES.items() if name != "Ant-Colony"]
            if strat_mode == "Casuale":
                bench_strategies = st.multiselect(
                    "Strategie possibili",
                    options=[f"{sid} — {name}" for sid, name in benchmark_strategy_entries],
                    default=[f"{sid} — {name}" for sid, name in benchmark_strategy_entries],
                    key="bench_strats",
                    placeholder="Seleziona almeno una strategia",
                )
                bench_strategy_ids = [int(s.split(" — ")[0]) for s in bench_strategies]
            else:
                fixed_strat_label = st.selectbox(
                    "Strategia fissa",
                    options=[f"{sid} — {name}" for sid, name in benchmark_strategy_entries],
                    index=1,
                    key="bench_fixed_strat",
                )
                bench_strategy_ids = [int(fixed_strat_label.split(" — ")[0])]

        st.markdown("")
        with st.container(border=True):
            st.markdown("#### Raggi")
            r1, r2 = st.columns(2, gap="large")

            with r1:
                st.markdown("**Visione**")
                vis_mode = st.segmented_control("Modalità visione", options=["Casuale", "Fissa"], default="Casuale", key="bench_vis_mode")
                if vis_mode == "Casuale":
                    bench_vis_range = st.slider("Range visione", min_value=1, max_value=5, value=(1, 3), key="bench_vis_range")
                    vis_values = list(range(bench_vis_range[0], bench_vis_range[1] + 1))
                else:
                    fixed_vis = st.slider("Visione fissa", min_value=1, max_value=5, value=2, key="bench_fixed_vis")
                    vis_values = [fixed_vis]

            with r2:
                st.markdown("**Comunicazione**")
                comm_mode = st.segmented_control("Modalità comunicazione", options=["Casuale", "Fissa"], default="Casuale", key="bench_comm_mode")
                if comm_mode == "Casuale":
                    bench_comm_range = st.slider("Range comunicazione", min_value=1, max_value=2, value=(1, 2), key="bench_comm_range")
                    comm_values = list(range(bench_comm_range[0], bench_comm_range[1] + 1))
                else:
                    fixed_comm = st.slider("Comunicazione fissa", min_value=1, max_value=2, value=2, key="bench_fixed_comm")
                    comm_values = [fixed_comm]

    choices_per_agent = len(bench_strategy_ids) * len(vis_values) * len(comm_values)
    max_unique_presets = choices_per_agent ** bench_num_agents if choices_per_agent > 0 else 0

    with right_col:
        st.markdown("### Esecuzione")
        with st.container(border=True):
            st.markdown("#### Riepilogo spazio di ricerca")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Scelte per agente", choices_per_agent)
            with m2:
                st.metric("Preset unici max", max_unique_presets)

            st.markdown("")
            default_bench_n = min(20, max_unique_presets) if max_unique_presets > 0 else 1
            bench_n = st.number_input(
                "N. preset da testare",
                min_value=1,
                max_value=max_unique_presets if max_unique_presets > 0 else 1,
                value=default_bench_n,
                step=1,
                key="bench_n",
                help="Numero di configurazioni casuali da eseguire senza duplicati, fin dove possibile.",
            )
            st.markdown("")
            bench_clicked = st.button("▶ Avvia benchmark", type="primary", use_container_width=True, key="bench_run")
            if bench_strategy_ids:
                selected_names = [STRATEGIES[sid][0] for sid in bench_strategy_ids]
                st.caption(
                    f"Strategie selezionate: {', '.join(selected_names)} · Visione: {vis_values} · Comunicazione: {comm_values} · Run/preset: singola"
                )
            else:
                st.caption("Nessuna strategia selezionata.")

    return {
        "bench_clicked": bench_clicked,
        "bench_num_agents": bench_num_agents,
        "bench_max_ticks": bench_max_ticks,
        "bench_strategy_ids": bench_strategy_ids,
        "vis_values": vis_values,
        "comm_values": comm_values,
        "bench_n": bench_n,
        "max_unique_presets": max_unique_presets,
    }


def _run_benchmark(instance_path: str, seed: int, controls: dict):
    if not controls["bench_strategy_ids"]:
        st.error("Seleziona almeno una strategia.")
        st.stop()
    if not os.path.isfile(instance_path):
        st.error(f"File istanza non trovato: `{instance_path}`")
        st.stop()

    from src.environment.environment import Environment
    from src.simulation.simulator import Simulator

    actual_n = min(controls["bench_n"], controls["max_unique_presets"])
    rng = _random.Random(seed if seed >= 0 else None)
    per_agent_space = list(itertools.product(controls["bench_strategy_ids"], controls["vis_values"], controls["comm_values"]))
    generated_presets = []
    seen_signatures = set()

    if actual_n >= controls["max_unique_presets"]:
        for combo in itertools.product(per_agent_space, repeat=controls["bench_num_agents"]):
            generated_presets.append(list(combo))
    else:
        attempts = 0
        max_attempts = actual_n * 50
        while len(generated_presets) < actual_n and attempts < max_attempts:
            preset = tuple(rng.choice(per_agent_space) for _ in range(controls["bench_num_agents"]))
            if preset not in seen_signatures:
                seen_signatures.add(preset)
                generated_presets.append(list(preset))
            attempts += 1

    st.markdown("---")
    st.markdown("### Avanzamento benchmark")
    bench_progress = st.progress(0.0, text="Avvio benchmark... 0.0%")
    bench_status = st.empty()

    all_results = []
    preset_curves = {}
    t0_bench = time.perf_counter()
    total_jobs = max(1, len(generated_presets))

    for sim_i, preset in enumerate(generated_presets):
        preset_name = f"Preset {sim_i + 1}"
        agent_cfgs = [
            {"agent_id": ai, "strategy_id": strat_id, "radius": vis_r, "comm_radius": comm_r}
            for ai, (strat_id, vis_r, comm_r) in enumerate(preset)
        ]
        config_str = " ".join(
            f"A{ai + 1}:{STRATEGIES[strat_id][0][:4]}(v{vis_r},c{comm_r})"
            for ai, (strat_id, vis_r, comm_r) in enumerate(preset)
        )
        strat_counts = {}
        for strat_id, _, _ in preset:
            sname = STRATEGIES[strat_id][0]
            strat_counts[sname] = strat_counts.get(sname, 0) + 1
        team_desc = " + ".join(f"{cnt}×{sn}" for sn, cnt in sorted(strat_counts.items()))
        run_seed = seed + (sim_i * 1000) if seed >= 0 else rng.randint(0, 10_000_000)

        env_obj = Environment.from_json(instance_path)
        sim = Simulator(
            env=env_obj,
            agents=build_agents(agent_cfgs, controls["bench_num_agents"]),
            max_ticks=controls["bench_max_ticks"],
            seed=run_seed,
            verbose=False,
            log_every=1,
        )

        t0_sim = time.perf_counter()
        for _ in sim.step_gen():
            pass
        elapsed_sim = time.perf_counter() - t0_sim
        s = sim.metrics.summary()
        preset_curves[preset_name] = build_delivery_curve(sim.metrics.history, controls["bench_max_ticks"])

        pct = (sim_i + 1) / total_jobs
        bench_progress.progress(pct, text=f"{pct * 100:.0f}% - Preset {sim_i + 1}/{len(generated_presets)}")
        all_results.append({
            "preset_name": preset_name,
            "config_str": config_str,
            "team_desc": team_desc,
            "agent_configs": agent_cfgs,
            "preset_raw": preset,
            "dominant_strategy": max(strat_counts, key=strat_counts.get),
            "avg_vis": round(float(np.mean([vis_r for _, vis_r, _ in preset])), 2),
            "avg_comm": round(float(np.mean([comm_r for _, _, comm_r in preset])), 2),
            "objects_delivered": s["objects_delivered"],
            "total_objects": s["total_objects"],
            "delivery_rate": round(float(s["delivery_rate"]), 4),
            "total_ticks": s["total_ticks"],
            "average_energy": round(float(s["average_energy_consumed"]), 3),
            "first_pickup_tick": s["first_pickup_tick"],
            "first_delivery_tick": s["first_delivery_tick"],
            "cpu_time": round(float(elapsed_sim), 3),
        })
        bench_status.info(
            f"{preset_name} · {team_desc} · completion {s['completion_rate'] * 100:.1f}% · Tick {s['total_ticks']:.1f}"
        )

    st.session_state["bench_results"] = {
        "all_results": all_results,
        "preset_curves": preset_curves,
        "actual_n": len(generated_presets),
        "total_bench_time": time.perf_counter() - t0_bench,
        "bench_strategy_ids": controls["bench_strategy_ids"],
        "vis_values": controls["vis_values"],
        "bench_max_ticks": controls["bench_max_ticks"],
    }
    bench_progress.progress(1.0, text="Benchmark completato (100.0%)")
    bench_status.empty()


def _create_benchmark_zip(df, results, seed, instance_path=""):
    """Crea uno ZIP con tutti i dati e i grafici del benchmark."""
    buf_zip = io.BytesIO()
    
    with zipfile.ZipFile(buf_zip, "w", zipfile.ZIP_DEFLATED) as z:
        # 1. Summary CSV
        csv_cols = [
            "preset_name", "config_str", "team_desc", "avg_vis", "avg_comm", "objects_delivered",
            "total_objects", "delivery_rate", "total_ticks", "average_energy", "cpu_time",
        ]
        csv_data = df[[c for c in csv_cols if c in df.columns]].to_csv(index=False)
        z.writestr("summary.csv", csv_data)
        
        # 2. Full results JSON (with agent_configs)
        z.writestr("results.json", json.dumps(results["all_results"], indent=2))
        
        # 3. Curves JSON
        curves_serializable = {k: v for k, v in results.get("preset_curves", {}).items()}
        z.writestr("curves.json", json.dumps(curves_serializable, indent=2))
        
        # 4. Metadata JSON
        metadata = {
            "format_version": "1.0",
            "generated_at": time.time(),
            "generated_at_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": seed,
            "instance_path": instance_path,
            "bench_max_ticks": results.get("bench_max_ticks"),
            "bench_strategy_ids": results.get("bench_strategy_ids"),
            "vis_values": results.get("vis_values"),
            "comm_values": results.get("comm_values"),
            "actual_presets_run": results.get("actual_n", 0),
            "total_bench_time_seconds": results.get("total_bench_time", 0),
        }
        z.writestr("metadata.json", json.dumps(metadata, indent=2))
        
        # 5. Genera e salva i 3 grafici principali
        df_rank = df.sort_values(
            by=["total_ticks", "delivery_rate", "average_energy"],
            ascending=[True, False, True]
        ).reset_index(drop=True)
        bench_max_ticks = int(results.get("bench_max_ticks", 500))
        preset_curves = results.get("preset_curves", {})
        
        # Grafico Tick
        fig_ticks, ax_ticks = plt.subplots(figsize=(min(max(10, len(df_rank) * 0.35), 38), 4.6), facecolor="#0e1117")
        style_dark_chart(ax_ticks)
        ax_ticks.bar(
            range(len(df_rank)),
            df_rank["total_ticks"].values,
            color=[STRATEGY_COLORS.get(s, "#888") for s in df_rank["dominant_strategy"].tolist()],
            edgecolor="#444",
            linewidth=0.5,
        )
        mean_ticks = float(np.mean(df_rank["total_ticks"].values))
        ax_ticks.axhline(y=mean_ticks, color="#FFD700", linestyle="--", linewidth=1.5, label=f"Media: {mean_ticks:.1f}")
        ax_ticks.set_title("Tick per preset", color="white")
        ax_ticks.set_xlabel("Preset", color="white")
        ax_ticks.set_ylabel("Tick", color="white")
        if len(df_rank) <= 60:
            ax_ticks.set_xticks(range(len(df_rank)))
            ax_ticks.set_xticklabels(df_rank["preset_name"].tolist(), rotation=35, ha="right", color="white", fontsize=7)
        ax_ticks.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")
        fig_ticks.tight_layout()
        buf_ticks = io.BytesIO()
        fig_ticks.savefig(buf_ticks, format="png", dpi=100, facecolor="#0e1117")
        buf_ticks.seek(0)
        z.writestr("plot_ticks.png", buf_ticks.read())
        plt.close(fig_ticks)
        
        # Grafico Oggetti
        fig_obj, ax_obj = plt.subplots(figsize=(min(max(10, len(df_rank) * 0.35), 38), 4.6), facecolor="#0e1117")
        style_dark_chart(ax_obj)
        total_obj = int(df["total_objects"].iloc[0]) if len(df) > 0 else 10
        y_objects = df_rank["objects_delivered"].values
        ax_obj.bar(
            range(len(df_rank)),
            y_objects,
            color=["#55A868" if d >= total_obj else "#DD8452" if d >= total_obj * 0.5 else "#C44E52" for d in y_objects],
            edgecolor="#444",
            linewidth=0.5,
        )
        ax_obj.axhline(y=total_obj, color="#FFD700", linestyle=":", linewidth=1.2, label=f"Totale oggetti: {total_obj}")
        ax_obj.set_title("Oggetti consegnati per preset", color="white")
        ax_obj.set_xlabel("Preset", color="white")
        ax_obj.set_ylabel("Oggetti consegnati", color="white")
        if len(df_rank) <= 60:
            ax_obj.set_xticks(range(len(df_rank)))
            ax_obj.set_xticklabels(df_rank["preset_name"].tolist(), rotation=35, ha="right", color="white", fontsize=7)
        ax_obj.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")
        fig_obj.tight_layout()
        buf_obj = io.BytesIO()
        fig_obj.savefig(buf_obj, format="png", dpi=100, facecolor="#0e1117")
        buf_obj.seek(0)
        z.writestr("plot_objects.png", buf_obj.read())
        plt.close(fig_obj)
        
        # Grafico Curve cumulative
        fig_curves, ax_curves = plt.subplots(figsize=(10, 5), facecolor="#0e1117")
        style_dark_chart(ax_curves)
        x_ticks = np.arange(1, bench_max_ticks + 1)
        curves_to_show = min(10, len(df_rank))
        preset_names = df_rank.head(curves_to_show)["preset_name"].tolist()
        palette = plt.get_cmap("tab20")
        for i, preset_name in enumerate(preset_names):
            curve = preset_curves.get(preset_name)
            if curve is None:
                continue
            curve_arr = np.array(curve, dtype=float)
            if curve_arr.ndim > 1:
                curve_arr = curve_arr[0]
            ax_curves.plot(x_ticks, curve_arr, linewidth=2, color=palette(i % 20), label=preset_name)
        ax_curves.set_title("Curve cumulative deliveries (Top 10)", color="white")
        ax_curves.set_xlabel("Tick", color="white")
        ax_curves.set_ylabel("Oggetti consegnati", color="white")
        ax_curves.set_ylim(bottom=0)
        if curves_to_show <= 20:
            ax_curves.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")
        fig_curves.tight_layout()
        buf_curves = io.BytesIO()
        fig_curves.savefig(buf_curves, format="png", dpi=100, facecolor="#0e1117")
        buf_curves.seek(0)
        z.writestr("plot_curves.png", buf_curves.read())
        plt.close(fig_curves)
    
    buf_zip.seek(0)
    return buf_zip.getvalue()


def _render_benchmark_plots(df_rank, df, preset_curves, bench_max_ticks):
    st.markdown("#### 📈 Grafici benchmark")
    fig_w = min(max(10, len(df_rank) * 0.35), 38)

    fig_ticks, ax_ticks = plt.subplots(figsize=(fig_w, 4.6), facecolor="#0e1117")
    style_dark_chart(ax_ticks)
    ax_ticks.bar(
        range(len(df_rank)),
        df_rank["total_ticks"].values,
        color=[STRATEGY_COLORS.get(s, "#888") for s in df_rank["dominant_strategy"].tolist()],
        edgecolor="#444",
        linewidth=0.5,
    )
    mean_ticks = float(np.mean(df_rank["total_ticks"].values))
    ax_ticks.axhline(y=mean_ticks, color="#FFD700", linestyle="--", linewidth=1.5, label=f"Media: {mean_ticks:.1f}")
    ax_ticks.set_title("Tick per preset", color="white")
    ax_ticks.set_xlabel("Preset", color="white")
    ax_ticks.set_ylabel("Tick", color="white")
    if len(df_rank) <= 60:
        ax_ticks.set_xticks(range(len(df_rank)))
        ax_ticks.set_xticklabels(df_rank["preset_name"].tolist(), rotation=35, ha="right", color="white", fontsize=7)
    else:
        ax_ticks.set_xticks([])
    ax_ticks.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")
    fig_ticks.tight_layout()
    st.pyplot(fig_ticks)
    plt.close(fig_ticks)

    fig_obj, ax_obj = plt.subplots(figsize=(fig_w, 4.6), facecolor="#0e1117")
    style_dark_chart(ax_obj)
    total_obj = int(df["total_objects"].iloc[0])
    y_objects = df_rank["objects_delivered"].values
    ax_obj.bar(
        range(len(df_rank)),
        y_objects,
        color=["#55A868" if d >= total_obj else "#DD8452" if d >= total_obj * 0.5 else "#C44E52" for d in y_objects],
        edgecolor="#444",
        linewidth=0.5,
    )
    ax_obj.axhline(y=total_obj, color="#FFD700", linestyle=":", linewidth=1.2, label=f"Totale oggetti: {total_obj}")
    ax_obj.set_title("Oggetti consegnati per preset", color="white")
    ax_obj.set_xlabel("Preset", color="white")
    ax_obj.set_ylabel("Oggetti consegnati", color="white")
    if len(df_rank) <= 60:
        ax_obj.set_xticks(range(len(df_rank)))
        ax_obj.set_xticklabels(df_rank["preset_name"].tolist(), rotation=35, ha="right", color="white", fontsize=7)
    else:
        ax_obj.set_xticks([])
    ax_obj.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")
    fig_obj.tight_layout()
    st.pyplot(fig_obj)
    plt.close(fig_obj)

    st.markdown("#### Curve cumulative deliveries")
    curves_to_show = st.slider("N. curve", min_value=1, max_value=max(1, len(df_rank)), value=min(10, len(df_rank)), key="bench_curves_to_show")
    if not preset_curves:
        st.info("Curve cumulative deliveries non disponibili: esegui un nuovo benchmark.")
        return

    fig_curves, ax_curves = plt.subplots(figsize=(10, 5), facecolor="#0e1117")
    style_dark_chart(ax_curves)
    x_ticks = np.arange(1, bench_max_ticks + 1)
    preset_names = df_rank.head(min(curves_to_show, len(df_rank)))["preset_name"].tolist()
    palette = plt.get_cmap("tab20")
    for i, preset_name in enumerate(preset_names):
        curve = preset_curves.get(preset_name)
        if curve is None:
            continue
        curve_arr = np.array(curve, dtype=float)
        if curve_arr.ndim > 1:
            curve_arr = curve_arr[0]
        ax_curves.plot(x_ticks, curve_arr, linewidth=2, color=palette(i % 20), label=preset_name)
    ax_curves.set_title("Curve cumulative deliveries", color="white")
    ax_curves.set_xlabel("Tick", color="white")
    ax_curves.set_ylabel("Oggetti consegnati", color="white")
    ax_curves.set_ylim(bottom=0)
    if len(preset_names) <= 20:
        ax_curves.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")
    fig_curves.tight_layout()
    st.pyplot(fig_curves)
    plt.close(fig_curves)
    if len(preset_names) > 10:
        st.caption("Legenda nascosta automaticamente quando le curve mostrate sono piu di 10.")


def _render_downloadable_top_presets(df_rank):
    st.markdown("#### ⬇ Top 10 preset scaricabili")
    medals = ["🥇", "🥈", "🥉", "4.", "5.", "6.", "7.", "8.", "9.", "10."]
    for i in range(min(10, len(df_rank))):
        row = df_rank.iloc[i]
        # Pulsante a sinistra, info a destra
        col_btn, col_info = st.columns([1, 30])
        with col_btn:
            dl_data = {"name": row["preset_name"], "num_agents": len(row["agent_configs"]), "agents": row["agent_configs"]}
            st.download_button(
                label="⬇",
                data=json.dumps(dl_data, indent=4),
                file_name=f"{row['preset_name'].replace(' ', '_')}.json",
                mime="application/json",
                key=f"dl_top_{i}",
            )
        with col_info:
            st.markdown(
                f"{medals[i]} **{row['preset_name']}** — {row['team_desc']} — consegnati {row['objects_delivered']}/{row['total_objects']} in **{row['total_ticks']} tick** — energia {row['average_energy']:.1f}"
            )
            st.caption(f"Dettaglio: {row['config_str']}")


def _render_benchmark_results():
    if "bench_results" not in st.session_state:
        return

    results = st.session_state["bench_results"]
    df = pd.DataFrame(results["all_results"])
    bench_max_ticks = int(results.get("bench_max_ticks", max(1, int(df["total_ticks"].max())) if not df.empty else 1))
    preset_curves = results.get("preset_curves", {})

    st.divider()
    st.subheader("📊 Risultati benchmark")

    # Download completo con ZIP
    zip_data = _create_benchmark_zip(df, results, seed=0, instance_path="")
    st.download_button(
        "💾 Scarica risultati completi (ZIP con dati, grafici e metadata)",
        data=zip_data,
        file_name=f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
        key="dl_all_zip",
    )

    df_rank = df.sort_values(by=["total_ticks", "delivery_rate", "average_energy"], ascending=[True, False, True]).reset_index(drop=True)
    st.markdown("#### 🏆 Top 10 preset")
    st.dataframe(pd.DataFrame({
        "Pos.": range(1, min(10, len(df_rank)) + 1),
        "Preset": df_rank.head(10)["preset_name"],
        "Team": df_rank.head(10)["team_desc"],
        "Configurazione": df_rank.head(10)["config_str"],
        "Completion": [f"{r * 100:.1f}%" for r in df_rank.head(10)["delivery_rate"]],
        "Tick": df_rank.head(10)["total_ticks"].round(2),
        "Energia media": df_rank.head(10)["average_energy"].round(2),
    }), width="stretch", hide_index=True)

    _render_downloadable_top_presets(df_rank)

    st.divider()
    st.markdown("#### 📋 Classifica totale")
    df_all_sorted = df.sort_values(by=["total_ticks", "delivery_rate"], ascending=[True, False]).reset_index(drop=True)
    st.dataframe(pd.DataFrame({
        "Pos.": range(1, len(df_all_sorted) + 1),
        "Preset": df_all_sorted["preset_name"],
        "Team": df_all_sorted["team_desc"],
        "Configurazione": df_all_sorted["config_str"],
        "Consegnati": df_all_sorted["objects_delivered"],
        "Completion": [f"{r * 100:.1f}%" for r in df_all_sorted["delivery_rate"]],
        "Tick": df_all_sorted["total_ticks"].round(2),
        "Energia media": df_all_sorted["average_energy"].round(1),
        "CPU (s)": df_all_sorted["cpu_time"],
    }), width="stretch", hide_index=True)

    _render_benchmark_plots(df_rank, df, preset_curves, bench_max_ticks)


def render_benchmark_tab(instance_path: str, seed: int):
    controls = _render_benchmark_controls()
    if controls["bench_clicked"]:
        _run_benchmark(instance_path, seed, controls)
    _render_benchmark_results()
