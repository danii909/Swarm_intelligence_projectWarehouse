from __future__ import annotations

import streamlit as st

from ui.benchmark_tab import render_benchmark_tab
from ui.header import render_header
from ui.sidebar import render_global_sidebar
from ui.simulation_tab import render_simulation_tab


def configure_page():
    st.set_page_config(
        page_title="Swarm Intelligence Project",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def run_app():
    configure_page()
    instance_path, seed = render_global_sidebar()
    render_header()
    tab_sim, tab_bench = st.tabs(["🎮 Simulazione", "🔬 Benchmark"])
    with tab_sim:
        render_simulation_tab(instance_path, seed)
    with tab_bench:
        render_benchmark_tab(instance_path, seed)
