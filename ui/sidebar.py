from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_global_sidebar():
    with st.sidebar:
        st.header("⚙️ Configurazione globale")
        st.divider()

        st.subheader("📁 Istanza")
        instances_found = sorted(
            str(p) for p in Path(".").glob("**/*.json")
            if "Consegna" in str(p) or "instances" in str(p).lower()
        )
        if not instances_found:
            instances_found = ["Consegna/A.json", "Consegna/B.json", "A.json", "B.json"]
        instances_found = [p for p in instances_found if Path(p).exists()] or instances_found

        instance_path = st.selectbox("File istanza", options=instances_found, index=0)

        st.subheader("⚙️ Seed")
        seed = st.number_input("(−1 = casuale)", min_value=-1, max_value=9999, value=42)
        st.divider()

    return instance_path, seed
