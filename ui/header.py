from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st


def render_header():
    logo_path = Path("assets/Elberr.png")
    if logo_path.is_file():
        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
        st.markdown(
            f"""<div style="display:flex; align-items:center; gap:12px; margin-top:-1rem; margin-bottom:1.5rem;">
                <img src="data:image/png;base64,{logo_b64}" width="72" style="border-radius:8px;">
                <div>
                    <h1 style="margin:0; font-size:3.5rem; padding:0; line-height:1.1;">E.L.B.E.R.R.</h1>
                    <p style="margin:0; color:#9aa0a6; font-size:0.95rem;">
                        Efficient Logistics by Exploration with Robotic Retrieval
                    </p>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.title("E.L.B.E.R.R.")
        st.caption("Efficient Logistics by Exploration with Robotic Retrieval")
