import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Rack Optimizer", layout="wide")

# --- INITIALIZE MEMORY & DEFAULTS ---
if 'history' not in st.session_state:
    st.session_state.history = []

classic_defaults = {
    "b_l": 400.0, "b_w": 200.0, "clear_ht": 30.0,
    "sb_l": False, "sb_r": False, "sb_t": False, "sb_b": False,
    "rt_l": 30.0, "rt_r": 30.0, "rt_t": 30.0, "rt_b": 30.0,
    "col_x": 40.0, "col_y": 40.0, "col_w": 12.0, "col_d": 12.0,
    "orient": "Vertical", "r_in": 42.0, "f_in": 12.0, "min_a": 10.0, "single": False,
    "sb_depth": 0.0
}

for key, val in classic_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- CALLBACK FOR RESET ---
def handle_clear_all():
    dims = ["b_l", "b_w", "clear_ht", "rt_l", "rt_r", "rt_t", "rt_b", "r_in", "f_in", "min_a", "sb_depth"]
    checks = ["sb_l", "sb_r", "sb_t", "sb_b", "single"]
    for k in dims: st.session_state[k] = 0.0
    for k in checks: st.session_state[k] = False
    st.session_state["orient"] = "Vertical"

# --- CALLBACK FUNCTIONS ---
def apply_l_to_all():
    st.session_state.rt_r = float(st.session_state.rt_l)
    st.session_state.rt_t = float(st.session_state.rt_l)
    st.session_state.rt_b = float(st.session_state.rt_l)

def apply_col_specs():
    st.session_state.rt_l = float(st.session_state.col_x)
    st.session_state.rt_r = float(st.session_state.col_x)
    st.session_state.rt_t = float(st.session_state.col_y)
    st.session_state.rt_b = float(st.session_state.col_y)

def update_sb_depth():
    if st.session_state.sb_l or st.session_state.sb_r:
        ref_span = st.session_state.col_x
    elif st.session_state.sb_t or st.session_state.sb_b:
        ref_span = st.session_state.col_y
    else:
        ref_span = 0
        
    if any([st.session_state.sb_l, st.session_state.sb_r, st.session_state.sb_t, st.session_state.sb_b]):
        if st.session_state.sb_depth == 0.0:
            st.session_state.sb_depth = (ref_span * 1.5) if ref_span > 0 else 60.0

# --- CSS: THEME, DROPDOWN & RESPONSIVE SIDEBAR ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800;900&family=Inter:wght@400;700&display=swap');
    .stApp { background-color: #000000; color: #ffffff; font-family: 'Inter', sans-serif; }
    * { border-radius: 0px !important; }
    
    h1 { color: #00f3ff !important; font-family: 'JetBrains Mono', monospace !important; font-weight: 900 !important; text-transform: uppercase; letter-spacing: -0.05em; margin-bottom: 0.2rem; }
    .subtitle { color: #ffffff; font-family: 'Inter', sans-serif; font-size: 1rem; opacity: 0.8; margin-bottom: 2rem; }
    h2, h3 { color: #ffffff !important; font-family: 'JetBrains Mono', monospace !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.05em; }
    
    .cyan-divider { border: none; border-top: 2px dotted #00f3ff; width: 100%; margin: 2rem 0; opacity: 0.6; }
    [data-testid="stWidgetLabel"] p { color: #00f3ff !important; font-family: 'JetBrains Mono', monospace !important; font-weight: 900 !important; font-size: 0.85em !important; text-transform: uppercase !important; }
    
    [data-testid="stTooltipIcon"] svg { fill: #00f3ff !important; }
    div[data-testid="stTooltipContent"] { background-color: #1a1a1a !important; color: #ffffff !important; border: 1px solid #00f3ff !important; padding: 10px !important; }

    /* Dropdown Chevron White */
    div[data-baseweb="select"] svg { fill: #ffffff !important; }

    /* Static White Border for Inputs and Selectbox */
    div[data-baseweb="input"] > div, .stNumberInput input, .stTextInput input, div[data-baseweb="select"] > div { 
        background-color: #1a1a1a !important; color: #ffffff !important; border: 1px solid #ffffff !important; font-family: 'JetBrains Mono', monospace !important; 
    }
    
    input::placeholder { color: #ffffff !important; opacity: 1.0 !important; }

    div[data-baseweb="popover"] ul { background-color: #1a1a1a !important; border: 1px solid #00f3ff !important; }
    div[data-baseweb="popover"] li { color: #ffffff !important; font-family: 'JetBrains Mono', monospace !important; }
    div[data-baseweb="popover"] li:hover { background-color: #ff00ff !important; color: white !important; }

    .stButton > button {
        background-color: rgb(255, 0, 255) !important; color: #ffffff !important; font-family: 'JetBrains Mono', monospace !important; font-weight: 900 !important; text-transform: uppercase !important; border: none !important; box-shadow: 4px 4px 0px #ffffff !important;
    }

    .terminal-table { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; margin-top: 20px; }
    .terminal-table th { color: #00f3ff; font-size: 0.75em; text-transform: uppercase; font-weight: 900; text-align: left; padding: 10px; border-bottom: 1px dashed #333; }
    .terminal-table td { color: #ffffff; font-size: 0.85em; padding: 10px; border-bottom: 1px solid #1a1a1a; }

    /* Fixed Receipt: Desktop vs Mobile Snapping */
    .fixed-receipt-sidebar { background-color: #000000; border: 2px solid #ffffff; padding: 20px; box-shadow: 10px 10px 0px #ff00ff; z-index: 9999; }
    @media (min-width: 1024px) {
        .fixed-receipt-sidebar { position: fixed; top: 80px; right: 20px; width: 325px; }
    }
    @media (max-width: 1023px) {
        .fixed-receipt-sidebar { position: relative; top: 0; right: 0; width: 100%; margin: 20px 0; }
    }

    .metric-card { background-color: #000000; border: 1px solid #00ff00; padding: 12px; margin-top: 15px; box-shadow: 5px 5px 0px #00ff00; }
    .warning-text { color: #ff00ff; font-family: 'JetBrains Mono', monospace; font-size: 0.75em; font-weight: 900; margin-bottom: 5px; text-transform: uppercase; }
    [data-testid="stSidebar"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

col_main, _ = st.columns([850, 310])

with col_main:
    st.title("RACK_OPTIMIZER")
    st.markdown('<div class="subtitle">A high-precision structural layout tool for calculating storage density and rack layouts for fulfilment centers.</div>', unsafe_allow_html=True)
    st.button("CLEAR ALL FIELDS", on_click=handle_clear_all)

    st.header("1. BUILDING DIMENSIONS")
    b_c = st.columns(3)
    b_l = b_c[0].number_input("Length (ft)", key="b_l", step=1.0, help="The long way across your warehouse floor.")
    b_w = b_c[1].number_input("Width (ft)", key="b_w", step=1.0, help="The narrow way across your warehouse floor.")
    clear_ht = b_c[2].number_input("Clear Height (ft)", key="clear_ht", step=1.0, help="How high can you stack before hitting the ceiling?")
    st.markdown('<div class="cyan-divider"></div>', unsafe_allow_html=True)

    st.header("2. CLEAR ZONES")
    sb_checks = st.columns(4)
    sb_l = sb_checks[0].checkbox("Speed Bay L", key="sb_l", on_change=update_sb_depth, help="Extra room on the left for traffic.")
    sb_r = sb_checks[1].checkbox("Speed Bay R", key="sb_r", on_change=update_sb_depth, help="Extra room on the right for traffic.")
    sb_t = sb_checks[2].checkbox("Speed Bay T", key="sb_t", on_change=update_sb_depth, help="Extra room on the top for traffic.")
    sb_b = sb_checks[3].checkbox("Speed Bay B", key="sb_b", on_change=update_sb_depth, help="Extra room on the bottom for traffic.")
    
    if any([st.session_state.sb_l, st.session_state.sb_r, st.session_state.sb_t, st.session_state.sb_b]):
        st.columns([1, 3])[0].number_input("Speed Bay Depth (ft)", key="sb_depth", step=1.0, help="How deep do those staging lanes need to be?")
    
    st.subheader("RACETRACK")
    rt_c = st.columns(4)
    rt_l = rt_c[0].number_input("L Buffer", key="rt_l", step=1.0, help="Gap from the racks to the left wall.")
    rt_r = rt_c[1].number_input("R Buffer", key="rt_r", step=1.0, help="Gap from the racks to the right wall.")
    rt_t = rt_c[2].number_input("T Buffer", key="rt_t", step=1.0, help="Gap from the racks to the top wall.")
    rt_b = rt_c[3].number_input("B Buffer", key="rt_b", step=1.0, help="Gap from the racks to the bottom wall.")
    
    rt_btns = st.columns([1.5, 1.5, 5])
    rt_btns[0].button("APPLY L. TO ALL", on_click=apply_l_to_all)
    rt_btns[1].button("APPLY COL. SPECS", on_click=apply_col_specs)
    st.markdown('<div class="cyan-divider"></div>', unsafe_allow_html=True)

    st.header("3. COLUMN SPECIFICATIONS")
    c_c = st.columns(4)
    col_x = c_c[0].number_input("X Span (ft)", key="col_x", step=1.0, help="Distance between column centers horizontally.")
    col_y = c_c[1].number_input("Y Span (ft)", key="col_y", step=1.0, help="Distance between column centers vertically.")
    col_w_ft = c_c[2].number_input("Col W (in)", key="col_w", step=1.0, help="Width of the physical post.")/12
    col_d_ft = c_c[3].number_input("Col D (in)", key="col_d", step=1.0, help="Depth of the physical post.")/12
    st.markdown('<div class="cyan-divider"></div>', unsafe_allow_html=True)

    st.header("4. RACK OPTIMIZATION")
    r_c = st.columns(4)
    orient = r_c[0].selectbox("Orientation", ["Vertical", "Horizontal"], key="orient", help="Up-down or side-to-side?")
    r_in = r_c[1].number_input("Rack D (in)", key="r_in", step=1.0, help="Depth of a single rack frame.")
    f_in = r_c[2].number_input("Flue (in)", key="f_in", step=1.0, help="Safety gap between back-to-back rows.")
    min_a = r_c[3].number_input("Min Aisle (ft)", key="min_a", step=1.0, help="Turning room for your forklifts.")
    allow_single = st.checkbox("Allow single aisles", key="single", help="Mix single and double rows to maximize pallet count.")
    st.markdown('<div class="cyan-divider"></div>', unsafe_allow_html=True)

    # --- MATH ENGINE ---
    ready = b_l > 0 and b_w > 0 and col_x > 0 and col_y > 0 and r_in > 0
    rack_cf, build_util, best_aisle, p_util, unique_final = 0, 0, 0, 0, []

    if ready:
        r_ft, f_ft = r_in/12, f_in/12
        wall_l, wall_r = (st.session_state.sb_depth if sb_l else 0), (b_l - (st.session_state.sb_depth if sb_r else 0))
        wall_b, wall_t = (st.session_state.sb_depth if sb_b else 0), (b_w - (st.session_state.sb_depth if sb_t else 0))
        avail_x, avail_y = wall_r - wall_l, wall_t - wall_b
        if avail_x > 0 and avail_y > 0:
            num_x, num_y = int(avail_x / col_x), int(avail_y / col_y)
            off_x, off_y = wall_l + (avail_x - (num_x * col_x)) / 2, wall_b + (avail_y - (num_y * col_y)) / 2

            def get_coords(limit, grid_start, step):
                coords, dist_a = [], min_a 
                if grid_start >= limit: return coords, dist_a
                grid = np.arange(grid_start, limit + 0.1, step)
                col_dim = col_d_ft if orient == "Horizontal" else col_w_ft
                eff_flue = max(f_ft, col_dim)
                for g in grid:
                    bay_units = [] 
                    if allow_single:
                        bay_units.append(("S", r_ft)); remaining = (step - (r_ft + min_a))
                    else:
                        bay_units.append(("D", r_ft * 2 + eff_flue)); remaining = (step - (r_ft * 2 + eff_flue + min_a))
                    while remaining >= (r_ft * 2 + f_ft + min_a):
                        bay_units.append(("D", r_ft * 2 + f_ft)); remaining -= (r_ft * 2 + f_ft + min_a)
                    if allow_single and remaining >= (r_ft + min_a):
                        bay_units.append(("S", r_ft)); remaining -= (r_ft + min_a)
                    num_a = len(bay_units)
                    dist_a = min_a + (remaining / num_a) if num_a > 0 else min_a
                    ptr = g
                    for i, (utype, _) in enumerate(bay_units):
                        if i == 0:
                            if utype == "S": coords.append((ptr - r_ft/2, ptr + r_ft/2)); ptr += (r_ft/2 + dist_a)
                            else: coords.append((ptr - eff_flue/2 - r_ft, ptr - eff_flue/2)); coords.append((ptr + eff_flue/2, ptr + eff_flue/2 + r_ft)); ptr += (eff_flue/2 + r_ft + dist_a)
                        else:
                            if utype == "S": coords.append((ptr, ptr + r_ft)); ptr += (r_ft + dist_a)
                            else: coords.append((ptr, ptr + r_ft)); coords.append((ptr + r_ft + f_ft, ptr + r_ft * 2 + f_ft)); ptr += (r_ft * 2 + f_ft + dist_a)
                return list(set(coords)), dist_a

            raw_coords, best_aisle = get_coords(wall_t if orient == "Horizontal" else wall_r, off_y if orient == "Horizontal" else off_x, col_y if orient == "Horizontal" else col_x)
            r_min_x, r_max_x = max(wall_l, rt_l), min(wall_r, b_l - rt_r)
            r_min_y, r_max_y = max(wall_b, rt_b), min(wall_t, b_w - rt_t)
            unique_final = sorted([r for r in raw_coords if r[0] >= (r_min_y if orient == "Horizontal" else r_min_x) and r[1] <= (r_max_y if orient == "Horizontal" else r_max_x)])
            rack_cf = len(unique_final) * r_ft * (r_max_x - r_min_x if orient == "Horizontal" else r_max_y - r_min_y) * clear_ht
            build_util = (rack_cf / (b_l * b_w * clear_ht)) * 100
            test_lim = (off_y if orient == "Horizontal" else off_x) + (col_y if orient == "Horizontal" else col_x)
            test_start = (off_y if orient == "Horizontal" else off_x)
            sample_bay_coords, _ = get_coords(test_lim, test_start, col_y if orient == "Horizontal" else col_x)
            p_util = ((len(sample_bay_coords) * r_ft * (st.session_state.col_y if orient == "Horizontal" else st.session_state.col_x)) / (col_x * col_y)) * 100

    # 5. VISUALIZATIONS
    st.header("5. VISUALIZATIONS")
    st.subheader("BUILDING VIEW")
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=b_l, y1=b_w, line=dict(color="#ffffff", width=2))
    
    if b_l > 0 and b_w > 0:
        if st.session_state.sb_l: fig.add_shape(type="line", x0=st.session_state.sb_depth, y0=0, x1=st.session_state.sb_depth, y1=b_w, line=dict(color="#ff00ff", width=2, dash="dot"))
        if st.session_state.sb_r: fig.add_shape(type="line", x0=b_l-st.session_state.sb_depth, y0=0, x1=b_l-st.session_state.sb_depth, y1=b_w, line=dict(color="#ff00ff", width=2, dash="dot"))
        if st.session_state.sb_b: fig.add_shape(type="line", x0=0, y0=st.session_state.sb_depth, x1=b_l, y1=st.session_state.sb_depth, line=dict(color="#ff00ff", width=2, dash="dot"))
        if st.session_state.sb_t: fig.add_shape(type="line", x0=0, y0=b_w-st.session_state.sb_depth, x1=b_l, y1=b_w-st.session_state.sb_depth, line=dict(color="#ff00ff", width=2, dash="dot"))

    if ready:
        fig.add_shape(type="rect", x0=r_min_x, y0=r_min_y, x1=r_max_x, y1=r_max_y, line=dict(color="#00ff00", width=1.5, dash="dot"))
        for r in unique_final:
            x0, x1 = (r_min_x, r_max_x) if orient == "Horizontal" else (r[0], r[1])
            y0, y1 = (r[0], r[1]) if orient == "Horizontal" else (r_min_y, r_max_y)
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, fillcolor="#00ff00", opacity=0.3, line_width=0.5, line_color="#00ff00")
        for x in np.arange(off_x if 'off_x' in locals() else 0, (b_l-st.session_state.sb_depth if st.session_state.sb_r else b_l) + 0.1, col_x):
            for y in np.arange(off_y if 'off_y' in locals() else 0, (b_w-st.session_state.sb_depth if st.session_state.sb_t else b_w) + 0.1, col_y):
                fig.add_shape(type="rect", x0=x-col_w_ft/2, y0=y-col_d_ft/2, x1=x+col_w_ft/2, y1=y+col_d_ft/2, fillcolor="#ff00ff")
    
    x_buf, y_buf = (b_l * 0.05 if b_l > 0 else 5), (b_w * 0.05 if b_w > 0 else 5)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=20, b=20), xaxis=dict(range=[-x_buf, b_l + x_buf], scaleanchor="y", scaleratio=1, showgrid=False), yaxis=dict(range=[-y_buf, b_w + y_buf], showgrid=False))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ENGINEERING PATTERN DETAIL")
    fig2 = go.Figure()
    for x in [0, col_x, col_x*2]:
        for y in [0, col_y, col_y*2]:
            fig2.add_shape(type="rect", x0=x-col_w_ft/2, y0=y-col_d_ft/2, x1=x+col_w_ft/2, y1=y+col_d_ft/2, fillcolor="#ff00ff")
    if ready:
        limit_val = col_y*2 if orient == "Horizontal" else col_x*2
        step_val = col_y if orient == "Horizontal" else col_x
        p_coords, _ = get_coords(limit_val, 0, step_val)
        visual_buffer = r_ft / 2
        for r in p_coords:
            if r[0] >= -visual_buffer and r[1] <= limit_val + visual_buffer:
                px0, px1 = (0, col_x*2) if orient == "Horizontal" else (r[0], r[1])
                py0, py1 = (r[0], r[1]) if orient == "Horizontal" else (0, col_y*2)
                fig2.add_shape(type="rect", x0=px0, y0=py0, x1=px1, y1=py1, line=dict(color="#39FF14", width=1.5), fillcolor="rgba(57, 255, 20, 0.4)")
    fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", width=800, height=450, xaxis=dict(range=[-10, max(col_x*2, 10)+10], scaleanchor="y", showgrid=False), yaxis=dict(range=[-10, max(col_y*2, 10)+10], showgrid=False))
    st.plotly_chart(fig2, use_container_width=True)

# --- SIDEBAR RECEIPT ---
area_val = f"{b_l*b_w:,.0f}" if (b_l > 0 and b_w > 0) else "0"
aisle_val = f"{best_aisle:.2f} ft" if ready else "N/A"
util_val = f"{build_util:.1f}%" if ready else "0.0%"
receipt_html = (
    f'<div class="fixed-receipt-sidebar">'
    f'<div style="font-weight: 900; color: #00f3ff; font-size: 0.8em; margin-bottom: 5px; font-family: JetBrains Mono;">BUILDING_METRICS</div>'
    f'<div style="color: #666; font-size: 0.85em; font-style: italic;">{b_l:,.0f}ft L × {b_w:,.0f}ft W</div>'
    f'<div style="font-weight: bold; color: white;">Area: {area_val} SF</div>'
    f'<div style="margin: 15px 0; border-top: 1px dashed #333;"></div>'
    f'<div style="font-weight: 900; color: #00f3ff; font-size: 0.8em; margin-bottom: 5px; font-family: JetBrains Mono;">RACK_DATA</div>'
    f'<div style="color: #666; font-size: 0.85em; font-style: italic;">{len(unique_final)} rows</div>'
    f'<div style="font-weight: bold; color: white;">Actual Aisle: {aisle_val}</div>'
    f'<div class="metric-card"><div style="font-weight: 900; font-size: 0.75em; color: #00ff00;">BUILDING_UTILIZATION</div><div style="font-size: 1.5em; font-weight: 900; color: #00ff00;">{util_val}</div></div>'
    f'<div class="metric-card" style="border-color: #ff00ff; box-shadow: 5px 5px 0px #ff00ff;"><div style="font-weight: 900; font-size: 0.75em; color: #ff00ff;">PATTERN_UTILIZATION</div><div style="font-size: 1.5em; font-weight: 900; color: #ff00ff;">{p_util:.1f}%</div></div>'
    f'<div class="metric-card" style="background-color: #ff00ff; border: none; box-shadow: 5px 5px 0px #ffffff;"><div style="font-weight: 900; font-size: 0.75em; color: #ffffff;">FINAL_STORAGE_CUBE</div><div style="font-size: 1.5em; font-weight: 900; color: #ffffff;">{rack_cf:,.0f}</div></div>'
    f'</div>'
)
st.markdown(receipt_html, unsafe_allow_html=True)

# --- SCENARIO LOG ---
with col_main:
    st.header("SCENARIO_LOG")
    st.markdown('<div class="warning-text">⚠️ Warning: Data is not persistent. Refreshing the browser will wipe the log.</div>', unsafe_allow_html=True)
    scen_name = st.text_input("SCENARIO NAME", placeholder="E.G. PLAN_ALPHA")
    st.markdown('<div id="PRECISION_SAVE_ZONE">', unsafe_allow_html=True)
    if st.button("SAVE SNAPSHOT"):
        if ready:
            st.session_state.history.append({"NAME": scen_name if scen_name else f"PLAN_{len(st.session_state.history)+1}", "AISLE": aisle_val, "BUILDING": util_val, "PATTERN": f"{p_util:.1f}%", "CUBE": f"{rack_cf:,.0f}"})
    st.markdown('</div>', unsafe_allow_html=True)
    if st.session_state.history:
        h = "<thead><tr><th>NAME</th><th>AISLE</th><th>BUILDING %</th><th>PATTERN %</th><th>CUBE</th></tr></thead>"
        rows = "".join([f'<tr><td>{e["NAME"]}</td><td>{e["AISLE"]}</td><td>{e["BUILDING"]}</td><td>{e["PATTERN"]}</td><td>{e["CUBE"]}</td></tr>' for e in st.session_state.history])
        st.markdown(f'<table class="terminal-table">{h}<tbody>{rows}</tbody></table>', unsafe_allow_html=True)
        if st.button("CLEAR LOG"): st.session_state.history = []; st.rerun()