import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Racking Layout Optimizer", layout="wide")

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
    if any([st.session_state.sb_l, st.session_state.sb_r, st.session_state.sb_t, st.session_state.sb_b]):
        if st.session_state.sb_depth == 0.0:
            st.session_state.sb_depth = (st.session_state.col_x * 1.5) if st.session_state.col_x > 0 else 60.0

# --- CSS: THEME & UI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800;900&family=Inter:wght@400;700&display=swap');
    .stApp { background-color: #000000; color: #ffffff; font-family: 'Inter', sans-serif; }
    * { border-radius: 0px !important; }
    h1 { color: #00f3ff !important; font-family: 'JetBrains Mono', monospace !important; font-weight: 900 !important; text-transform: uppercase; }
    h2, h3 { color: #ffffff !important; font-family: 'JetBrains Mono', monospace !important; font-weight: 600 !important; text-transform: uppercase; }
    .cyan-divider { border: none; border-top: 2px dotted #00f3ff; width: 100%; margin: 2rem 0; opacity: 0.6; }
    [data-testid="stWidgetLabel"] p { color: #00f3ff !important; font-family: 'JetBrains Mono', monospace !important; font-weight: 900; font-size: 0.85em !important; text-transform: uppercase; }
    [data-testid="stTooltipIcon"] svg { fill: #00f3ff !important; }
    div[data-testid="stTooltipContent"] { background-color: #1a1a1a !important; color: #ffffff !important; border: 1px solid #00f3ff !important; padding: 10px !important; }
    .st-b7 { background-color: rgb(26, 26, 26) !important; }
    .st-bb { color: #ffffff !important; }
    .st-ey { background-color: rgb(255, 0, 255) !important; border: 1px solid rgb(255, 0, 255) !important; }
    div[data-baseweb="input"] > div, .stNumberInput input, .stTextInput input { background-color: #1a1a1a !important; color: #ffffff !important; border: 1px solid #333 !important; font-family: 'JetBrains Mono', monospace !important; }
    input::placeholder { color: #ffffff !important; opacity: 1.0 !important; }
    .stButton > button { background-color: rgb(255, 0, 255) !important; color: #ffffff !important; font-family: 'JetBrains Mono', monospace !important; font-weight: 900 !important; text-transform: uppercase !important; border: none !important; box-shadow: 4px 4px 0px #ffffff !important; }
    #PRECISION_SAVE_ZONE button { margin-top: 10px !important; width: auto !important; }
    .terminal-table { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; margin-top: 20px; }
    .terminal-table th { color: #00f3ff; font-size: 0.75em; text-transform: uppercase; font-weight: 900; text-align: left; padding: 10px; border-bottom: 1px dashed #333; }
    .terminal-table td { color: #ffffff; font-size: 0.85em; padding: 10px; border-bottom: 1px solid #1a1a1a; }
    .fixed-receipt-sidebar { background-color: #000000; border: 2px solid #ffffff; padding: 20px; box-shadow: 10px 10px 0px #ff00ff; z-index: 9999; }
    @media (min-width: 1024px) { .fixed-receipt-sidebar { position: fixed; top: 80px; right: 20px; width: 325px; } }
    @media (max-width: 1023px) { .fixed-receipt-sidebar { position: relative; top: 0; right: 0; width: 100%; margin: 20px 0; } }
    .metric-card { background-color: #000000; border: 1px solid #00ff00; padding: 12px; margin-top: 15px; box-shadow: 5px 5px 0px #00ff00; }
    [data-testid="stSidebar"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

# --- USER INPUT SECTIONS ---
col_main, _ = st.columns([850, 310])

with col_main:
    st.title("RACKING_LAYOUT_OPTIMIZER")
    if st.button("CLEAR ALL FIELDS"):
        for key in classic_defaults: st.session_state[key] = classic_defaults[key]
        st.session_state.sb_depth = 0.0
        st.rerun()

    st.header("1. BUILDING DIMENSIONS")
    b_c = st.columns(3)
    b_l = b_c[0].number_input("Length (ft)", key="b_l", step=1.0, help="The total exterior wall length of your warehouse.")
    b_w = b_c[1].number_input("Width (ft)", key="b_w", step=1.0, help="The narrow way across the warehouse floor.")
    clear_ht = b_c[2].number_input("Clear Height (ft)", key="clear_ht", step=1.0, help="Vertical distance from floor to ceiling obstructions.")
    st.markdown('<div class="cyan-divider"></div>', unsafe_allow_html=True)

    st.header("2. CLEAR ZONES")
    sb_checks = st.columns(4)
    sb_l = sb_checks[0].checkbox("Speed Bay L", key="sb_l", on_change=update_sb_depth, help="Carve out traffic room on the left wall.")
    sb_r = sb_checks[1].checkbox("Speed Bay R", key="sb_r", on_change=update_sb_depth, help="Carve out traffic room on the right wall.")
    sb_t = sb_checks[2].checkbox("Speed Bay T", key="sb_t", on_change=update_sb_depth, help="Carve out traffic room on the top wall.")
    sb_b = sb_checks[3].checkbox("Speed Bay B", key="sb_b", on_change=update_sb_depth, help="Carve out traffic room on the bottom wall.")
    
    if any([st.session_state.sb_l, st.session_state.sb_r, st.session_state.sb_t, st.session_state.sb_b]):
        st.columns([1, 3])[0].number_input("Speed Bay Depth (ft)", key="sb_depth", step=1.0, help="Uniform depth for those staging lanes.")
    
    st.subheader("RACETRACK")
    rt_c = st.columns(4)
    rt_l = rt_c[0].number_input("L Buffer", key="rt_l", step=1.0, help="Safety gap from the racks to the left wall.")
    rt_r = rt_c[1].number_input("R Buffer", key="rt_r", step=1.0, help="Safety gap from the racks to the right wall.")
    rt_t = rt_c[2].number_input("T Buffer", key="rt_t", step=1.0, help="Safety gap from the racks to the top wall.")
    rt_b = rt_c[3].number_input("B Buffer", key="rt_b", step=1.0, help="Safety gap from the racks to the bottom wall.")
    
    rt_btns = st.columns([1.5, 1.5, 5])
    rt_btns[0].button("APPLY L. TO ALL", on_click=apply_l_to_all)
    rt_btns[1].button("APPLY COL. SPECS", on_click=apply_col_specs)
    st.markdown('<div class="cyan-divider"></div>', unsafe_allow_html=True)

    st.header("3. COLUMN SPECIFICATIONS")
    c_c = st.columns(4)
    col_x = c_c[0].number_input("X Span (ft)", key="col_x", step=1.0, help="Distance between column centers horizontally.")
    col_y = c_c[1].number_input("Y Span (ft)", key="col_y", step=1.0, help="Distance between column centers vertically.")
    col_w_ft, col_d_ft = c_c[2].number_input("Col W (in)", key="col_w", step=1.0, help="Width of the column post.")/12, c_c[3].number_input("Col D (in)", key="col_d", step=1.0, help="Depth of the column post.")/12
    st.markdown('<div class="cyan-divider"></div>', unsafe_allow_html=True)

    st.header("4. RACKING OPTIMIZATION")
    r_c = st.columns(4)
    orient = r_c[0].selectbox("Orientation", ["Vertical", "Horizontal"], key="orient", help="Which direction the rows travel.")
    r_in = r_c[1].number_input("Rack D (in)", key="r_in", step=1.0, help="Depth of one rack frame.")
    f_in = r_c[2].number_input("Flue (in)", key="f_in", step=1.0, help="Gap between back-to-back rack rows.")
    min_a = r_c[3].number_input("Min Aisle (ft)", key="min_a", step=1.0, help="Forklift turning room.")
    allow_single = st.checkbox("Allow single aisles", key="single", help="Distribute singles and doubles to maximize density and center aisles.")
    st.markdown('<div class="cyan-divider"></div>', unsafe_allow_html=True)

    # --- MATH ENGINE ---
    ready = b_l > 0 and b_w > 0 and col_x > 0 and col_y > 0 and r_in > 0
    if ready:
        r_ft, f_ft = r_in/12, f_in/12
        wall_l, wall_r = (st.session_state.sb_depth if sb_l else 0), (b_l - (st.session_state.sb_depth if sb_r else 0))
        wall_b, wall_t = (st.session_state.sb_depth if sb_b else 0), (b_w - (st.session_state.sb_depth if sb_t else 0))
        
        avail_x, avail_y = wall_r - wall_l, wall_t - wall_b
        if avail_x > 0 and avail_y > 0:
            num_spans_x, num_spans_y = int(avail_x / col_x), int(avail_y / col_y)
            slack_x, slack_y = avail_x - (num_spans_x * col_x), avail_y - (num_spans_y * col_y)
            vis_off_x, vis_off_y = wall_l + (slack_x / 2), wall_b + (slack_y / 2)

            def get_coords(limit, grid_start, step):
                coords = []
                grid = np.arange(grid_start, limit + 0.1, step)
                col_dim = col_d_ft if orient == "Horizontal" else col_w_ft
                eff_flue = max(f_ft, col_dim)

                for g in grid:
                    bay_end = g + step
                    # Phase 1: Determine Count & Types for this bay
                    bay_units = [] # List of rack depths (r_ft or r_ft*2+f_ft)
                    
                    # Add base column-rack
                    if allow_single:
                        bay_units.append(("S", r_ft))
                        remaining = step - (r_ft + min_a)
                    else:
                        bay_units.append(("D", r_ft * 2 + eff_flue))
                        remaining = step - (r_ft * 2 + eff_flue + min_a)
                    
                    # Fill remainder with doubles, then a single if room
                    while remaining >= (r_ft * 2 + f_ft + min_a):
                        bay_units.append(("D", r_ft * 2 + f_ft))
                        remaining -= (r_ft * 2 + f_ft + min_a)
                    if allow_single and remaining >= (r_ft + min_a):
                        bay_units.append(("S", r_ft))
                        remaining -= (r_ft + min_a)
                    
                    # Phase 2: Distribute Aisles Evenly
                    num_aisles = len(bay_units) # One aisle after every unit
                    distributed_aisle = min_a + (remaining / num_aisles) if num_aisles > 0 else min_a
                    
                    # Phase 3: Plot Coords
                    ptr = g
                    for i, (type, depth) in enumerate(bay_units):
                        if i == 0: # Handle first unit center-aligned to column
                            if type == "S":
                                coords.append((ptr - r_ft/2, ptr + r_ft/2))
                                ptr += (r_ft/2 + distributed_aisle)
                            else:
                                coords.append((ptr - eff_flue/2 - r_ft, ptr - eff_flue/2))
                                coords.append((ptr + eff_flue/2, ptr + eff_flue/2 + r_ft))
                                ptr += (eff_flue/2 + r_ft + distributed_aisle)
                        else:
                            if type == "S":
                                coords.append((ptr, ptr + r_ft))
                                ptr += (r_ft + distributed_aisle)
                            else:
                                coords.append((ptr, ptr + r_ft))
                                coords.append((ptr + r_ft + f_ft, ptr + r_ft * 2 + f_ft))
                                ptr += (r_ft * 2 + f_ft + distributed_aisle)
                
                return list(set(coords)), distributed_aisle

            raw_coords, final_aisle_val = get_coords(wall_t if orient == "Horizontal" else wall_r, vis_off_y if orient == "Horizontal" else vis_off_x, col_y if orient == "Horizontal" else col_x)
            r_min_x, r_max_x = max(wall_l, rt_l), min(wall_r, b_l - rt_r)
            r_min_y, r_max_y = max(wall_b, rt_b), min(wall_t, b_w - rt_t)
            unique_final = sorted([r for r in raw_coords if r[0] >= (r_min_y if orient == "Horizontal" else r_min_x) and r[1] <= (r_max_y if orient == "Horizontal" else r_max_x)])
            
            rack_cf = len(unique_final) * r_ft * (r_max_x - r_min_x if orient == "Horizontal" else r_max_y - r_min_y) * clear_ht
            build_util = (rack_cf / (b_l * b_w * clear_ht)) * 100
            
            sample_bay_coords, _ = get_coords(vis_off_y + (col_y if orient == "Horizontal" else col_x), vis_off_y if orient == "Horizontal" else vis_off_x, col_y if orient == "Horizontal" else col_x)
            p_util = ((len(sample_bay_coords) * r_ft * (st.session_state.col_y if orient == "Horizontal" else st.session_state.col_x)) / (col_x * col_y)) * 100
            best_aisle = final_aisle_val
        else: rack_cf, build_util, best_aisle, p_util, unique_final = 0, 0, 0, 0, []
    else: rack_cf, build_util, best_aisle, p_util, unique_final = 0, 0, 0, 0, []

    # 5. VISUALIZATIONS
    st.header("5. VISUALIZATIONS")
    st.subheader("BUILDING VIEW")
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=b_l, y1=b_w, line=dict(color="#ffffff", width=2))
    if ready:
        fig.add_shape(type="rect", x0=r_min_x, y0=r_min_y, x1=r_max_x, y1=r_max_y, line=dict(color="#00ff00", width=1.5, dash="dot"))
        for r in unique_final:
            x0, x1 = (r_min_x, r_max_x) if orient == "Horizontal" else (r[0], r[1])
            y0, y1 = (r[0], r[1]) if orient == "Horizontal" else (r_min_y, r_max_y)
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, fillcolor="#00ff00", opacity=0.3, line_width=0.5, line_color="#00ff00")
        for x in np.arange(vis_off_x if 'vis_off_x' in locals() else 0, wall_r + 0.1, col_x):
            for y in np.arange(vis_off_y if 'vis_off_y' in locals() else 0, wall_t + 0.1, col_y):
                fig.add_shape(type="rect", x0=x-col_w_ft/2, y0=y-col_d_ft/2, x1=x+col_w_ft/2, y1=y+col_d_ft/2, fillcolor="rgb(255, 0, 255)")
        if sb_l: fig.add_shape(type="line", x0=wall_l, y0=0, x1=wall_l, y1=b_w, line=dict(color="#ff00ff", width=2, dash="dot"))
        if sb_r: fig.add_shape(type="line", x0=wall_r, y0=0, x1=wall_r, y1=b_w, line=dict(color="#ff00ff", width=2, dash="dot"))
        if sb_b: fig.add_shape(type="line", x0=0, y0=wall_b, x1=b_l, y1=wall_b, line=dict(color="#ff00ff", width=2, dash="dot"))
        if sb_t: fig.add_shape(type="line", x0=0, y0=wall_t, x1=b_l, y1=wall_t, line=dict(color="#ff00ff", width=2, dash="dot"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=20, b=20), xaxis=dict(range=[-b_l*0.05, b_l*1.05], scaleanchor="y", scaleratio=1, showgrid=False, constrain="domain"), yaxis=dict(range=[-b_w*0.05, b_w*1.05], showgrid=False, constrain="domain"), height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ENGINEERING PATTERN DETAIL")
    fig2 = go.Figure()
    for x in [0, col_x, col_x*2]:
        for y in [0, col_y, col_y*2]:
            fig2.add_shape(type="rect", x0=x-col_w_ft/2, y0=y-col_d_ft/2, x1=x+col_w_ft/2, y1=y+col_d_ft/2, fillcolor="#ff00ff")
    if ready:
        limit_val = col_y*2 if orient == "Horizontal" else col_x*2
        p_coords, _ = get_coords(limit_val, 0, col_y if orient == "Horizontal" else col_x)
        for r in p_coords:
            if r[0] >= 0 and r[1] <= limit_val:
                px0, px1 = (0, col_x*2) if orient == "Horizontal" else (r[0], r[1])
                py0, py1 = (r[0], r[1]) if orient == "Horizontal" else (0, col_y*2)
                fig2.add_shape(type="rect", x0=px0, y0=py0, x1=px1, y1=py1, line=dict(color="#39FF14", width=1.5), fillcolor="rgba(57, 255, 20, 0.4)")
    fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", width=800, height=450, xaxis=dict(range=[-10, max(col_x*2, 10)+10], scaleanchor="y", showgrid=False), yaxis=dict(range=[-10, max(col_y*2, 10)+10], showgrid=False))
    st.plotly_chart(fig2, use_container_width=True)

# --- SIDEBAR RECEIPT ---
receipt_html = (
    f'<div class="fixed-receipt-sidebar">'
    f'<div style="font-weight: 900; color: #00f3ff; font-size: 0.8em; margin-bottom: 5px; font-family: JetBrains Mono;">BUILDING_METRICS</div>'
    f'<div style="color: #666; font-size: 0.85em; font-style: italic;">{b_l:,.0f}ft L × {b_w:,.0f}ft W</div>'
    f'<div style="font-weight: bold; color: white;">Area: {b_l*b_w:,.0f} SF</div>'
    f'<div style="margin: 15px 0; border-top: 1px dashed #333;"></div>'
    f'<div style="font-weight: 900; color: #00f3ff; font-size: 0.8em; margin-bottom: 5px; font-family: JetBrains Mono;">RACKING_DATA</div>'
    f'<div style="color: #666; font-size: 0.85em; font-style: italic;">{len(unique_final)} rows</div>'
    f'<div style="font-weight: bold; color: white;">Actual Aisle: {best_aisle:.2f} ft</div>'
    f'<div style="margin: 15px 0; border-top: 1px dashed #333;"></div>'
    f'<div class="metric-card"><div style="font-weight: 900; font-size: 0.75em; color: #00ff00;">BUILDING_UTILIZATION</div><div style="font-size: 1.5em; font-weight: 900; color: #00ff00;">{build_util:.1f}%</div></div>'
    f'<div class="metric-card" style="border-color: #ff00ff; box-shadow: 5px 5px 0px #ff00ff;"><div style="font-weight: 900; font-size: 0.75em; color: #ff00ff;">PATTERN_UTILIZATION</div><div style="font-size: 1.5em; font-weight: 900; color: #ff00ff;">{p_util:.1f}%</div></div>'
    f'<div class="metric-card" style="background-color: #ff00ff; border: none; box-shadow: 5px 5px 0px #ffffff;"><div style="font-weight: 900; font-size: 0.75em; color: #ffffff;">FINAL_STORAGE_CUBE</div><div style="font-size: 1.5em; font-weight: 900; color: #ffffff;">{rack_cf:,.0f}</div></div>'
    f'</div>'
)
with col_main: st.markdown(receipt_html, unsafe_allow_html=True)

# --- SCENARIO LOG ---
with col_main:
    st.header("SCENARIO_LOG")
    scen_name = st.text_input("SCENARIO NAME", placeholder="E.G. PLAN_ALPHA", help="Identify this setup in your log.")
    st.markdown('<div id="PRECISION_SAVE_ZONE">', unsafe_allow_html=True)
    if st.button("SAVE SNAPSHOT"):
        if ready:
            st.session_state.history.append({"NAME": scen_name if scen_name else f"PLAN_{len(st.session_state.history)+1}", "AISLE": f"{best_aisle:.2f} ft", "BUILD": f"{build_util:.1f}%", "PATTERN": f"{p_util:.1f}%", "CUBE": f"{rack_cf:,.0f}"})
    st.markdown('</div>', unsafe_allow_html=True)
    if st.session_state.history:
        h = "<thead><tr><th>SCENARIO NAME</th><th>AISLE WIDTH</th><th>BUILDING UTILIZATION</th><th>PATTERN UTILIZATION</th><th>TOTAL STORAGE CUBE</th></tr></thead>"
        r = "".join([f'<tr><td>{e["NAME"]}</td><td>{e["AISLE"]}</td><td>{e["BUILD"]}</td><td>{e["PATTERN"]}</td><td>{e["CUBE"]}</td></tr>' for e in st.session_state.history])
        st.markdown(f'<table class="terminal-table">{h}<tbody>{r}</tbody></table>', unsafe_allow_html=True)
        if st.button("CLEAR LOG"): st.session_state.history = []; st.rerun()