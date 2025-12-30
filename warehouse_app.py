import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Warehouse Cube Optimizer", layout="wide")

# --- CSS: INJECT STYLES ---
st.markdown("""
    <style>
    .block-container { 
        max-width: 100% !important; padding-top: 2rem; 
        margin-left: 0 !important; margin-right: 0 !important; 
        padding-left: 50px !important; 
    }
    [data-testid="column"]:nth-child(1) { min-width: 850px !important; max-width: 850px !important; }
    
    .fixed-receipt-sidebar { 
        position: fixed; top: 80px; right: 20px; width: 320px; 
        background-color: white; border: 2px solid #333; padding: 15px; 
        box-shadow: 8px 8px 0px #ddd; font-family: 'Courier New', Courier, monospace; 
        z-index: 9999; color: black; max-height: 90vh; overflow-y: auto;
    }
    [data-testid="stSidebar"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. USER INPUTS ---
col_main, _ = st.columns([850, 310])

with col_main:
    st.title("📦 Warehouse Optimizer")

    # STEP 1: DIMENSIONS
    st.header("Step 1: Building Dimensions")
    b_c1, b_c2, b_c3 = st.columns(3)
    with b_c1: b_l = st.number_input("Building Length (ft)", value=400.0, step=1.0)
    with b_c2: b_w = st.number_input("Building Width (ft)", value=200.0, step=1.0)
    with b_c3: clear_ht = st.number_input("Clear Height (ft)", value=30.0, step=1.0)
    st.divider()

    # STEP 2: CLEAR ZONES
    st.header("Step 2: Clear Zones")
    sb_cols = st.columns(4)
    with sb_cols[0]: sb_l = st.checkbox("Speed Bay Left")
    with sb_cols[1]: sb_r = st.checkbox("Speed Bay Right")
    with sb_cols[2]: sb_t = st.checkbox("Speed Bay Top")
    with sb_cols[3]: sb_b = st.checkbox("Speed Bay Bottom")
    sb_depth = st.number_input("Speed Bay Depth (ft)", value=75.0, step=1.0) if any([sb_l, sb_r, sb_t, sb_b]) else 0.0

    st.subheader("Race Track")
    rt_cols = st.columns(4)
    with rt_cols[0]: rt_val_l = st.number_input("L Buffer (ft)", value=30.0, step=1.0)
    with rt_cols[1]: rt_val_r = st.number_input("R Buffer (ft)", value=30.0, step=1.0)
    with rt_cols[2]: rt_val_t = st.number_input("T Buffer (ft)", value=30.0, step=1.0)
    with rt_cols[3]: rt_val_b = st.number_input("B Buffer (ft)", value=30.0, step=1.0)
    st.divider()

    # STEP 3: COLUMNS
    st.header("Step 3: Column Specifications")
    c_cols = st.columns(4)
    with c_cols[0]: col_x = st.number_input("X Spacing (ft)", value=40.0, step=1.0)
    with c_cols[1]: col_y = st.number_input("Y Spacing (ft)", value=40.0, step=1.0)
    with c_cols[2]: col_w_in = st.number_input("Col Width (in)", value=12.0, step=1.0)
    with c_cols[3]: col_d_in = st.number_input("Col Depth (in)", value=12.0, step=1.0)
    col_w_ft, col_d_ft = col_w_in/12, col_d_in/12
    st.divider()

    # STEP 4: RACKING
    st.header("Step 4: Racking Optimization")
    r_cols = st.columns(4)
    with r_cols[0]: orientation = st.selectbox("Orientation", ["Vertical", "Horizontal"])
    with r_cols[1]: r_in = st.number_input("Rack Depth (in)", value=42.0, step=1.0)
    with r_cols[2]: f_in = st.number_input("Flue (in)", value=12.0, step=1.0)
    with r_cols[3]: min_aisle = st.number_input("Min Aisle (ft)", value=10.0, step=1.0)
    
    allow_single_aisles = st.checkbox("Allow single aisles", value=False)

    # --- MATH ENGINE ---
    r_ft, f_ft = r_in/12, f_in/12
    col_dim = col_d_ft if orientation == "Horizontal" else col_w_ft
    eff_flue = max(f_ft, col_dim)
    bay_span = col_y if orientation == "Horizontal" else col_x

    def solve_bay(strat):
        w_anchor = r_ft if strat == "Single" else (r_ft * 2 + eff_flue)
        usable = bay_span - w_anchor
        # Calculate max possible doubles while keeping aisle >= min_aisle
        n_db = int(usable / ((r_ft*2) + f_ft + min_aisle))
        rem = usable - (n_db * ((r_ft*2) + f_ft))
        u_aisle = rem / (n_db + 1)
        if u_aisle < min_aisle and n_db > 0:
            n_db -= 1
            rem = usable - (n_db * ((r_ft*2) + f_ft))
            u_aisle = rem / (n_db + 1)
        return (2 if strat == "Double" else 1) + (n_db * 2), u_aisle, n_db

    # Strategy duel
    if allow_single_aisles and col_dim <= r_ft:
        c_db, a_db, n_db_val = solve_bay("Double")
        c_sg, a_sg, n_sg_val = solve_bay("Single")
        if c_sg > c_db:
            use_single_anchor, best_aisle, best_n_db = True, a_sg, n_sg_val
        else:
            use_single_anchor, best_aisle, best_n_db = False, a_db, n_db_val
    else:
        use_single_anchor = False
        _, best_aisle, best_n_db = solve_bay("Double")

    # Boundaries
    r_min_x = max(sb_depth if sb_l else 0, rt_val_l)
    r_max_x = b_l - max(sb_depth if sb_r else 0, rt_val_r)
    r_min_y = max(sb_depth if sb_b else 0, rt_val_b)
    r_max_y = b_w - max(sb_depth if sb_t else 0, rt_val_t)
    
    # Generate Coordinates
    final_coords = []
    grid = np.arange(0, (b_w if orientation == "Horizontal" else b_l) + 1, (col_y if orientation == "Horizontal" else col_x))
    for g in grid:
        if use_single_anchor:
            final_coords.append((g - r_ft/2, g + r_ft/2))
            curr = g + r_ft/2 + best_aisle
        else:
            final_coords.append((g - eff_flue/2 - r_ft, g - eff_flue/2))
            final_coords.append((g + eff_flue/2, g + eff_flue/2 + r_ft))
            curr = g + eff_flue/2 + r_ft + best_aisle
        for _ in range(best_n_db):
            final_coords.append((curr, curr + r_ft))
            final_coords.append((curr + r_ft + f_ft, curr + r_ft * 2 + f_ft))
            curr += (r_ft * 2 + f_ft + best_aisle)

    low_b, high_b = (r_min_y if orientation == "Horizontal" else r_min_x), (r_max_y if orientation == "Horizontal" else r_max_x)
    unique_final = sorted(list(set([r for r in final_coords if r[0] >= low_b and r[1] <= high_b])))

    # --- PLOTS ---
    st.subheader("Building View")
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=b_l, y1=b_w, line=dict(color="black", width=3))
    sb_c = "rgba(255, 165, 0, 0.3)"
    if sb_l: fig.add_shape(type="rect", x0=0, y0=0, x1=sb_depth, y1=b_w, fillcolor=sb_c, line_width=0)
    if sb_r: fig.add_shape(type="rect", x0=b_l-sb_depth, y0=0, x1=b_l, y1=b_w, fillcolor=sb_c, line_width=0)
    if sb_t: fig.add_shape(type="rect", x0=0, y0=b_w-sb_depth, x1=b_l, y1=b_w, fillcolor=sb_c, line_width=0)
    if sb_b: fig.add_shape(type="rect", x0=0, y0=0, x1=b_l, y1=sb_depth, fillcolor=sb_c, line_width=0)
    fig.add_shape(type="rect", x0=rt_val_l, y0=rt_val_b, x1=b_l-rt_val_r, y1=b_w-rt_val_t, line=dict(color="grey", dash="dot", width=2))
    for r in unique_final:
        if orientation == "Horizontal":
            fig1_x0, fig1_x1 = r_min_x, r_max_x
            fig1_y0, fig1_y1 = r[0], r[1]
        else:
            fig1_x0, fig1_x1 = r[0], r[1]
            fig1_y0, fig1_y1 = r_min_y, r_max_y
        fig.add_shape(type="rect", x0=fig1_x0, y0=fig1_y0, x1=fig1_x1, y1=fig1_y1, fillcolor="purple", opacity=0.4, line_width=1)

    for x in np.arange(0, b_l + 1, col_x):
        for y in np.arange(0, b_w + 1, col_y):
            in_sb = (sb_l and x < sb_depth) or (sb_r and x > b_l-sb_depth) or (sb_b and y < sb_depth) or (sb_t and y > b_w-sb_depth)
            if not in_sb:
                fig.add_shape(type="rect", x0=x-col_w_ft/2, y0=y-col_d_ft/2, x1=x+col_w_ft/2, y1=y+col_d_ft/2, fillcolor="red")
    fig.update_layout(width=800, height=400, xaxis=dict(range=[-10, b_l+10], scaleanchor="y"), yaxis=dict(range=[-10, b_w+10]))
    st.plotly_chart(fig)

# --- 2. RECEIPT CALCULATION ---
b_sf = b_l * b_w
b_cf = b_sf * clear_ht
r_len_val = (r_max_x - r_min_x if orientation == "Horizontal" else r_max_y - r_min_y)
rack_sf = len(unique_final) * r_ft * r_len_val
rack_cf = rack_sf * clear_ht
util = (rack_cf / b_cf) * 100 if b_cf > 0 else 0

# Use a single continuous string without line breaks to force markdown to render HTML correctly
receipt_html = (
    f'<div class="fixed-receipt-sidebar">'
    f'<div style="font-weight: bold; margin-bottom: 2px;">BUILDING METRICS</div>'
    f'<div style="color: #888; font-size: 0.85em; font-style: italic;">{b_l:,.1f}ft L × {b_w:,.1f}ft W</div>'
    f'<div style="font-weight: bold;">Area: {b_sf:,.0f} SF</div>'
    f'<div style="color: #888; font-size: 0.85em; font-style: italic; margin-top: 4px;">{b_sf:,.0f} SF × {clear_ht:,.1f}ft H</div>'
    f'<div style="font-weight: bold;">Volume: {b_cf:,.0f} CF</div>'
    f'<div style="border-top: 1px dashed #333; margin: 12px 0;"></div>'
    f'<div style="font-weight: bold; margin-bottom: 2px;">AISLE CONFIGURATION</div>'
    f'<div style="color: #888; font-size: 0.85em; font-style: italic;">Bay ({bay_span:,.1f}ft) - Anchor ({"Single" if use_single_anchor else "Double"})</div>'
    f'<div style="font-weight: bold;">Uniform Aisle: {best_aisle:.2f} ft</div>'
    f'<div style="color: #888; font-size: 0.85em; font-style: italic; margin-top: 4px;">Min Required: {min_aisle:,.1f} ft</div>'
    f'<div style="border-top: 1px dashed #333; margin: 12px 0;"></div>'
    f'<div style="font-weight: bold; margin-bottom: 2px;">RACKING CALCULATIONS</div>'
    f'<div style="color: #888; font-size: 0.85em; font-style: italic;">{len(unique_final)} rows × {r_ft:,.2f}ft D × {r_len_val:,.1f}ft L</div>'
    f'<div style="font-weight: bold;">Rack Footprint: {rack_sf:,.0f} SF</div>'
    f'<div style="color: #888; font-size: 0.85em; font-style: italic; margin-top: 4px;">{rack_sf:,.0f} SF × {clear_ht:,.1f}ft H</div>'
    f'<div style="font-weight: bold;">Rack Cube: {rack_cf:,.0f} CF</div>'
    f'<div style="border-top: 1px dashed #333; margin: 12px 0;"></div>'
    f'<div style="font-weight: bold; margin-bottom: 2px;">TOTAL CUBE UTILIZATION</div>'
    f'<div style="color: #888; font-size: 0.85em; font-style: italic;">{rack_cf:,.0f} CF / {b_cf:,.0f} CF</div>'
    f'<div style="font-weight: bold; color: #6a0dad; font-size: 1.2em;">{util:.1f}% Utilization</div>'
    f'<div style="background-color: #f9f0ff; padding: 10px; border: 1px solid #6a0dad; margin-top: 12px;">'
    f'<div style="font-weight: bold; font-size: 0.9em;">FINAL STORAGE CUBE</div>'
    f'<div style="font-size: 1.4em; font-weight: bold; color: #6a0dad;">{rack_cf:,.0f} CF</div>'
    f'</div>'
    f'</div>'
)

st.markdown(receipt_html, unsafe_allow_html=True)