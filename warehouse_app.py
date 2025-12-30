import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Warehouse Cube Optimizer", layout="wide")

# --- ADVANCED CSS FOR LEFT-ALIGNED MAIN & FIXED RIGHT SIDEBAR ---
st.markdown("""
    <style>
    /* 1. Anchor the main app container to the left */
    .block-container { 
        max-width: 100% !important; 
        padding-top: 2rem; 
        margin-left: 0 !important;
        margin-right: 0 !important;
        padding-left: 50px !important;
    }
    
    /* 2. Constrain the main content width */
    [data-testid="column"]:nth-child(1) {
        min-width: 850px !important;
        max-width: 850px !important;
    }

    /* 3. FIXED RIGHT SIDEBAR RECEIPT (100px from top) */
    .fixed-receipt-sidebar {
        position: fixed;
        top: 100px;
        right: 40px;
        width: 260px;
        background-color: #ffffff;
        border: 2px solid #333;
        padding: 20px;
        box-shadow: 10px 10px 0px #eee;
        font-family: 'Courier New', Courier, monospace;
        line-height: 1.4;
        font-size: 0.85em;
        z-index: 9999;
        color: #000;
    }

    .receipt-hr { border-top: 2px dashed #333; margin: 10px 0; }
    .receipt-total { font-size: 1.2em; font-weight: bold; color: #6a0dad; }
    
    /* Hide default Streamlit sidebar */
    [data-testid="stSidebar"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. USER INPUTS (VERTICAL FLOW) ---
col_main, _ = st.columns([850, 300])

with col_main:
    st.title("📦 Warehouse Optimizer")

    # Step 1: Building
    st.header("Step 1: Building Dimensions")
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1: b_l = st.number_input("Building Length (ft)", value=400.0)
    with b_col2: b_w = st.number_input("Building Width (ft)", value=200.0)
    with b_col3: clear_ht = st.number_input("Clear Height (ft)", value=30.0)
    st.divider()

    # Step 2: Clear Zones
    st.header("Step 2: Clear Zones")
    st.subheader("Speed Bays")
    sb_cols = st.columns(4)
    with sb_cols[0]: sb_l = st.checkbox("Left")
    with sb_cols[1]: sb_r = st.checkbox("Right")
    with sb_cols[2]: sb_t = st.checkbox("Top")
    with sb_cols[3]: sb_b = st.checkbox("Bottom")
    
    any_sb = any([sb_l, sb_r, sb_t, sb_b])
    sb_depth = st.number_input("Speed Bay Depth (ft)", value=75.0) if any_sb else 0.0

    st.subheader("Race Track")
    rt_cols = st.columns(4)
    with rt_cols[0]: rt_val_l = st.number_input("L Buffer (ft)", value=30.0)
    with rt_cols[1]: rt_val_r = st.number_input("R Buffer (ft)", value=30.0)
    with rt_cols[2]: rt_val_t = st.number_input("T Buffer (ft)", value=30.0)
    with rt_cols[3]: rt_val_b = st.number_input("B Buffer (ft)", value=30.0)
    st.divider()

    # Step 3: Columns
    st.header("Step 3: Column Specifications")
    c_cols = st.columns(4)
    with c_cols[0]: col_x = st.number_input("X Spacing (ft)", value=50.0)
    with c_cols[1]: col_y = st.number_input("Y Spacing (ft)", value=50.0)
    with c_cols[2]: col_w_in = st.number_input("Width (in)", value=12.0)
    with c_cols[3]: col_d_in = st.number_input("Depth (in)", value=12.0)
    col_w_ft, col_d_ft = col_w_in/12, col_d_in/12
    st.divider()

    # Step 4: Racking
    st.header("Step 4: Racking Optimization")
    r_cols = st.columns(4)
    with r_cols[0]: orientation = st.selectbox("Orientation", ["Vertical", "Horizontal"])
    with r_cols[1]: r_in = st.number_input("Rack Depth (in)", value=42.0)
    with r_cols[2]: f_in = st.number_input("Flue (in)", value=12.0)
    with r_cols[3]: min_aisle = st.number_input("Min Aisle (ft)", value=9.5)

    # --- MATH ENGINE ---
    r_ft, f_ft = r_in/12, f_in/12
    min_p = (r_ft*2) + f_ft + min_aisle
    target_grid = col_y if orientation == "Horizontal" else col_x
    mod_bay = int(target_grid / min_p) if target_grid >= min_p else 1
    opt_p = target_grid / mod_bay
    opt_a = opt_p - (r_ft*2) - f_ft

    c_min_x = sb_depth if sb_l else 0.0
    c_min_y = sb_depth if sb_b else 0.0
    c_max_x = b_l - (sb_depth if sb_r else 0.0)
    c_max_y = b_w - (sb_depth if sb_t else 0.0)
    r_min_x, r_max_x = max(c_min_x, rt_val_l), b_l - max((sb_depth if sb_r else 0.0), rt_val_r)
    r_min_y, r_max_y = max(c_min_y, rt_val_b), b_w - max((sb_depth if sb_t else 0.0), rt_val_t)

    # --- DIAGRAMS ---
    st.subheader("Building View")
    fig1 = go.Figure()
    fig1.add_shape(type="rect", x0=0, y0=0, x1=b_l, y1=b_w, line=dict(color="black", width=3))
    
    # Zones Highlights
    fig1.add_shape(type="rect", x0=rt_val_l, y0=rt_val_b, x1=b_l-rt_val_r, y1=b_w-rt_val_t, 
                  fillcolor="rgba(128,128,128,0.15)", line=dict(color="grey", dash="dot"))
    sb_fill = "rgba(255, 165, 0, 0.3)"
    if sb_l: fig1.add_shape(type="rect", x0=0, y0=0, x1=sb_depth, y1=b_w, fillcolor=sb_fill, line_width=0)
    if sb_r: fig1.add_shape(type="rect", x0=b_l-sb_depth, y0=0, x1=b_l, y1=b_w, fillcolor=sb_fill, line_width=0)
    if sb_t: fig1.add_shape(type="rect", x0=0, y0=b_w-sb_depth, x1=b_l, y1=b_w, fillcolor=sb_fill, line_width=0)
    if sb_b: fig1.add_shape(type="rect", x0=0, y0=0, x1=b_l, y1=sb_depth, fillcolor=sb_fill, line_width=0)

    rack_fp, row_count = 0.0, 0
    rack_color = "purple"
    
    if orientation == "Horizontal":
        steps = np.arange(c_min_y, b_w + 0.1, col_y)
        row_len = r_max_x - r_min_x
        for sy in steps:
            for m in range(mod_bay):
                fc = sy - (m * opt_p); r1, r2 = fc - f_ft/2 - r_ft, fc + f_ft/2
                if r1 >= r_min_y and r2+r_ft <= r_max_y:
                    fig1.add_shape(type="rect", x0=r_min_x, y0=r1, x1=r_max_x, y1=r1+r_ft, fillcolor=rack_color, opacity=0.4)
                    fig1.add_shape(type="rect", x0=r_min_x, y0=r2, x1=r_max_x, y1=r2+r_ft, fillcolor=rack_color, opacity=0.4)
                    rack_fp += row_len * r_ft * 2; row_count += 2
    else:
        steps = np.arange(c_min_x, b_l + 0.1, col_x)
        row_len = r_max_y - r_min_y
        for sx in steps:
            for m in range(mod_bay):
                fc = sx - (m * opt_p); r1, r2 = fc - f_ft/2 - r_ft, fc + f_ft/2
                if r1 >= r_min_x and r2+r_ft <= r_max_x:
                    fig1.add_shape(type="rect", x0=r1, y0=r_min_y, x1=r1+r_ft, y1=r_max_y, fillcolor=rack_color, opacity=0.4)
                    fig1.add_shape(type="rect", x0=r2, y0=r_min_y, x1=r2+r_ft, y1=r_max_y, fillcolor=rack_color, opacity=0.4)
                    rack_fp += row_len * r_ft * 2; row_count += 2

    for x in np.arange(c_min_x, c_max_x + 0.1, col_x):
        for y in np.arange(c_min_y, c_max_y + 0.1, col_y):
            fig1.add_shape(type="rect", x0=x-col_w_ft/2, y0=y-col_d_ft/2, x1=x+col_w_ft/2, y1=y+col_d_ft/2, fillcolor="red")
    
    fig1.update_layout(xaxis=dict(range=[-10, b_l+10], scaleanchor="y"), yaxis=dict(range=[-10, b_w+10]), width=800, height=400)
    st.plotly_chart(fig1)

    st.subheader("Pattern Detail")
    fig2 = go.Figure()
    for x in [0, col_x, col_x*2]:
        for y in [0, col_y, col_y*2]:
            fig2.add_shape(type="rect", x0=x-col_w_ft/2, y0=y-col_d_ft/2, x1=x+col_w_ft/2, y1=y+col_d_ft/2, fillcolor="red")
    
    if orientation == "Horizontal":
        for start_y in [col_y, col_y*2]:
            for m in range(mod_bay):
                fc = start_y - (m * opt_p); r1, r2 = fc - f_ft/2 - r_ft, fc + f_ft/2
                fig2.add_shape(type="rect", x0=0, y0=r1, x1=col_x*2, y1=r1+r_ft, fillcolor=rack_color, opacity=0.3)
                fig2.add_shape(type="rect", x0=0, y0=r2, x1=col_x*2, y1=r2+r_ft, fillcolor=rack_color, opacity=0.3)
    else:
        for start_x in [col_x, col_x*2]:
            for m in range(mod_bay):
                fc = start_x - (m * opt_p); r1, r2 = fc - f_ft/2 - r_ft, fc + f_ft/2
                fig2.add_shape(type="rect", x0=r1, y0=0, x1=r1+r_ft, y1=col_y*2, fillcolor=rack_color, opacity=0.3)
                fig2.add_shape(type="rect", x0=r2, y0=0, x1=r2+r_ft, y1=col_y*2, fillcolor=rack_color, opacity=0.3)

    fig2.update_layout(width=800, height=450, xaxis=dict(range=[-10, col_x*2+10], scaleanchor="y"), yaxis=dict(range=[-10, col_y*2+10]))
    st.plotly_chart(fig2)

# --- 2. FIXED SIDEBAR RECEIPT ---
total_rack_cube = rack_fp * clear_ht
st.markdown(f"""
    <div class="fixed-receipt-sidebar">
        <b>BLDG SUMMARY</b><br>
        Dim: {b_l}x{b_w}<br>
        Area: {(b_l*b_w):,.0f} sf<br>
        Cube: {(b_l*b_w*clear_ht):,.0f} cf
        <div class="receipt-hr"></div>
        <b>RACKING MATH</b><br>
        Rows: {row_count}<br>
        Row Len: {row_len:.1f} ft<br>
        Rack D: {r_ft:.2f} ft<br>
        Footprint: {rack_fp:,.0f} sf
        <div class="receipt-hr"></div>
        <b>TOTAL RACK CUBE</b><br>
        <div class="receipt-total">{total_rack_cube:,.0f} CF</div>
        <div class="receipt-hr"></div>
        <b>OPTIMIZATION</b><br>
        Opt Aisle: {opt_a:.2f} ft<br>
        Mods/Bay: {mod_bay}
    </div>
""", unsafe_allow_html=True)