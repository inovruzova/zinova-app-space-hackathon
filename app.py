import os
from datetime import datetime
import streamlit as st
import folium
from folium import raster_layers
from streamlit_folium import st_folium
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================================
# ENV + LLM CLIENT SETUP
# ============================================================================
load_dotenv()  # Load variables from .env

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# ============================================================================
# DUMMY STATIC DATA - Single Scenario
# ============================================================================
SCENARIO_DATA = {
    "scene_id": "SCENE_001",
    "scene_name": "Caspian",
    "date": "2020-07-04",
    "description": "Satellite observation of Caspian Sea region",
}

# Three zones placed offshore in the sea.
DANGER_ZONES = {
    "zones": [
        {
            "zone_id": "Z1",
            "lat": 40.20, 
            "lon": 49.80,
            "scene_id": "SCENARIO_001",
        },
        {
            "zone_id": "Z2",
            "lat": 40.05,
            "lon": 49.9,
            "scene_id": "SCENARIO_001",
        },
        {
            "zone_id": "Z3",
            "lat": 39.90,
            "lon": 50.00,
            "scene_id": "SCENARIO_001",
        },
    ]
}

# Dummy historical data per zone (for chatbot context)
HISTORY_DATA = {
    "Z1": [
        {"date": "2023-05-10", "area_km2": 1.8, "spill_id": "H1"},
        {"date": "2023-08-21", "area_km2": 2.2, "spill_id": "H2"},
        {"date": "2024-01-03", "area_km2": 1.1, "spill_id": "H3"},
    ],
    "Z2": [
        {"date": "2022-11-02", "area_km2": 3.0, "spill_id": "H4"},
    ],
    "Z3": [
        {"date": "2023-02-15", "area_km2": 0.9, "spill_id": "H5"},
        {"date": "2023-09-10", "area_km2": 1.2, "spill_id": "H6"},
    ],
}

ZONE_OVERLAYS = {
    "Z1": {
        "image": "./overlays/overlay.jpg",
        "bounds": [[40.15, 49.70], [40.25, 49.90]],
        "opacity": 0.8,
    },
    "Z2": {
        "image": "./overlays/overlay_2.jpg",
        "bounds": [[40.00, 49.80], [40.10, 50.00]],
        "opacity": 0.8,
    },
    "Z3": {
        "image": "./overlays/overlay_3.jpg",
        "bounds": [[39.85, 49.90], [39.95, 50.10]],
        "opacity": 0.8,
    },
}

# ============================================================================
# Helper Functions - Data & LLM
# ============================================================================

def get_spill_geojson(zone_id: str):
    """
    Return dummy "spills" as feature collections WITHOUT drawing polygons.
    We only use properties for:
      - cleanup status
      - info box text
      - assistant context
    Geometry is not used for visualization anymore.
    """
    spill_data = {
        "Z1": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "spill_id": "S1",
                        "area_km2": 2.5,
                        "oil_type": "crude",
                        "thickness_class": "thick",
                        "confidence": 0.92,
                    },
                    # Geometry kept minimal, not used for drawing
                    "geometry": {
                        "type": "Point",
                        "coordinates": [49.70, 40.45],
                    },
                },
            ],
        },
        "Z2": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "spill_id": "S2",
                        "area_km2": 3.8,
                        "oil_type": "crude",
                        "thickness_class": "medium",
                        "confidence": 0.88,
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [50.10, 40.40],
                    },
                },
            ],
        },
        "Z3": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "spill_id": "S3",
                        "area_km2": 1.5,
                        "oil_type": "crude",
                        "thickness_class": "thin",
                        "confidence": 0.79,
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [50.60, 40.32],
                    },
                },
            ],
        },
    }
    return spill_data.get(zone_id, {"type": "FeatureCollection", "features": []})


def get_zone_meta(zone_id: str):
    for z in DANGER_ZONES["zones"]:
        if z["zone_id"] == zone_id:
            return z
    return None


def get_history_summary(zone_id: str) -> str:
    events = HISTORY_DATA.get(zone_id, [])
    if not events:
        return "No historical spills recorded for this zone."

    total_events = len(events)
    total_area = sum(e["area_km2"] for e in events)
    max_area = max(e["area_km2"] for e in events)
    latest = max(events, key=lambda e: e["date"])

    return (
        f"This zone has {total_events} historical spills, "
        f"total affected area ~{total_area:.1f} km². "
        f"Largest historical spill ~{max_area:.1f} km². "
        f"Most recent spill on {latest['date']} with area ~{latest['area_km2']:.1f} km²."
    )


def get_ai_response(zone_meta, spill_props, history_text: str, user_question: str) -> str:
    """Call LLM API with contextual prompt."""
    if client is None:
        return (
            "⚠️ LLM API key not configured. "
            "Set OPENAI_API_KEY in your environment to enable the assistant."
        )

    zone_name = zone_meta.get("name", zone_meta.get("zone_id", "Unknown zone"))
    scene_id = zone_meta.get("scene_id", "Unknown scene")
    lat = zone_meta.get("lat")
    lon = zone_meta.get("lon")

    spill_id = spill_props.get("spill_id", "Unknown")
    thickness = spill_props.get("thickness_class", "unknown")
    area = spill_props.get("area_km2", 0)
    confidence = spill_props.get("confidence", 0)
    oil_type = spill_props.get("oil_type", "unknown")

    system_prompt = (
        "You are an expert assisting operators with offshore oil spill analysis. "
        "Use the provided zone, spill attributes, and historical patterns to infer "
        "likely sources, risk level, and recommended actions. Be concise but specific "
        "and avoid inventing data not implied by the context."
    )

    user_context = f"""
Zone context:
- Name: {zone_name}
- Zone ID: {zone_meta.get('zone_id')}
- Scene ID: {scene_id}
- Approximate coordinates: {lat} N, {lon} E

Current spill:
- Spill ID: {spill_id}
- Oil type: {oil_type}
- Area: {area} km²
- Thickness class: {thickness}
- Detection confidence: {confidence:.0%}

Historical pattern summary:
{history_text}

Operator question:
{user_question}

Respond as an expert spill analyst. Include:
- Interpretation of historical trend
- Likely source/risk drivers
- Risk level (LOW / MEDIUM / HIGH)
- 2–3 concrete recommended actions.
    """.strip()

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error calling LLM API: {e}"


def init_cleanup_status() -> dict:
    """
    Initialize cleanup status for all spills.
    Status: 'idle' | 'cleaning' | 'done'
    """
    status = {}
    for zone in DANGER_ZONES["zones"]:
        features = get_spill_geojson(zone["zone_id"]).get("features", [])
        for f in features:
            sid = f["properties"].get("spill_id")
            if sid:
                status[sid] = "idle"
    return status


def get_spill_color(thickness: str, status: str) -> str:
    """
    Decide polygon color based on spill thickness and cleanup status.
    NOTE: Kept for potential future visualizations, but not used now
    since we no longer draw polygons.
    """
    if status == "done":
        return "green"
    if status == "cleaning":
        return "blue"

    base_colors = {"thick": "red", "medium": "orange", "thin": "yellow"}
    return base_colors.get(thickness, "gray")

# ============================================================================
# Page configuration
# ============================================================================
st.set_page_config(
    page_title="ZINova",
    page_icon="./logo/ZINova.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state
if "selected_zone_id" not in st.session_state:
    st.session_state.selected_zone_id = None
if "selected_spill_id" not in st.session_state:
    st.session_state.selected_spill_id = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "cleanup_status" not in st.session_state:
    st.session_state.cleanup_status = init_cleanup_status()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.image("./logo/ZINova.png", width="content")

    st.info(
        "**How to use:**\n"
        "1. Click on red danger zone markers to zoom in\n"
        "2. View overlay and info box on the map\n"
        "3. Use cleanup control near the map\n"
        "4. Use the assistant panel to analyze a spill"
    )

    st.subheader("Current Scenario")
    st.write(f"**Scene:** {SCENARIO_DATA['scene_id']}")
    st.write(f"**Date:** {SCENARIO_DATA['date']}")
    st.caption(SCENARIO_DATA['description'])

    st.divider()

    st.subheader("Zone Selection")
    zone_options = ["None"] + [
        f"{z['zone_id']}" for z in DANGER_ZONES["zones"]
    ]
    current_selection = st.session_state.selected_zone_id or "None"
    idx = 0
    if current_selection != "None":
        for i, z in enumerate(DANGER_ZONES["zones"]):
            if z["zone_id"] == st.session_state.selected_zone_id:
                idx = i + 1
                break

    selected_zone_option = st.selectbox(
        "Select Danger Zone",
        options=zone_options,
        index=idx,
        help="Select from dropdown or click on the map",
    )

    new_zone_id = None
    if selected_zone_option != "None":
        new_zone_id = selected_zone_option.split(" - ")[0]

    if new_zone_id != st.session_state.selected_zone_id:
        st.session_state.selected_zone_id = new_zone_id
        st.session_state.selected_spill_id = None
        st.session_state.chat_messages = []
        st.rerun()

    st.divider()

    # 🧹 Cleanup Device Status
    with st.expander("🧹 Cleanup Device Status", expanded=False):
        rows = []
        for zone in DANGER_ZONES["zones"]:
            zid = zone["zone_id"]
            features = get_spill_geojson(zid).get("features", [])
            for f in features:
                sid = f["properties"].get("spill_id", "Unknown")
                status_val = st.session_state.cleanup_status.get(sid, "idle")
                rows.append(
                    {
                        "Zone": zid,
                        "Spill ID": sid,
                        "Status": status_val.capitalize(),
                    }
                )
        if rows:
            st.table(rows)
        else:
            st.write("No spills found in scenario.")

# ============================================================================
# MAIN TITLE
# ============================================================================
st.title("Detect, Analyze, Clean.")
st.markdown(
    """
This platform visualizes detected oil spills (via overlays) and provides AI-driven insights
about potential sources, risk, historical patterns, with decision of clean-up actions.
"""
)
st.markdown("---")

# ============================================================================
# Layout: Map (col1) + Assistant (col2) ALWAYS
# ============================================================================
col1, col2 = st.columns([2, 1])

# ============================================================================
# MAP COLUMN
# ============================================================================
with col1:
    st.subheader("🛰️ Interactive Satellite Map - Caspian Sea Danger Zones")
    st.markdown("---")

    # Determine map center
    if st.session_state.selected_zone_id:
        zm = get_zone_meta(st.session_state.selected_zone_id)
        if zm:
            map_center = [zm["lat"], zm["lon"]]
            zoom_level = 10
        else:
            map_center = [40.4, 50.0]
            zoom_level = 8
    else:
        map_center = [40.4, 50.0]
        zoom_level = 8

    m = folium.Map(
        location=map_center,
        zoom_start=zoom_level,
        tiles="Esri.WorldImagery",
    )

    # Draw info box + overlays for selected zone (no polygons)
    if st.session_state.selected_zone_id:
        zone_id = st.session_state.selected_zone_id
        spill_geojson = get_spill_geojson(zone_id)
        features = spill_geojson.get("features", [])

        if features:
            is_overlay_zone = zone_id in ZONE_OVERLAYS
            lines = []

            # Build info box text (no polygon drawing)
            for ft in features:
                p = ft["properties"]
                sid = p.get("spill_id", "Unknown")
                th = p.get("thickness_class", "unknown")
                conf = p.get("confidence", 0)
                oil_type = p.get("oil_type", "unknown")
                area_km2 = p.get("area_km2", 0)
                status_val = st.session_state.cleanup_status.get(sid, "idle")

                lines.append(
                    f"<b>{sid}</b>: {oil_type}, {area_km2} km², {th}, "
                    f"{conf:.0%} conf., status: {status_val}"
                )

            # Info box (stays the same for all zones)
            info_html = "<br>".join(lines)
            info_html = f"""
            <div style="
                background-color: rgba(0, 0, 0, 0.75);
                color: white;
                padding: 8px 10px;
                border-radius: 8px;
                font-size: 11px;
                min-width: 180px;
                max-width: 260px;
                display: inline-block;
                box-shadow: 0 0 6px rgba(0,0,0,0.5);
            ">
                <div style="font-weight: 600; margin-bottom: 4px;">
                    Spills in {zone_id}
                </div>
                {info_html}
            </div>
            """

            label_lat = map_center[0] + 0.10
            label_lon = map_center[1] + 0.10

            folium.Marker(
                location=[label_lat, label_lon],
                icon=folium.DivIcon(html=info_html),
            ).add_to(m)

            # 🖼️ If this zone has an overlay configured, add PNG on top
            if is_overlay_zone:
                cfg = ZONE_OVERLAYS[zone_id]
                raster_layers.ImageOverlay(
                    name=f"{zone_id} overlay",
                    image=cfg["image"],
                    bounds=cfg["bounds"],
                    opacity=cfg.get("opacity", 0.8),
                    interactive=False,
                    cross_origin=False,
                    zindex=3,
                ).add_to(m)

    # Danger zone markers
    for zone in DANGER_ZONES["zones"]:
        is_sel = st.session_state.selected_zone_id == zone["zone_id"]
        marker_color = "darkred" if is_sel else "red"

        popup_html = f"""
        <div style="font-family: Arial; min-width: 150px;">
            <p style="margin: 3px 0;"><b>Zone ID:</b> {zone['zone_id']}</p>
            <p style="margin: 3px 0;"><b>Scene:</b> {zone['scene_id']}</p>
            <p style="margin: 3px 0;"><b>Coordinates:</b> {zone['lat']:.2f}°N, {zone['lon']:.2f}°E</p>
            <p style="margin: 5px 0; color: #ccc; font-size: 0.9em;">Click to zoom and view overlay</p>
        </div>
        """

        folium.Marker(
            location=[zone["lat"], zone["lon"]],
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=zone["zone_id"],
            icon=folium.Icon(color=marker_color, icon="exclamation-triangle", prefix="fa"),
        ).add_to(m)

    map_data = st_folium(
        m,
        width=None,
        height=400,
        returned_objects=["last_object_clicked"],
        key="caspian_map",
    )

    # Click → select zone
    if map_data and map_data.get("last_object_clicked"):
        clat = map_data["last_object_clicked"]["lat"]
        clon = map_data["last_object_clicked"]["lng"]
        min_dist = float("inf")
        closest = None
        for zone in DANGER_ZONES["zones"]:
            d = ((zone["lat"] - clat) ** 2 + (zone["lon"] - clon) ** 2) ** 0.5
            if d < min_dist:
                min_dist = d
                closest = zone
        if closest and min_dist < 0.2:
            st.session_state.selected_zone_id = closest["zone_id"]
            st.session_state.selected_spill_id = None
            st.session_state.chat_messages = []
            st.rerun()

    # CLEAN-UP CONTROL NEAR THE MAP
    if st.session_state.selected_zone_id:
        spill_geojson = get_spill_geojson(st.session_state.selected_zone_id)
        features = spill_geojson.get("features", [])

        if features:
            st.caption("🧹 Cleanup control for selected zone")

            cl_left, cl_right = st.columns([2, 1])

            with cl_left:
                spill_labels, spill_ids = [], []
                for f in features:
                    p = f["properties"]
                    sid = p.get("spill_id", "Unknown")
                    label = f"{sid} | {p.get('thickness_class','unknown')} | {p.get('area_km2',0)} km²"
                    spill_labels.append(label)
                    spill_ids.append(sid)

                if st.session_state.selected_spill_id not in spill_ids:
                    st.session_state.selected_spill_id = spill_ids[0]

                default_idx = spill_ids.index(st.session_state.selected_spill_id)

                selected_spill_label = st.selectbox(
                    "Spill for cleanup",
                    options=spill_labels,
                    index=default_idx,
                    key="cleanup_spill_select",
                    help="Choose which spill to send the cleanup device to.",
                    label_visibility="collapsed",
                )
                st.session_state.selected_spill_id = spill_ids[spill_labels.index(selected_spill_label)]

            with cl_right:
                current_status = st.session_state.cleanup_status.get(
                    st.session_state.selected_spill_id, "idle"
                )

                if current_status in ["idle"]:
                    btn_label = "Clean This Spill"
                elif current_status in ["cleaning"]:
                    btn_label = "Cleaning..."
                else:
                    btn_label = "Already Cleaned"

                disabled = current_status == "done"

                if st.button(btn_label, use_container_width=True, disabled=disabled):
                    if current_status == "idle":
                        st.session_state.cleanup_status[st.session_state.selected_spill_id] = "cleaning"
                        st.success(
                            f"Simulated cleanup device dispatched for spill "
                            f"{st.session_state.selected_spill_id} in zone {st.session_state.selected_zone_id}."
                        )
                    elif current_status == "cleaning":
                        st.session_state.cleanup_status[st.session_state.selected_spill_id] = "done"
                        st.success(
                            f"Spill {st.session_state.selected_spill_id} marked as cleaned."
                        )
                    st.rerun()

            st.caption(
                f"Status for {st.session_state.selected_spill_id}: "
                f"**{st.session_state.cleanup_status.get(st.session_state.selected_spill_id, 'idle').capitalize()}**"
            )

# ============================================================================
# CHAT COLUMN (assistant always visible)
# ============================================================================
with col2:
    st.subheader("🤖 Assistant")
    st.markdown("---")

    zone_id = st.session_state.selected_zone_id
    if not zone_id:
        st.info("Select a danger zone on the map to enable the AI assistant.")
    else:
        spill_geojson = get_spill_geojson(zone_id)
        features = spill_geojson.get("features", [])

        if not features:
            st.info("No spills detected in this zone. AI assistant is disabled.")
        else:
            # Spill selection for AI reasoning
            spill_labels, spill_ids = [], []
            for f in features:
                p = f["properties"]
                sid = p.get("spill_id", "Unknown")
                label = f"{sid} | {p.get('thickness_class','unknown')} | {p.get('area_km2',0)} km²"
                spill_labels.append(label)
                spill_ids.append(sid)

            if st.session_state.selected_spill_id not in spill_ids:
                st.session_state.selected_spill_id = spill_ids[0]

            default_idx = spill_ids.index(st.session_state.selected_spill_id)
            selected_spill_label = st.selectbox(
                "Spill Focus",
                options=spill_labels,
                index=default_idx,
                help="AI will analyze this spill.",
                key="assistant_spill_select",
            )
            st.session_state.selected_spill_id = spill_ids[spill_labels.index(selected_spill_label)]

            # Build context
            sel_feature = next(
                f for f in features
                if f["properties"].get("spill_id") == st.session_state.selected_spill_id
            )
            spill_props = sel_feature["properties"]
            zone_meta = get_zone_meta(zone_id)
            history_text = get_history_summary(zone_id)

            # Chat container with fixed height & scroll
            with st.container(height=400, border=True):
                if not st.session_state.chat_messages:
                    st.caption("Ask a question about this spill or area to get started.")

                for msg in st.session_state.chat_messages:
                    role = msg["role"]
                    content = msg["content"]

                    if role == "user":
                        with st.chat_message("user"):
                            st.markdown(content)
                    else:
                        with st.chat_message("assistant"):
                            st.markdown(content)

            # Chat input
            user_query = st.chat_input("Ask about this spill or area...")
            if user_query:
                st.session_state.chat_messages.append({"role": "user", "content": user_query})
                ai_answer = get_ai_response(zone_meta, spill_props, history_text, user_query)
                st.session_state.chat_messages.append({"role": "assistant", "content": ai_answer})
                st.rerun()

            with st.expander("Historical Spill Pattern (used for AI reasoning)", expanded=False):
                st.write(history_text)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption(
    f"Scene: {SCENARIO_DATA['scene_id']} | Date: {SCENARIO_DATA['date']} | "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)