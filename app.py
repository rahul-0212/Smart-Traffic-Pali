import streamlit as st
import pandas as pd
import pyodbc
import plotly.express as px
import joblib
from pathlib import Path

# ==========================================
# 1. PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Smart Traffic Pali | AI Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. PROFESSIONAL DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: #101d35;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    font-size: 32px;
    font-weight: 800;
    color: #10213a;
    margin-bottom: 4px;
}

.sub-title {
    color: #64748b;
    font-size: 15px;
}

.hero {
    background: linear-gradient(120deg, #10213a, #174b75);
    padding: 28px;
    border-radius: 16px;
    color: white;
    margin-bottom: 20px;
}

.hero h1 {
    color: white;
    font-size: 30px;
    margin-bottom: 8px;
}

.hero p {
    color: #dbeafe;
    font-size: 15px;
}

.section-title {
    color: #10213a;
    font-size: 23px;
    font-weight: 750;
    margin-top: 12px;
}

[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
}

[data-testid="stMetricLabel"] {
    color: #64748b;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #10213a;
    font-weight: 800;
}

div[data-testid="stTabs"] button {
    font-weight: 700;
}

.stDownloadButton button {
    background: #174b75;
    color: white;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. DATABASE CONNECTION
# ==========================================

@st.cache_data(ttl=60)
def load_data():

    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=RAHUL\\MSSQL;"
        "DATABASE=SmartTrafficPali;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )

    conn = pyodbc.connect(connection_string)

    query = """
    SELECT
        [timestamp],
        city,
        road_segment_id,
        road_name,
        current_speed_kmph,
        free_flow_speed_kmph,
        speed_ratio,
        congestion_level,
        data_source,
        is_simulated
    FROM TrafficData
    """

    data = pd.read_sql(query, conn)
    conn.close()

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    data["current_speed_kmph"] = pd.to_numeric(
        data["current_speed_kmph"], errors="coerce"
    )

    data["free_flow_speed_kmph"] = pd.to_numeric(
        data["free_flow_speed_kmph"], errors="coerce"
    )

    return data


# ==========================================
# 4. SIDEBAR
# ==========================================

st.sidebar.markdown("## 🚦 SMART TRAFFIC")
st.sidebar.caption("PALI • RAJASTHAN")

if st.sidebar.button("🔄 Refresh Dashboard", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

try:
    df = load_data()

except Exception as e:
    st.error("Database connection failed!")
    st.code(str(e))
    st.stop()

if df.empty:
    st.warning("No traffic records found.")
    st.stop()

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Traffic Filters")

road_options = ["All Roads"] + sorted(
    df["road_name"].dropna().unique().tolist()
)

selected_road = st.sidebar.selectbox(
    "Select Road",
    road_options
)

congestion_options = ["All Levels"] + sorted(
    df["congestion_level"].dropna().unique().tolist()
)

selected_congestion = st.sidebar.selectbox(
    "Congestion Level",
    congestion_options
)

filtered_df = df.copy()

if selected_road != "All Roads":
    filtered_df = filtered_df[
        filtered_df["road_name"] == selected_road
    ]

if selected_congestion != "All Levels":
    filtered_df = filtered_df[
        filtered_df["congestion_level"] == selected_congestion
    ]

st.sidebar.divider()
st.sidebar.caption("B.Tech Final Year Project")
st.sidebar.caption("AI • Machine Learning • SQL Server")


# ==========================================
# 5. HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <h1>🚦 Smart Traffic Intelligence</h1>
    <p>AI-Based Traffic Prediction & Congestion Management System</p>
    <p>📍 Pali, Rajasthan | Real-time Dashboard Interface</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Data transparency: The current Pali traffic records are "
    "simulated demo data, not actual field observations."
)


# ==========================================
# 6. OVERVIEW METRICS
# ==========================================

st.markdown(
    '<div class="section-title">📊 Traffic Overview</div>',
    unsafe_allow_html=True
)

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

total_records = len(filtered_df)

average_speed = filtered_df[
    "current_speed_kmph"
].mean()

average_free_speed = filtered_df[
    "free_flow_speed_kmph"
].mean()

high_congestion = (
    filtered_df["congestion_level"] == "High"
).sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "📋 Total Records",
    f"{total_records:,}"
)

col2.metric(
    "🚗 Average Speed",
    f"{average_speed:.2f} KM/H"
)

col3.metric(
    "🛣️ Free Flow Speed",
    f"{average_free_speed:.2f} KM/H"
)

col4.metric(
    "⚠️ High Congestion",
    f"{high_congestion:,}"
)

st.divider()


# ==========================================
# 7. DASHBOARD TABS
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Overview",
    "🤖 AI Prediction",
    "📈 Analytics",
    "📋 Traffic Records"
])


# ==========================================
# TAB 1 - OVERVIEW
# ==========================================

with tab1:

    st.subheader("Traffic Situation Overview")

    left, right = st.columns(2)

    congestion_counts = (
        filtered_df["congestion_level"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .reset_index()
    )

    congestion_counts.columns = [
        "Congestion Level", "Records"
    ]

    with left:

        fig_pie = px.pie(
            congestion_counts,
            names="Congestion Level",
            values="Records",
            hole=0.55,
            title="Congestion Distribution",
            color="Congestion Level",
            color_discrete_map={
                "Low": "#16a34a",
                "Medium": "#f59e0b",
                "High": "#dc2626"
            }
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    with right:

        fig_bar = px.bar(
            congestion_counts,
            x="Congestion Level",
            y="Records",
            title="Congestion Records",
            text="Records",
            color="Congestion Level",
            color_discrete_map={
                "Low": "#16a34a",
                "Medium": "#f59e0b",
                "High": "#dc2626"
            }
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

    st.subheader("Road-wise Average Speed")

    road_data = (
        filtered_df.groupby("road_name")["current_speed_kmph"]
        .mean()
        .reset_index()
    )

    fig_road = px.bar(
        road_data,
        x="road_name",
        y="current_speed_kmph",
        text_auto=".2f",
        title="Average Speed by Road",
        labels={
            "road_name": "Road",
            "current_speed_kmph": "Speed (KM/H)"
        }
    )

    st.plotly_chart(
        fig_road,
        use_container_width=True
    )


# ==========================================
# TAB 2 - AI PREDICTION
# ==========================================

with tab2:

    st.subheader("🤖 AI Traffic Speed Prediction")

    st.caption(
        "Experimental predictions using models trained "
        "on the public METR-LA benchmark dataset."
    )

    available_roads = sorted(
        df["road_name"].dropna().unique().tolist()
    )

    if selected_road == "All Roads":

        prediction_road = st.selectbox(
            "Choose Road for AI Prediction",
            available_roads,
            key="ai_road"
        )

    else:
        prediction_road = selected_road

    prediction_data = df[
        df["road_name"] == prediction_road
    ].copy()

    prediction_data = prediction_data.sort_values(
        "timestamp"
    )

    last_12 = prediction_data[
        "current_speed_kmph"
    ].dropna().tail(12)

    free_flow_speed = prediction_data[
        "free_flow_speed_kmph"
    ].dropna().median()

    if len(last_12) < 12:

        st.warning(
            "At least 12 valid readings are required."
        )

    elif pd.isna(free_flow_speed) or free_flow_speed <= 0:

        st.error("Valid free-flow speed is not available.")

    else:

        latest_time = prediction_data[
            prediction_data["current_speed_kmph"].notna()
        ]["timestamp"].iloc[-1]

        KMH_TO_MPH = 1 / 1.60934
        MPH_TO_KMH = 1.60934

        readings = [
            speed * KMH_TO_MPH
            for speed in last_12.to_list()
        ]

        features_data = {
            "hour": latest_time.hour,
            "day": latest_time.dayofweek,
            "lag_1": readings[-1],
            "lag_2": readings[-2],
            "lag_3": readings[-3],
            "lag_6": readings[-6],
            "lag_12": readings[-12]
        }

        X = pd.DataFrame([features_data])

        try:

            predictions = {}
            congestion_predictions = {}

            for steps in [1, 3, 6]:

                model_file = (
                    Path("models") /
                    f"model_{steps}_steps.joblib"
                )

                saved_data = joblib.load(model_file)

                model = saved_data["model"]
                features = saved_data["features"]

                prediction_mph = model.predict(
                    X[features]
                )[0]

                prediction_kmph = (
                    float(prediction_mph) * MPH_TO_KMH
                )

                horizon = saved_data["horizon"]

                predictions[horizon] = round(
                    prediction_kmph, 2
                )

                speed_ratio = (
                    prediction_kmph / free_flow_speed
                )

                if speed_ratio <= 0.35:
                    congestion = "High"

                elif speed_ratio <= 0.65:
                    congestion = "Medium"

                else:
                    congestion = "Low"

                congestion_predictions[horizon] = {
                    "speed": round(prediction_kmph, 2),
                    "speed_ratio": round(speed_ratio, 3),
                    "congestion": congestion
                }

            st.markdown(
                f"### 🛣️ Selected Road: {prediction_road}"
            )

            st.caption(
                f"Reference free-flow speed: "
                f"{free_flow_speed:.2f} KM/H"
            )

            st.markdown("#### Predicted Traffic Speed")

            p1, p2, p3 = st.columns(3)

            p1.metric(
                "5 Minutes",
                f"{predictions['5 Minutes']:.2f} KM/H"
            )

            p2.metric(
                "15 Minutes",
                f"{predictions['15 Minutes']:.2f} KM/H"
            )

            p3.metric(
                "30 Minutes",
                f"{predictions['30 Minutes']:.2f} KM/H"
            )

            st.markdown("#### 🚦 Future Congestion Status")

            c1, c2, c3 = st.columns(3)

            forecast_list = [
                ("5 Minutes", c1),
                ("15 Minutes", c2),
                ("30 Minutes", c3)
            ]

            for horizon, column in forecast_list:

                result = congestion_predictions[horizon]

                column.metric(
                    horizon,
                    result["congestion"]
                )

                column.caption(
                    f"Speed: {result['speed']:.2f} KM/H"
                )

                column.caption(
                    f"Ratio: {result['speed_ratio']}"
                )

            high_alerts = [
                horizon
                for horizon, result
                in congestion_predictions.items()
                if result["congestion"] == "High"
            ]

            medium_alerts = [
                horizon
                for horizon, result
                in congestion_predictions.items()
                if result["congestion"] == "Medium"
            ]

            if high_alerts:

                st.error(
                    "🚨 HIGH CONGESTION ALERT: "
                    + ", ".join(high_alerts)
                )

            elif medium_alerts:

                st.warning(
                    "⚠️ MEDIUM CONGESTION: "
                    + ", ".join(medium_alerts)
                )

            else:

                st.success(
                    "✅ No High Congestion Alert in "
                    "the selected forecast periods."
                )

            prediction_chart = pd.DataFrame({
                "Forecast": [
                    "5 Minutes",
                    "15 Minutes",
                    "30 Minutes"
                ],
                "Predicted Speed (KM/H)": [
                    predictions["5 Minutes"],
                    predictions["15 Minutes"],
                    predictions["30 Minutes"]
                ]
            })

            fig_prediction = px.line(
                prediction_chart,
                x="Forecast",
                y="Predicted Speed (KM/H)",
                markers=True,
                title="AI Traffic Speed Forecast"
            )

            st.plotly_chart(
                fig_prediction,
                use_container_width=True
            )

            st.info(
                "Experimental predictions only. Pali demo readings "
                "are converted from KM/H to MPH for model input. "
                "Predictions are converted back to KM/H. Congestion "
                "thresholds are demo rules, not validated traffic "
                "engineering standards. The model has not been "
                "validated on real Pali traffic observations."
            )

        except Exception as e:

            st.error("AI prediction could not be generated.")
            st.code(str(e))


# ==========================================
# TAB 3 - ANALYTICS
# ==========================================

with tab3:

    st.subheader("📈 Traffic Speed Analytics")

    speed_data = (
        filtered_df
        .set_index("timestamp")
        .resample("1h")["current_speed_kmph"]
        .mean()
        .reset_index()
    )

    fig_speed = px.line(
        speed_data,
        x="timestamp",
        y="current_speed_kmph",
        title="Average Traffic Speed Over Time",
        markers=True,
        labels={
            "current_speed_kmph": "Speed (KM/H)",
            "timestamp": "Time"
        }
    )

    st.plotly_chart(
        fig_speed,
        use_container_width=True
    )

    st.subheader("Road Comparison")

    road_comparison = (
        filtered_df.groupby("road_name")
        .agg(
            Average_Speed=("current_speed_kmph", "mean"),
            Free_Flow_Speed=("free_flow_speed_kmph", "mean"),
            Total_Readings=("road_segment_id", "count")
        )
        .reset_index()
    )

    st.dataframe(
        road_comparison,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# TAB 4 - TRAFFIC RECORDS
# ==========================================

with tab4:

    st.subheader("📋 Traffic Database Records")

    st.caption(
        f"Showing {len(filtered_df):,} records "
        "based on selected filters."
    )

    st.dataframe(
        filtered_df.sort_values(
            "timestamp",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Traffic Report CSV",
        data=csv_data,
        file_name="pali_traffic_report.csv",
        mime="text/csv"
    )


# ==========================================
# 8. FOOTER
# ==========================================

st.divider()

st.markdown("""
<div style="text-align:center; color:#64748b; padding:15px;">
    <b>🚦 Smart Traffic Prediction System</b><br>
    B.Tech Final Year Project | Pali, Rajasthan<br>
    Python • Machine Learning • Streamlit • SQL Server
</div>
""", unsafe_allow_html=True)


# streamlit run app.py