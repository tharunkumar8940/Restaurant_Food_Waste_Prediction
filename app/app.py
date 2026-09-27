# ============================================================
# RESTAURANT FOOD WASTE & DEMAND PREDICTION SYSTEM
# Professional Streamlit Application
# ============================================================

from pathlib import Path
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Restaurant AI | Food Waste & Demand Prediction",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


# ============================================================
# FILE PATHS
# ============================================================

RAW_DATA_FILE = RAW_DIR / "restaurant_orders.csv"
ML_DATA_FILE = PROCESSED_DIR / "restaurant_ml_dataset.csv"

BEST_MODEL_FILE = MODELS_DIR / "best_demand_model.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "demand_preprocessor.pkl"
FEATURES_FILE = MODELS_DIR / "demand_features.pkl"

DEMAND_FILE = OUTPUTS_DIR / "food_demand_analysis.csv"
WASTE_FILE = OUTPUTS_DIR / "food_waste_analysis.csv"
CHICKEN_FILE = OUTPUTS_DIR / "chicken_food_analysis.csv"
RECOMMENDATION_FILE = OUTPUTS_DIR / "food_recommendations.csv"
PREDICTION_FILE = OUTPUTS_DIR / "demand_predictions.csv"
MODEL_COMPARISON_FILE = OUTPUTS_DIR / "model_comparison.csv"
SUMMARY_FILE = OUTPUTS_DIR / "project_summary.csv"
MONTHLY_FILE = OUTPUTS_DIR / "monthly_demand.csv"
DAILY_FILE = OUTPUTS_DIR / "daily_demand.csv"


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Header */
    .main-header {
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        background: linear-gradient(
            135deg,
            #172554 0%,
            #1e3a8a 50%,
            #2563eb 100%
        );
        color: white;
        margin-bottom: 1.5rem;
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
    }

    .main-header p {
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-size: 1rem;
        opacity: 0.9;
    }

    /* Section titles */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #172554;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    /* KPI cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .metric-value {
        color: #172554;
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    /* Info box */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 0.9rem 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Recommendation */
    .recommendation-box {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.7rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem 0 1rem 0;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# REQUIRED FILE CHECK
# ============================================================

required_files = {
    "Restaurant dataset": RAW_DATA_FILE,
    "Demand model": BEST_MODEL_FILE,
    "Preprocessor": PREPROCESSOR_FILE,
    "Demand features": FEATURES_FILE,
    "Demand analysis": DEMAND_FILE,
    "Waste analysis": WASTE_FILE,
    "Chicken analysis": CHICKEN_FILE,
    "Recommendations": RECOMMENDATION_FILE,
    "Model comparison": MODEL_COMPARISON_FILE,
    "Project summary": SUMMARY_FILE,
}

missing_files = [
    name
    for name, path in required_files.items()
    if not path.exists()
]

if missing_files:
    st.error("Some required project files are missing.")

    st.write("Missing files:")

    for file_name in missing_files:
        st.write(f"• {file_name}")

    st.info(
        "Please make sure the Jupyter Notebook has been executed "
        "and the model/output files have been created."
    )

    st.stop()


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data
def load_raw_data():
    data = pd.read_csv(RAW_DATA_FILE)

    if "Order_Date" in data.columns:
        data["Order_Date"] = pd.to_datetime(
            data["Order_Date"],
            errors="coerce"
        )

    return data


@st.cache_data
def load_demand_analysis():
    data = pd.read_csv(
        DEMAND_FILE,
        index_col=0
    )

    data.index.name = "Food_Item"

    return data.reset_index()


@st.cache_data
def load_waste_analysis():
    data = pd.read_csv(
        WASTE_FILE,
        index_col=0
    )

    data.index.name = "Food_Item"

    return data.reset_index()


@st.cache_data
def load_chicken_analysis():
    data = pd.read_csv(
        CHICKEN_FILE,
        index_col=0
    )

    data.index.name = "Food_Item"

    return data.reset_index()


@st.cache_data
def load_recommendations():
    return pd.read_csv(RECOMMENDATION_FILE)


@st.cache_data
def load_model_comparison():
    return pd.read_csv(MODEL_COMPARISON_FILE)


@st.cache_data
def load_summary():
    return pd.read_csv(SUMMARY_FILE)


@st.cache_data
def load_monthly_demand():
    return pd.read_csv(MONTHLY_FILE)


@st.cache_data
def load_daily_demand():
    return pd.read_csv(DAILY_FILE)


@st.cache_resource
def load_model():
    return joblib.load(BEST_MODEL_FILE)


@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_FILE)


@st.cache_resource
def load_features():
    return joblib.load(FEATURES_FILE)


# ============================================================
# LOAD PROJECT DATA
# ============================================================

try:
    df = load_raw_data()
    demand_analysis = load_demand_analysis()
    waste_analysis = load_waste_analysis()
    chicken_analysis = load_chicken_analysis()
    recommendations = load_recommendations()
    model_comparison = load_model_comparison()
    summary = load_summary()
    monthly_demand = load_monthly_demand()
    daily_demand = load_daily_demand()

    model = load_model()
    preprocessor = load_preprocessor()
    features = load_features()

except Exception as error:
    st.error("Unable to load the project files.")
    st.exception(error)
    st.stop()


# ============================================================
# PROJECT METRICS
# ============================================================

total_orders = int(df["Quantity"].sum())
total_prepared = int(df["Preparation_Quantity"].sum())
total_wasted = int(df["Wasted_Quantity"].sum())
total_revenue = float(df["Total_Amount"].sum())

if total_prepared > 0:
    overall_waste_percentage = (
        total_wasted / total_prepared
    ) * 100
else:
    overall_waste_percentage = 0

number_of_food_items = df["Food_Item"].nunique()

best_model_name = (
    model_comparison.iloc[0]["Model"]
    if not model_comparison.empty
    else "Trained Regression Model"
)

best_model_r2 = (
    float(model_comparison.iloc[0]["R2_Score"])
    if "R2_Score" in model_comparison.columns
    and not model_comparison.empty
    else None
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🍽️ Restaurant AI")

st.sidebar.markdown(
    "### Food Waste & Demand Prediction"
)

st.sidebar.divider()

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Demand Analysis",
        "♻️ Waste Analysis",
        "🍗 Chicken Analysis",
        "🔮 Demand Prediction",
        "💡 Recommendations",
        "📋 Reports",
    ],
)

st.sidebar.divider()

st.sidebar.markdown("### System Information")

st.sidebar.write(
    f"**Food Items:** {number_of_food_items}"
)

st.sidebar.write(
    f"**Orders:** {len(df):,}"
)

st.sidebar.write(
    f"**Model:** {best_model_name}"
)

st.sidebar.success("System Online")

st.sidebar.caption(
    "Dataset: Synthetic demonstration data"
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🍽️ Restaurant AI Intelligence System</h1>
        <p>
            Food waste analysis, demand prediction and
            data-driven preparation recommendations
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD
# ============================================================

if menu == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">Business Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-box">
        This dashboard analyzes restaurant order patterns,
        identifies food waste, predicts demand and provides
        preparation recommendations using machine learning.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Food Quantity Ordered",
            f"{total_orders:,}"
        )

    with col2:
        st.metric(
            "Food Wasted",
            f"{total_wasted:,}"
        )

    with col3:
        st.metric(
            "Total Revenue",
            f"₹{total_revenue:,.0f}"
        )

    with col4:
        st.metric(
            "Waste Percentage",
            f"{overall_waste_percentage:.2f}%"
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="section-title">Top Food Items by Demand</div>',
            unsafe_allow_html=True,
        )

        top_demand = (
            demand_analysis
            .sort_values(
                "Total_Orders",
                ascending=False
            )
            .head(10)
            .set_index("Food_Item")
        )

        st.bar_chart(
            top_demand["Total_Orders"]
        )

    with col2:

        st.markdown(
            '<div class="section-title">Food Waste by Item</div>',
            unsafe_allow_html=True,
        )

        top_waste = (
            waste_analysis
            .sort_values(
                "Total_Wasted",
                ascending=False
            )
            .head(10)
            .set_index("Food_Item")
        )

        st.bar_chart(
            top_waste["Total_Wasted"]
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="section-title">Machine Learning Model</div>',
            unsafe_allow_html=True,
        )

        st.write(
            f"**Selected Model:** {best_model_name}"
        )

        if best_model_r2 is not None:
            st.write(
                f"**R² Score:** {best_model_r2:.4f}"
            )

        st.write(
            "**Purpose:** Predict expected food demand."
        )

    with col2:

        st.markdown(
            '<div class="section-title">Recommendation Summary</div>',
            unsafe_allow_html=True,
        )

        recommendation_counts = (
            recommendations["Recommendation"]
            .value_counts()
        )

        st.dataframe(
            recommendation_counts.rename(
                "Food Items"
            ).to_frame(),
            use_container_width=True
        )


# ============================================================
# DEMAND ANALYSIS
# ============================================================

elif menu == "📊 Demand Analysis":

    st.markdown(
        '<div class="section-title">Food Demand Analysis</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Analyze which menu items receive the highest and lowest "
        "customer demand."
    )

    selected_category = st.selectbox(
        "Filter by Category",
        ["All"] + sorted(
            df["Category"].dropna().unique().tolist()
        ),
    )

    filtered_demand = demand_analysis.copy()

    if selected_category != "All":

        category_items = df[
            df["Category"] == selected_category
        ]["Food_Item"].unique()

        filtered_demand = filtered_demand[
            filtered_demand["Food_Item"].isin(
                category_items
            )
        ]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Highest Demand Item",
            filtered_demand.loc[
                filtered_demand["Total_Orders"].idxmax(),
                "Food_Item"
            ]
            if not filtered_demand.empty
            else "N/A"
        )

    with col2:

        st.metric(
            "Lowest Demand Item",
            filtered_demand.loc[
                filtered_demand["Total_Orders"].idxmin(),
                "Food_Item"
            ]
            if not filtered_demand.empty
            else "N/A"
        )

    if not filtered_demand.empty:

        chart_data = (
            filtered_demand
            .sort_values(
                "Total_Orders",
                ascending=False
            )
            .set_index("Food_Item")
        )

        st.bar_chart(
            chart_data["Total_Orders"]
        )

        st.dataframe(
            filtered_demand.sort_values(
                "Total_Orders",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# WASTE ANALYSIS
# ============================================================

elif menu == "♻️ Waste Analysis":

    st.markdown(
        '<div class="section-title">Food Waste Analysis</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Identify menu items with high preparation waste "
        "and analyze their waste percentage."
    )

    waste_threshold = st.slider(
        "High Waste Threshold (%)",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
    )

    filtered_waste = waste_analysis.copy()

    filtered_waste["Waste_Status"] = np.where(
        filtered_waste["Waste_Percentage"]
        > waste_threshold,
        "High Waste",
        "Low Waste"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Wasted",
            f"{total_wasted:,}"
        )

    with col2:
        st.metric(
            "Overall Waste",
            f"{overall_waste_percentage:.2f}%"
        )

    with col3:
        high_waste_count = (
            filtered_waste["Waste_Status"]
            == "High Waste"
        ).sum()

        st.metric(
            "High-Waste Items",
            int(high_waste_count)
        )

    st.markdown("---")

    chart_data = (
        filtered_waste
        .sort_values(
            "Total_Wasted",
            ascending=False
        )
        .set_index("Food_Item")
    )

    st.bar_chart(
        chart_data["Total_Wasted"]
    )

    st.dataframe(
        filtered_waste.sort_values(
            "Waste_Percentage",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "High waste does not automatically mean the menu item "
        "should be removed. Preparation quantity should first "
        "be optimized using demand and waste patterns."
    )


# ============================================================
# CHICKEN ANALYSIS
# ============================================================

elif menu == "🍗 Chicken Analysis":

    st.markdown(
        '<div class="section-title">Chicken Food Analysis</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Focused analysis of chicken-based menu items."
    )

    chicken_items = df[
        df["Category"] == "Chicken"
    ]

    if chicken_items.empty:

        st.warning(
            "No chicken items were found in the dataset."
        )

    else:

        total_chicken_orders = int(
            chicken_items["Quantity"].sum()
        )

        total_chicken_waste = int(
            chicken_items["Wasted_Quantity"].sum()
        )

        chicken_revenue = float(
            chicken_items["Total_Amount"].sum()
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Chicken Quantity Ordered",
                f"{total_chicken_orders:,}"
            )

        with col2:
            st.metric(
                "Chicken Waste",
                f"{total_chicken_waste:,}"
            )

        with col3:
            st.metric(
                "Chicken Revenue",
                f"₹{chicken_revenue:,.0f}"
            )

        st.markdown("---")

        chicken_display = chicken_analysis.copy()

        chicken_display = chicken_display.sort_values(
            "Total_Orders",
            ascending=False
        )

        st.bar_chart(
            chicken_display.set_index(
                "Food_Item"
            )["Total_Orders"]
        )

        st.dataframe(
            chicken_display,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# DEMAND PREDICTION
# ============================================================

elif menu == "🔮 Demand Prediction":

    st.markdown(
        '<div class="section-title">AI Demand Prediction</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Enter future order conditions to estimate the expected "
        "quantity for a selected food item."
    )

    st.markdown(
        """
        <div class="info-box">
        The prediction model estimates food demand using menu,
        category, price, date, day, meal time and customer type.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        food_item = st.selectbox(
            "Food Item",
            sorted(
                df["Food_Item"]
                .dropna()
                .unique()
                .tolist()
            ),
        )

        prediction_date = st.date_input(
            "Prediction Date",
            value=date.today(),
        )

    with col2:

        meal_time = st.selectbox(
            "Meal Time",
            ["Breakfast", "Lunch", "Dinner"],
        )

        customer_type = st.selectbox(
            "Customer Type",
            ["New", "Regular"],
        )

    if st.button(
        "🔮 Predict Food Demand",
        type="primary",
        use_container_width=True,
    ):

        selected_row = df[
            df["Food_Item"] == food_item
        ]

        if selected_row.empty:

            st.error(
                "Selected food item was not found."
            )

        else:

            category = selected_row.iloc[0]["Category"]
            unit_price = float(
                selected_row.iloc[0]["Unit_Price"]
            )

            prediction_datetime = pd.Timestamp(
                prediction_date
            )

            day_name = (
                prediction_datetime.day_name()
            )

            day_number = (
                prediction_datetime.day
            )

            month_number = (
                prediction_datetime.month
            )

            day_of_week_number = (
                prediction_datetime.dayofweek
            )

            is_weekend = int(
                day_of_week_number >= 5
            )

            input_data = pd.DataFrame(
                {
                    "Food_Item": [food_item],
                    "Category": [category],
                    "Unit_Price": [unit_price],
                    "Day": [day_name],
                    "Meal_Time": [meal_time],
                    "Customer_Type": [customer_type],
                    "Year": [prediction_datetime.year],
                    "Month_Number": [month_number],
                    "Day_Number": [day_number],
                    "Day_of_Week_Number": [
                        day_of_week_number
                    ],
                    "Is_Weekend": [is_weekend],
                }
            )

            try:

                input_data = input_data[
                    features
                ]

                processed_input = (
                    preprocessor.transform(
                        input_data
                    )
                )

                prediction = model.predict(
                    processed_input
                )[0]

                predicted_quantity = max(
                    0,
                    int(round(prediction))
                )

                st.markdown("---")

                st.success(
                    "Demand prediction completed successfully."
                )

                result_col1, result_col2, result_col3 = (
                    st.columns(3)
                )

                with result_col1:

                    st.metric(
                        "Food Item",
                        food_item
                    )

                with result_col2:

                    st.metric(
                        "Predicted Demand",
                        f"{predicted_quantity} units"
                    )

                with result_col3:

                    st.metric(
                        "Unit Price",
                        f"₹{unit_price:,.0f}"
                    )

                st.info(
                    f"The model estimates approximately "
                    f"{predicted_quantity} units of "
                    f"{food_item} may be ordered for "
                    f"{prediction_date.strftime('%d %B %Y')} "
                    f"during {meal_time}."
                )

            except Exception as error:

                st.error(
                    "Prediction failed."
                )

                st.exception(error)


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif menu == "💡 Recommendations":

    st.markdown(
        '<div class="section-title">Food Preparation Recommendations</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Recommendations combine historical demand and waste "
        "patterns to support restaurant preparation decisions."
    )

    recommendation_filter = st.selectbox(
        "Recommendation Type",
        [
            "All",
            "Increase Preparation",
            "Optimize Preparation",
            "Reduce Preparation",
            "Monitor",
        ],
    )

    filtered_recommendations = (
        recommendations.copy()
    )

    if recommendation_filter != "All":

        filtered_recommendations = (
            filtered_recommendations[
                filtered_recommendations[
                    "Recommendation"
                ] == recommendation_filter
            ]
        )

    st.dataframe(
        filtered_recommendations,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    recommendation_counts = (
        filtered_recommendations[
            "Recommendation"
        ]
        .value_counts()
    )

    if not recommendation_counts.empty:

        st.bar_chart(
            recommendation_counts
        )

    st.markdown(
        """
        ### Recommendation Logic

        **High Demand + Low Waste**  
        → Increase Preparation

        **High Demand + High Waste**  
        → Optimize Preparation

        **Low Demand + High Waste**  
        → Reduce Preparation

        **Low Demand + Low Waste**  
        → Monitor

        These recommendations are intended to support
        operational decisions; they do not automatically
        remove food items from the menu.
        """
    )


# ============================================================
# REPORTS
# ============================================================

elif menu == "📋 Reports":

    st.markdown(
        '<div class="section-title">Project Reports</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Detailed project outputs generated during the "
        "data science and machine learning workflow."
    )

    # --------------------------------------------------------
    # PROJECT SUMMARY
    # --------------------------------------------------------

    st.markdown("### 📌 Project Summary")

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.markdown("### 🤖 Model Comparison")

    st.dataframe(
        model_comparison,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # MONTHLY DEMAND
    # --------------------------------------------------------

    st.markdown("### 📅 Monthly Demand")

    if not monthly_demand.empty:

        monthly_display = monthly_demand.copy()

        monthly_display.columns = [
            "Month",
            "Total Quantity"
        ]

        monthly_chart = (
            monthly_display
            .set_index("Month")
        )

        st.line_chart(
            monthly_chart["Total Quantity"]
        )

        st.dataframe(
            monthly_display,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # DAILY DEMAND
    # --------------------------------------------------------

    st.markdown("### 📆 Weekly Demand")

    if not daily_demand.empty:

        daily_display = daily_demand.copy()

        daily_display.columns = [
            "Day",
            "Total Quantity"
        ]

        st.bar_chart(
            daily_display.set_index(
                "Day"
            )["Total Quantity"]
        )

    # --------------------------------------------------------
    # RAW DATA PREVIEW
    # --------------------------------------------------------

    st.markdown("### 🗃️ Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # DOWNLOAD REPORTS
    # --------------------------------------------------------

    st.markdown("### ⬇️ Download Reports")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.download_button(
            "Download Demand Analysis",
            demand_analysis.to_csv(
                index=False
            ),
            "food_demand_analysis.csv",
            "text/csv",
            use_container_width=True,
        )

    with col2:

        st.download_button(
            "Download Waste Analysis",
            waste_analysis.to_csv(
                index=False
            ),
            "food_waste_analysis.csv",
            "text/csv",
            use_container_width=True,
        )

    with col3:

        st.download_button(
            "Download Recommendations",
            recommendations.to_csv(
                index=False
            ),
            "food_recommendations.csv",
            "text/csv",
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Restaurant AI Intelligence System<br>
        Food Waste Analysis • Demand Prediction •
        Machine Learning • Business Recommendations
        <br><br>
        Built with Python, Pandas, Scikit-learn,
        Joblib and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)