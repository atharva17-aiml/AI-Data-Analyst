import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os
import html
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    confusion_matrix,
    classification_report
)


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "df" not in st.session_state:
    st.session_state["df"] = None

if "trained_model" not in st.session_state:
    st.session_state["trained_model"] = None

if "model_info" not in st.session_state:
    st.session_state["model_info"] = None

if "model_metrics" not in st.session_state:
    st.session_state["model_metrics"] = None

if "feature_importance" not in st.session_state:
    st.session_state["feature_importance"] = None

if "recommended_target" not in st.session_state:
    st.session_state["recommended_target"] = None

if "recommended_problem" not in st.session_state:
    st.session_state["recommended_problem"] = None

if "recommended_model" not in st.session_state:
    st.session_state["recommended_model"] = None

if "ai_suggestion" not in st.session_state:
    st.session_state["ai_suggestion"] = None

if "report" not in st.session_state:
    st.session_state["report"] = None


# =========================================================
# GROQ CLIENT
# =========================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

if GROQ_API_KEY:

    client = Groq(
        api_key=GROQ_API_KEY
    )

else:

    client = None


# =========================================================
# UI STYLE
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0E1117;
        color: white;
    }

    .stButton > button {
        background-color: #22c55e;
        color: white;
        border-radius: 8px;
        min-height: 42px;
        width: 100%;
        font-weight: 600;
        border: none;
    }

    .stButton > button:hover {
        background-color: #16a34a;
        color: white;
    }

    div[data-testid="stMetric"] {
        background-color: #161B22;
        padding: 15px;
        border-radius: 10px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .recommendation-box {
        background-color: #161B22;
        border-left: 5px solid #22c55e;
        padding: 18px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# AI FUNCTION
# =========================================================

def ask_ai(prompt):

    if client is None:

        return (
            "❌ GROQ_API_KEY is missing. "
            "Please add GROQ_API_KEY to your .env file."
        )

    try:

        response = client.chat.completions.create(

            model="openai/gpt-oss-120b",

            messages=[
                {
                    "role": "system",
                    "content": """
You are an expert AI Data Analyst.

Analyze only the information provided.

Rules:
- Never invent dataset values.
- Clearly distinguish facts from recommendations.
- Prefer practical machine-learning recommendations.
- Explain technical concepts in simple academic language.
- Use headings and bullet points.
- When recommending an ML target, avoid ID columns.
- Do not recommend columns with extremely high cardinality as targets.
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        return (
            f"❌ AI Error: {str(e)}"
        )


# =========================================================
# DATASET CONTEXT
# =========================================================

def get_dataset_context(df):

    context = []

    context.append(
        f"Dataset contains {df.shape[0]} rows "
        f"and {df.shape[1]} columns."
    )

    context.append(
        "\nColumns:\n"
        + ", ".join(
            [str(c) for c in df.columns]
        )
    )

    context.append(
        "\nData Types:\n"
        + df.dtypes.to_string()
    )

    context.append(
        "\nMissing Values:\n"
        + df.isnull().sum().to_string()
    )

    context.append(
        "\nUnique Values:\n"
        + df.nunique(
            dropna=False
        ).to_string()
    )

    context.append(
        "\nDuplicate Rows: "
        + str(
            df.duplicated().sum()
        )
    )

    try:

        numeric_summary = (
            df
            .select_dtypes(
                include="number"
            )
            .describe()
            .round(2)
            .to_string()
        )

        context.append(
            "\nNumeric Statistical Summary:\n"
            + numeric_summary
        )

    except Exception:

        pass

    return "\n".join(context)


# =========================================================
# IDENTIFY ID-LIKE COLUMNS
# =========================================================

def identify_id_columns(df):

    id_columns = []

    known_ids = {
        "empid",
        "employeeid",
        "employee_id",
        "employee id",
        "id",
        "email",
        "ademail",
        "employeeemail",
        "employee email"
    }

    for column in df.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
        )

        unique_count = (
            df[column]
            .nunique(
                dropna=False
            )
        )

        unique_ratio = (
            unique_count
            /
            max(len(df), 1)
        )

        if normalized in known_ids:

            id_columns.append(
                column
            )

        elif (
            unique_ratio >= 0.95
            and
            unique_count > 100
        ):

            id_columns.append(
                column
            )

    return list(
        dict.fromkeys(
            id_columns
        )
    )


# =========================================================
# AUTOMATIC ML TARGET RECOMMENDATION
# =========================================================

def recommend_target(df):

    id_columns = identify_id_columns(
        df
    )

    candidates = []

    for column in df.columns:

        if column in id_columns:

            continue

        missing_count = int(
            df[column].isnull().sum()
        )

        missing_ratio = (
            missing_count
            /
            max(len(df), 1)
        )

        unique_count = int(
            df[column].nunique(
                dropna=True
            )
        )

        if unique_count < 2:

            continue

        score = 0

        name = (
            str(column)
            .strip()
            .lower()
        )

        # -----------------------------------------
        # Strong HR/business targets
        # -----------------------------------------

        if name == "terminationtype":

            score += 100

        elif name in {
            "performance score",
            "performancescore"
        }:

            score += 90

        elif name in {
            "current employee rating",
            "currentemployeerating"
        }:

            score += 85

        elif "exit" in name:

            score += 80

        elif "rating" in name:

            score += 70

        elif "performance" in name:

            score += 70


        # -----------------------------------------
        # Reasonable categorical target
        # -----------------------------------------

        if (
            df[column].dtype == "object"
            and
            unique_count <= 20
        ):

            score += 40


        # -----------------------------------------
        # Reasonable numeric target
        # -----------------------------------------

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            if (
                2 <= unique_count <= 20
            ):

                score += 25

            elif unique_count > 20:

                score += 15


        # -----------------------------------------
        # Penalize excessive missing values
        # -----------------------------------------

        if missing_ratio > 0.50:

            score -= 50

        elif missing_ratio > 0.30:

            score -= 20


        # -----------------------------------------
        # Penalize extremely high cardinality
        # -----------------------------------------

        if (
            unique_count > 100
            and
            unique_count
            >
            len(df) * 0.50
        ):

            score -= 50


        candidates.append(
            (
                column,
                score
            )
        )

    if not candidates:

        return None

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return candidates[0][0]


# =========================================================
# DETERMINE PROBLEM TYPE
# =========================================================

def determine_problem_type(
    df,
    target
):

    if target is None:

        return None

    series = (
        df[target]
        .dropna()
    )

    if len(series) == 0:

        return None

    unique_count = (
        series.nunique()
    )

    if (
        series.dtype == "object"
        or
        str(
            series.dtype
        ) == "category"
    ):

        return "Classification"

    if unique_count <= 10:

        return "Classification"

    return "Regression"


# =========================================================
# PREPROCESS DATA FOR ML
# =========================================================

def prepare_ml_data(
    df,
    target
):

    data = df.copy()

    # -----------------------------------------
    # Remove rows without target
    # -----------------------------------------

    data = data.dropna(
        subset=[target]
    )

    if len(data) < 20:

        raise ValueError(
            "At least 20 rows with a valid "
            "target value are required."
        )

    X = data.drop(
        columns=[target]
    ).copy()

    y = data[target].copy()


    # -----------------------------------------
    # Remove ID columns
    # -----------------------------------------

    id_columns = identify_id_columns(
        X
    )

    if id_columns:

        X = X.drop(
            columns=id_columns,
            errors="ignore"
        )


    # -----------------------------------------
    # Remove free-text termination description
    # -----------------------------------------

    free_text_columns = []

    for column in X.columns:

        name = (
            str(column)
            .lower()
            .replace("_", "")
            .replace(" ", "")
        )

        if name in {
            "terminationdescription",
            "description",
            "comments",
            "comment"
        }:

            free_text_columns.append(
                column
            )


    if free_text_columns:

        X = X.drop(
            columns=free_text_columns
        )


    # -----------------------------------------
    # Convert object date columns
    # -----------------------------------------

    date_columns = []

    for column in X.columns:

        column_name = (
            str(column)
            .lower()
        )

        if (
            "date" in column_name
            or
            column_name == "dob"
        ):

            if (
                X[column].dtype
                == "object"
            ):

                converted = pd.to_datetime(
                    X[column],
                    errors="coerce"
                )

                valid_ratio = (
                    converted.notna()
                    .mean()
                )

                if valid_ratio >= 0.50:

                    X[
                        column + "_year"
                    ] = converted.dt.year

                    X[
                        column + "_month"
                    ] = converted.dt.month

                    X[
                        column + "_day"
                    ] = converted.dt.day

                    date_columns.append(
                        column
                    )


    if date_columns:

        X = X.drop(
            columns=date_columns
        )


    # -----------------------------------------
    # Remove completely empty columns
    # -----------------------------------------

    X = X.dropna(
        axis=1,
        how="all"
    )


    # -----------------------------------------
    # Numeric columns
    # -----------------------------------------

    numeric_columns = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
    )

    for column in numeric_columns:

        median_value = (
            X[column]
            .median()
        )

        if pd.isna(
            median_value
        ):

            median_value = 0

        X[column] = (
            X[column]
            .fillna(
                median_value
            )
        )


    # -----------------------------------------
    # Categorical columns
    # -----------------------------------------

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "category"
            ]
        )
        .columns
    )


    low_cardinality = []
    high_cardinality = []

    for column in categorical_columns:

        unique_count = int(
            X[column].nunique(
                dropna=False
            )
        )

        if unique_count <= 15:

            low_cardinality.append(
                column
            )

        else:

            high_cardinality.append(
                column
            )


    # -----------------------------------------
    # Frequency encode high-cardinality
    # -----------------------------------------

    for column in high_cardinality:

        X[column] = (
            X[column]
            .fillna("Unknown")
            .astype(str)
        )

        frequency_map = (
            X[column]
            .value_counts(
                normalize=True
            )
        )

        X[column] = (
            X[column]
            .map(
                frequency_map
            )
            .fillna(0)
        )


    # -----------------------------------------
    # One-hot encode low-cardinality
    # -----------------------------------------

    if low_cardinality:

        X = pd.get_dummies(
            X,
            columns=low_cardinality,
            drop_first=True,
            dtype="float32"
        )


    # -----------------------------------------
    # Convert remaining values
    # -----------------------------------------

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )


    X = X.fillna(
        0
    )


    # -----------------------------------------
    # Memory optimization
    # -----------------------------------------

    X = X.astype(
        "float32"
    )


    if X.shape[1] == 0:

        raise ValueError(
            "No usable features remain after "
            "preprocessing."
        )


    return X, y


# =========================================================
# PDF REPORT
# =========================================================

def create_pdf(
    text
):

    path = (
        "AI_Report.pdf"
    )

    document = SimpleDocTemplate(
        path,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm
    )

    styles = (
        getSampleStyleSheet()
    )

    styles[
        "Normal"
    ].fontSize = 10

    styles[
        "Normal"
    ].leading = 14

    story = []

    story.append(
        Paragraph(
            "AI Data Analyst Report",
            styles["Title"]
        )
    )

    story.append(
        Spacer(
            1,
            0.5 * cm
        )
    )

    for line in text.split(
        "\n"
    ):

        if not line.strip():

            continue

        safe_line = html.escape(
            line
        )

        story.append(
            Paragraph(
                safe_line,
                styles["Normal"]
            )
        )

        story.append(
            Spacer(
                1,
                0.12 * cm
            )
        )

    document.build(
        story
    )

    return path


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div style="
        display:flex;
        align-items:center;
        margin-bottom:20px;
    ">
        <div>
            <h1 style="margin-bottom:0;">
                🤖 AI Data Analyst
            </h1>
            <p style="color:#9ca3af;">
                AI-powered data analysis and machine learning
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FILE UPLOAD
# =========================================================

st.markdown(
    "### 📁 Dataset Upload"
)

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=[
        "csv",
        "xlsx"
    ]
)


# =========================================================
# LOAD DATASET
# =========================================================

if uploaded_file is not None:

    try:

        if (
            uploaded_file
            .name
            .lower()
            .endswith(
                ".csv"
            )
        ):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

        st.session_state[
            "df"
        ] = df

        # Reset ML state for a new file
        st.session_state[
            "trained_model"
        ] = None

        st.session_state[
            "model_info"
        ] = None

        st.session_state[
            "model_metrics"
        ] = None

        st.session_state[
            "feature_importance"
        ] = None

        st.session_state[
            "ai_suggestion"
        ] = None

        st.session_state[
            "report"
        ] = None

    except Exception as e:

        st.error(
            f"❌ Unable to read dataset: {e}"
        )

        st.stop()


# =========================================================
# MAIN APPLICATION
# =========================================================

if st.session_state[
    "df"
] is None:

    st.info(
        "👆 Upload a CSV or Excel dataset to begin."
    )

    st.stop()


df = st.session_state[
    "df"
]


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.markdown(
    "### 📊 Dataset Overview"
)

col1, col2, col3, col4 = (
    st.columns(4)
)

with col1:

    st.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )

with col2:

    st.metric(
        "Columns",
        df.shape[1]
    )

with col3:

    st.metric(
        "Missing Values",
        f"{int(df.isnull().sum().sum()):,}"
    )

with col4:

    st.metric(
        "Duplicate Rows",
        f"{int(df.duplicated().sum()):,}"
    )


# =========================================================
# DATA PREVIEW
# =========================================================

st.markdown(
    "### 👀 Dataset Preview"
)

st.dataframe(
    df.head(20),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# DATA QUALITY
# =========================================================

st.markdown(
    "### 🔍 Data Quality"
)

quality_df = pd.DataFrame(
    {
        "Column": df.columns,
        "Data Type": [
            str(dtype)
            for dtype in df.dtypes
        ],
        "Missing Values": [
            int(value)
            for value in df.isnull().sum()
        ],
        "Missing %": [
            round(
                value
                /
                len(df)
                *
                100,
                2
            )
            for value in df.isnull().sum()
        ],
        "Unique Values": [
            int(
                df[column]
                .nunique()
            )
            for column in df.columns
        ]
    }
)

st.dataframe(
    quality_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# AI INSIGHTS
# =========================================================

st.markdown(
    "### ⚡ AI-Powered Insights"
)

if st.button(
    "Generate AI Insights",
    key="generate_insights"
):

    with st.spinner(
        "AI is analyzing the dataset..."
    ):

        context = (
            get_dataset_context(
                df
            )
        )

        prompt = f"""
Analyze the following dataset.

{context}

Provide:

1. Dataset Overview
2. Important Patterns
3. Important Numerical Findings
4. Missing Data Issues
5. Potential Anomalies
6. Relationships Between Variables
7. Important Business/Academic Insights
8. Recommendations

Do not invent values.
Use only information supported by the dataset.
"""

        result = ask_ai(
            prompt
        )

        st.markdown(
            result
        )


# =========================================================
# ML MODEL RECOMMENDATION
# =========================================================

st.markdown(
    "### 🤖 ML Model Recommendation"
)

recommended_target = (
    recommend_target(
        df
    )
)

recommended_problem = (
    determine_problem_type(
        df,
        recommended_target
    )
)


if recommended_problem == "Classification":

    recommended_model = (
        "Random Forest Classifier"
    )

elif recommended_problem == "Regression":

    recommended_model = (
        "Random Forest Regressor"
    )

else:

    recommended_model = (
        "Unable to determine"
    )


# Store recommendation
st.session_state[
    "recommended_target"
] = recommended_target

st.session_state[
    "recommended_problem"
] = recommended_problem

st.session_state[
    "recommended_model"
] = recommended_model


if recommended_target is not None:

    rec_col1, rec_col2, rec_col3 = (
        st.columns(3)
    )

    with rec_col1:

        st.metric(
            "Recommended Target",
            recommended_target
        )

    with rec_col2:

        st.metric(
            "Problem Type",
            recommended_problem
        )

    with rec_col3:

        st.metric(
            "Recommended Model",
            recommended_model
        )


    st.markdown(
        f"""
        <div class="recommendation-box">

        <b>🎯 Recommended ML approach</b>

        <br><br>

        Target:
        <b>{recommended_target}</b>

        <br>

        Problem:
        <b>{recommended_problem}</b>

        <br>

        Model:
        <b>{recommended_model}</b>

        <br><br>

        The system automatically excludes employee IDs,
        email-like identifiers, extremely high-cardinality
        columns and free-text fields during preprocessing.

        </div>
        """,
        unsafe_allow_html=True
    )


    if st.button(
        "🧠 Explain This Recommendation",
        key="explain_recommendation"
    ):

        with st.spinner(
            "Generating explanation..."
        ):

            prompt = f"""
You are reviewing an uploaded dataset for an academic
AI and Machine Learning project.

Dataset:

{get_dataset_context(df)}

The system selected:

Target: {recommended_target}
Problem Type: {recommended_problem}
Recommended Model: {recommended_model}

Explain:

1. Why this target is suitable.
2. Why this problem type is suitable.
3. Why the recommended model is appropriate.
4. What preprocessing is required.
5. What evaluation metrics should be used.
6. What limitations should be mentioned in an academic project.

Do not invent information.
"""

            explanation = ask_ai(
                prompt
            )

            st.session_state[
                "ai_suggestion"
            ] = explanation

            st.markdown(
                explanation
            )

else:

    st.warning(
        "No suitable ML target could be automatically identified."
    )


# =========================================================
# ML MODEL TRAINING
# =========================================================

st.markdown(
    "### 🚀 Machine Learning"

)

st.write(
    "Train a memory-efficient ML model using the recommended target."
)


# Target options excluding obvious IDs
id_columns = identify_id_columns(
    df
)

target_options = [
    column
    for column in df.columns
    if column not in id_columns
]


if not target_options:

    st.error(
        "No suitable target columns are available."
    )

else:

    if (
        recommended_target
        in target_options
    ):

        default_index = (
            target_options.index(
                recommended_target
            )
        )

    else:

        default_index = 0


    selected_target = st.selectbox(
        "Target Column",
        target_options,
        index=default_index,
        key="ml_target_selector"
    )


    train_col1, train_col2 = (
        st.columns(2)
    )

    with train_col1:

        st.write(
            f"**Selected Target:** `{selected_target}`"
        )

    with train_col2:

        selected_problem = (
            determine_problem_type(
                df,
                selected_target
            )
        )

        st.write(
            f"**Detected Problem:** `{selected_problem}`"
        )


    if st.button(
        "🚀 Train Recommended Model",
        key="train_recommended_model"
    ):

        try:

            with st.spinner(
                "Preparing data and training model..."
            ):

                X, y = (
                    prepare_ml_data(
                        df,
                        selected_target
                    )
                )


                # =============================================
                # PROBLEM TYPE
                # =============================================

                problem_type = (
                    determine_problem_type(
                        df,
                        selected_target
                    )
                )


                # =============================================
                # CLASSIFICATION
                # =============================================

                if (
                    problem_type
                    ==
                    "Classification"
                ):

                    y = (
                        y
                        .astype(str)
                    )


                    class_count = (
                        y.nunique()
                    )

                    if class_count < 2:

                        raise ValueError(
                            "Target must contain at least "
                            "two classes."
                        )


                    class_sizes = (
                        y.value_counts()
                    )


                    test_rows = max(
                        int(
                            len(y)
                            * 0.20
                        ),
                        1
                    )


                    can_stratify = (
                        class_sizes.min() >= 2
                        and
                        test_rows >= class_count
                    )


                    if can_stratify:

                        X_train, X_test, y_train, y_test = (
                            train_test_split(
                                X,
                                y,
                                test_size=0.20,
                                random_state=42,
                                stratify=y
                            )
                        )

                    else:

                        X_train, X_test, y_train, y_test = (
                            train_test_split(
                                X,
                                y,
                                test_size=0.20,
                                random_state=42
                            )
                        )


                    model = (
                        RandomForestClassifier(
                            n_estimators=150,
                            max_depth=15,
                            min_samples_leaf=2,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=-1
                        )
                    )


                    model.fit(
                        X_train,
                        y_train
                    )


                    predictions = (
                        model.predict(
                            X_test
                        )
                    )


                    accuracy = (
                        accuracy_score(
                            y_test,
                            predictions
                        )
                    )


                    weighted_f1 = (
                        f1_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0
                        )
                    )


                    macro_f1 = (
                        f1_score(
                            y_test,
                            predictions,
                            average="macro",
                            zero_division=0
                        )
                    )


                    st.session_state[
                        "trained_model"
                    ] = model

                    st.session_state[
                        "model_info"
                    ] = {
                        "target": selected_target,
                        "problem": "Classification",
                        "model": "Random Forest Classifier"
                    }

                    st.session_state[
                        "model_metrics"
                    ] = {
                        "Accuracy": accuracy,
                        "Weighted F1": weighted_f1,
                        "Macro F1": macro_f1
                    }


                    # =========================================
                    # FEATURE IMPORTANCE
                    # =========================================

                    importance_df = pd.DataFrame(
                        {
                            "Feature": X.columns,
                            "Importance":
                                model.feature_importances_
                        }
                    )

                    importance_df = (
                        importance_df
                        .sort_values(
                            "Importance",
                            ascending=False
                        )
                        .head(15)
                    )

                    st.session_state[
                        "feature_importance"
                    ] = importance_df


                    # =========================================
                    # RESULTS
                    # =========================================

                    st.success(
                        "✅ Classification Model Trained Successfully"
                    )


                    metric1, metric2, metric3 = (
                        st.columns(3)
                    )

                    with metric1:

                        st.metric(
                            "Accuracy",
                            f"{accuracy:.2%}"
                        )

                    with metric2:

                        st.metric(
                            "Weighted F1",
                            f"{weighted_f1:.2%}"
                        )

                    with metric3:

                        st.metric(
                            "Macro F1",
                            f"{macro_f1:.2%}"
                        )


                    # =========================================
                    # FEATURE IMPORTANCE
                    # =========================================

                    st.markdown(
                        "#### 🔎 Top Predictive Features"
                    )

                    st.dataframe(
                        importance_df,
                        use_container_width=True,
                        hide_index=True
                    )


                    # =========================================
                    # CONFUSION MATRIX
                    # =========================================

                    st.markdown(
                        "#### 📊 Confusion Matrix"
                    )

                    labels = sorted(
                        y.unique()
                    )

                    cm = confusion_matrix(
                        y_test,
                        predictions,
                        labels=labels
                    )


                    fig, ax = plt.subplots(
                        figsize=(8, 6)
                    )

                    image = ax.imshow(
                        cm
                    )

                    fig.colorbar(
                        image,
                        ax=ax
                    )

                    ax.set_xticks(
                        range(len(labels))
                    )

                    ax.set_yticks(
                        range(len(labels))
                    )

                    ax.set_xticklabels(
                        labels,
                        rotation=45,
                        ha="right"
                    )

                    ax.set_yticklabels(
                        labels
                    )

                    ax.set_xlabel(
                        "Predicted"
                    )

                    ax.set_ylabel(
                        "Actual"
                    )

                    ax.set_title(
                        "Confusion Matrix"
                    )

                    for i in range(
                        len(labels)
                    ):

                        for j in range(
                            len(labels)
                        ):

                            ax.text(
                                j,
                                i,
                                cm[i, j],
                                ha="center",
                                va="center"
                            )

                    plt.tight_layout()

                    st.pyplot(
                        fig
                    )

                    plt.close(
                        fig
                    )


                # =============================================
                # REGRESSION
                # =============================================

                else:

                    y = pd.to_numeric(
                        y,
                        errors="coerce"
                    )


                    valid_rows = (
                        y.notna()
                    )


                    X = X.loc[
                        valid_rows
                    ]

                    y = y.loc[
                        valid_rows
                    ]


                    if len(y) < 20:

                        raise ValueError(
                            "Not enough valid numeric "
                            "target values."
                        )


                    X_train, X_test, y_train, y_test = (
                        train_test_split(
                            X,
                            y,
                            test_size=0.20,
                            random_state=42
                        )
                    )


                    model = (
                        RandomForestRegressor(
                            n_estimators=150,
                            max_depth=15,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=-1
                        )
                    )


                    model.fit(
                        X_train,
                        y_train
                    )


                    predictions = (
                        model.predict(
                            X_test
                        )
                    )


                    r2 = (
                        r2_score(
                            y_test,
                            predictions
                        )
                    )


                    mae = (
                        mean_absolute_error(
                            y_test,
                            predictions
                        )
                    )


                    st.session_state[
                        "trained_model"
                    ] = model

                    st.session_state[
                        "model_info"
                    ] = {
                        "target": selected_target,
                        "problem": "Regression",
                        "model": "Random Forest Regressor"
                    }

                    st.session_state[
                        "model_metrics"
                    ] = {
                        "R² Score": r2,
                        "MAE": mae
                    }


                    importance_df = pd.DataFrame(
                        {
                            "Feature": X.columns,
                            "Importance":
                                model.feature_importances_
                        }
                    )

                    importance_df = (
                        importance_df
                        .sort_values(
                            "Importance",
                            ascending=False
                        )
                        .head(15)
                    )


                    st.session_state[
                        "feature_importance"
                    ] = importance_df


                    st.success(
                        "✅ Regression Model Trained Successfully"
                    )


                    metric1, metric2 = (
                        st.columns(2)
                    )

                    with metric1:

                        st.metric(
                            "R² Score",
                            f"{r2:.3f}"
                        )

                    with metric2:

                        st.metric(
                            "MAE",
                            f"{mae:.3f}"
                        )


                    st.markdown(
                        "#### 🔎 Top Predictive Features"
                    )

                    st.dataframe(
                        importance_df,
                        use_container_width=True,
                        hide_index=True
                    )


        except MemoryError:

            st.error(
                """
                ❌ Memory Error

                The dataset is too large for the current
                preprocessing configuration.
                """
            )

        except Exception as e:

            st.error(
                f"❌ Model Training Error: {e}"
            )


# =========================================================
# CORRELATION ANALYSIS
# =========================================================

st.markdown("### 📊 Correlation Analysis")

# Create a clean copy for correlation analysis
corr_df = df.copy()

# Remove ID-like columns
id_columns = identify_id_columns(corr_df)

if id_columns:
    corr_df = corr_df.drop(
        columns=id_columns,
        errors="ignore"
    )

# Remove unwanted index columns
unwanted_index_columns = []

for column in corr_df.columns:

    column_name = (
        str(column)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
    )

    if column_name in {
        "index",
        "unnamed0",
        "unnamed:0",
        "rowid",
        "rowindex"
    }:
        unwanted_index_columns.append(column)

if unwanted_index_columns:
    corr_df = corr_df.drop(
        columns=unwanted_index_columns,
        errors="ignore"
    )

# Select numeric columns
numeric_columns = (
    corr_df
    .select_dtypes(include="number")
    .columns
)

if len(numeric_columns) >= 2:

    # Calculate correlation
    correlation = (
        corr_df[numeric_columns]
        .corr()
        .round(2)
    )

    # Correlation table
    st.markdown("#### 🔢 Numeric Feature Correlations")

    st.dataframe(
        correlation,
        use_container_width=True,
        height=180
    )

    # Single correlation heatmap
    if len(numeric_columns) <= 15:

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        image = ax.imshow(
            correlation,
            cmap="viridis",
            aspect="auto"
        )

        # Color bar
        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

        # X-axis
        ax.set_xticks(
            range(len(correlation.columns))
        )

        ax.set_xticklabels(
            correlation.columns,
            rotation=45,
            ha="right",
            fontsize=9
        )

        # Y-axis
        ax.set_yticks(
            range(len(correlation.columns))
        )

        ax.set_yticklabels(
            correlation.columns,
            fontsize=9
        )

        # Title
        ax.set_title(
            "Correlation Matrix",
            fontsize=14,
            pad=12
        )

        # Display correlation values
        for i in range(
            len(correlation.columns)
        ):

            for j in range(
                len(correlation.columns)
            ):

                ax.text(
                    j,
                    i,
                    f"{correlation.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9
                )

        plt.tight_layout()

        # Render ONLY ONE chart
        st.pyplot(
            fig,
            use_container_width=False
        )

        plt.close(fig)

    else:

        st.info(
            "The heatmap is hidden because the dataset "
            "contains more than 15 numeric columns. "
            "The correlation table is still available above."
        )

else:

    st.info(
        "At least two meaningful numeric columns "
        "are required for correlation analysis."
    )


# =========================================================
# VISUALIZATION
# =========================================================

st.markdown(
    "### 📈 Data Visualization"
)

visualization_columns = (
    list(df.columns)
)

selected_column = st.selectbox(
    "Select Column",
    visualization_columns,
    key="visualization_column"
)

chart_type = st.selectbox(
    "Chart Type",
    [
        "Distribution",
        "Bar Chart",
        "Box Plot"
    ],
    key="chart_type"
)


if st.button(
    "Generate Visualization",
    key="generate_chart"
):

    column_data = (
        df[selected_column]
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    if chart_type == "Distribution":

        if pd.api.types.is_numeric_dtype(
            column_data
        ):

            ax.hist(
                column_data.dropna(),
                bins=20
            )

            ax.set_title(
                f"Distribution of {selected_column}"
            )

            ax.set_xlabel(
                selected_column
            )

            ax.set_ylabel(
                "Frequency"
            )

        else:

            counts = (
                column_data
                .value_counts()
                .head(10)
            )

            counts.plot(
                kind="bar",
                ax=ax
            )

            ax.set_title(
                f"Top Values - {selected_column}"
            )

            ax.set_ylabel(
                "Count"
            )


    elif chart_type == "Bar Chart":

        counts = (
            column_data
            .value_counts()
            .head(10)
        )

        counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            f"Top 10 Values - {selected_column}"
        )

        ax.set_ylabel(
            "Count"
        )


    elif chart_type == "Box Plot":

        if pd.api.types.is_numeric_dtype(
            column_data
        ):

            ax.boxplot(
                column_data.dropna()
            )

            ax.set_title(
                f"Box Plot - {selected_column}"
            )

        else:

            plt.close(
                fig
            )

            st.warning(
                "Box Plot requires a numeric column."
            )

            st.stop()


    plt.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(
        fig
    )


# =========================================================
# AI DATA ASSISTANT
# =========================================================

st.markdown(
    "### 🧠 AI Data Assistant"
)

question = st.text_input(
    "Ask a question about your dataset",
    placeholder=(
        "Example: Which department has the highest "
        "average employee rating?"
    ),
    key="ai_question"
)


if st.button(
    "Ask AI",
    key="ask_ai_button"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "AI is analyzing your question..."
        ):

            context = (
                get_dataset_context(
                    df
                )
            )

            prompt = f"""
You are an AI Data Analysis Assistant.

Dataset:

{context}

User Question:

{question}

Answer using only the dataset information.

Do not invent values.

If the dataset does not contain enough information,
clearly explain what information is missing.
"""

            answer = ask_ai(
                prompt
            )

            st.markdown(
                answer
            )


# =========================================================
# REPORT GENERATION
# =========================================================

st.markdown(
    "### 📑 AI Analysis Report"
)

if st.button(
    "Generate Professional PDF Report",
    key="generate_report"
):

    with st.spinner(
        "Generating professional AI report..."
    ):

        context = (
            get_dataset_context(
                df
            )
        )


        model_summary = ""

        if (
            st.session_state[
                "model_info"
            ]
            is not None
        ):

            model_summary = f"""
ML MODEL INFORMATION:

Target:
{st.session_state["model_info"]["target"]}

Problem:
{st.session_state["model_info"]["problem"]}

Model:
{st.session_state["model_info"]["model"]}

Metrics:
{st.session_state["model_metrics"]}
"""


        prompt = f"""
Create a professional academic data analysis report.

DATASET:

{context}

{model_summary}

Report structure:

1. Executive Summary
2. Dataset Overview
3. Data Quality Analysis
4. Important Statistical Findings
5. Patterns and Relationships
6. AI-Generated Insights
7. Machine Learning Approach
8. Model Evaluation
9. Important Predictive Features
10. Limitations
11. Recommendations
12. Conclusion

This is an MCA AI/ML academic project.

Use professional but understandable language.

Do not invent values.
"""

        report = ask_ai(
            prompt
        )

        st.session_state[
            "report"
        ] = report


        pdf_path = create_pdf(
            report
        )


        st.markdown(
            report
        )


        with open(
            pdf_path,
            "rb"
        ) as pdf_file:

            pdf_bytes = (
                pdf_file.read()
            )


        st.download_button(
            "⬇️ Download PDF Report",
            pdf_bytes,
            "AI_Data_Analyst_Report.pdf",
            "application/pdf",
            key="download_report"
        )


# =========================================================
# PROJECT SUMMARY
# =========================================================

st.markdown(
    "### 🎓 Project Summary"
)

summary_col1, summary_col2 = (
    st.columns(2)
)

with summary_col1:

    st.markdown(
        """
        **Core Modules**

        ✅ Dataset Upload  
        ✅ Data Preview  
        ✅ Data Quality Analysis  
        ✅ AI-Powered Insights  
        ✅ ML Model Recommendation  
        ✅ ML Model Training  
        ✅ Model Evaluation  
        ✅ Feature Importance  
        """
    )

with summary_col2:

    st.markdown(
        """
        **Advanced Modules**

        ✅ Correlation Analysis  
        ✅ Data Visualization  
        ✅ Natural Language Data Q&A  
        ✅ AI Report Generation  
        ✅ PDF Export  
        """
    )


st.markdown(
    """
    <div style="
        text-align:center;
        color:#6b7280;
        margin-top:40px;
        padding:20px;
    ">
        AI Data Analyst — MCA AI/ML Academic Project
    </div>
    """,
    unsafe_allow_html=True
)