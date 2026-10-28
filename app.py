from pathlib import Path
import html
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
import streamlit as st

from auth import get_supabase_client, show_auth_page


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PlacementAI | Student Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "placement_model.pkl"
PLACEMENT_DATA_PATH = BASE_DIR / "data" / "placement_dataset.csv"
JOB_DATA_PATH = BASE_DIR / "data" / "job_roles.csv"


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: Inter, sans-serif;
    }

    .stApp {
        background: #0b1020;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, 0.95);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111a32, #101426);
        border-right: 1px solid #273452;
    }

    [data-testid="stSidebar"] * {
        color: #edf2ff;
    }

    .hero {
        background: linear-gradient(120deg, #18284d, #25245a 60%, #173e55);
        padding: 30px;
        border: 1px solid #394875;
        border-radius: 20px;
        margin-bottom: 22px;
        box-sizing: border-box;
        min-width: 0;
    }

    .hero h1 {
        color: white;
        font-size: 32px;
        font-weight: 800;
        margin: 0 0 10px 0;
        overflow-wrap: anywhere;
    }

    .hero p {
        color: #d0dcff;
        font-size: 15px;
        line-height: 1.8;
        overflow-wrap: anywhere;
    }

    .eyebrow {
        color: #9cc8ff;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .section-heading {
        font-size: 22px;
        font-weight: 750;
        color: #f3f6ff;
        margin: 22px 0 12px;
        overflow-wrap: anywhere;
    }

    .metric-card,
    .feature-card {
        background: linear-gradient(145deg, #17223c, #151b30);
        border: 1px solid #2c3b5b;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 12px;
        box-sizing: border-box;
        min-width: 0;
        max-width: 100%;
        height: auto;
        overflow: visible;
        overflow-wrap: anywhere;
    }

    .metric-card {
        min-height: 115px;
    }

    .feature-card {
        min-height: 155px;
    }

    .metric-label {
        color: #a9b8d7;
        font-size: 13px;
        margin-bottom: 10px;
        overflow-wrap: anywhere;
    }

    .metric-value {
        color: white;
        font-size: 25px;
        font-weight: 800;
        line-height: 1.5;
        white-space: normal !important;
        overflow: visible !important;
        overflow-wrap: anywhere !important;
        word-break: normal !important;
        text-overflow: clip !important;
        max-width: 100%;
        height: auto !important;
        max-height: none !important;
    }

    .feature-icon {
        font-size: 27px;
        margin-bottom: 10px;
    }

    .feature-title {
        color: #f2f5ff;
        font-size: 17px;
        font-weight: 750;
        margin-bottom: 8px;
        overflow-wrap: anywhere;
    }

    .feature-description {
        color: #aebcda;
        font-size: 13px;
        line-height: 1.7;
        overflow-wrap: anywhere;
    }

    .info-banner {
        background: #172743;
        border: 1px solid #344b73;
        border-radius: 12px;
        padding: 15px;
        color: #dce8ff;
        margin: 12px 0;
        overflow-wrap: anywhere;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 650;
        border: 1px solid #526ca4;
        white-space: normal;
        height: auto;
    }

    div[data-testid="stForm"] {
        background: #121a2d;
        border: 1px solid #293956;
        border-radius: 15px;
        padding: 20px;
    }

    h1, h2, h3 {
        color: #f4f6ff;
        overflow-wrap: anywhere;
    }

    p, label {
        color: #d5def1;
    }

    [data-testid="stMetric"] {
        background: #151e33;
        border: 1px solid #293956;
        padding: 14px;
        border-radius: 13px;
        min-width: 0;
    }

    [data-testid="stMetricValue"] {
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }

    .footer {
        color: #8191b1;
        font-size: 12px;
        text-align: center;
        padding: 25px 0 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# AUTHENTICATION AND SUPABASE SESSION
# ============================================================

if "student_id" not in st.session_state:
    show_auth_page()
    st.stop()

try:
    supabase = get_supabase_client()

    access_token = st.session_state.get("access_token")
    refresh_token = st.session_state.get("refresh_token")

    if not access_token or not refresh_token:
        raise RuntimeError("Login session is missing or expired.")

    supabase.auth.set_session(access_token, refresh_token)

except Exception:
    for key in (
        "student_id",
        "student_email",
        "access_token",
        "refresh_token",
        "last_prediction",
        "last_gap_analysis",
        "pending_page",
    ):
        st.session_state.pop(key, None)

    st.warning("Your session has expired. Please log in again.")
    show_auth_page()
    st.stop()

student_id = str(st.session_state["student_id"])
student_email = st.session_state.get("student_email", "")


# ============================================================
# LOAD MODEL AND DATASETS
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(str(MODEL_PATH))

    bundle = joblib.load(MODEL_PATH)

    if isinstance(bundle, dict) and "model" in bundle:
        model_object = bundle["model"]
    else:
        model_object = bundle

    if not hasattr(model_object, "predict"):
        raise TypeError(
            "The saved model does not have a predict() method."
        )

    return model_object


@st.cache_data
def load_datasets():
    if not PLACEMENT_DATA_PATH.is_file():
        raise FileNotFoundError(str(PLACEMENT_DATA_PATH))

    if not JOB_DATA_PATH.is_file():
        raise FileNotFoundError(str(JOB_DATA_PATH))

    placement_df = pd.read_csv(PLACEMENT_DATA_PATH)
    jobs_df = pd.read_csv(JOB_DATA_PATH)

    placement_df.columns = (
        placement_df.columns.astype(str).str.strip()
    )
    jobs_df.columns = (
        jobs_df.columns.astype(str).str.strip()
    )

    return placement_df, jobs_df


try:
    model = load_model()
    placement_df, jobs_df = load_datasets()

except Exception as error:
    st.error("Unable to load the model or required CSV files.")
    st.code(str(error))
    st.info(
        "Check models/placement_model.pkl and both CSV files "
        "inside the data folder."
    )
    st.stop()


# ============================================================
# JOB DATA VALIDATION
# ============================================================

required_job_columns = [
    "Job Title",
    "Job Description",
    "Skills",
    "Certifications",
]

missing_columns = [
    column
    for column in required_job_columns
    if column not in jobs_df.columns
]

if missing_columns:
    st.error(
        "Missing columns in data/job_roles.csv: "
        + ", ".join(missing_columns)
    )
    st.stop()

for column in required_job_columns:
    jobs_df[column] = jobs_df[column].fillna("").astype(str)

job_titles = sorted(
    title.strip()
    for title in jobs_df["Job Title"].unique()
    if title.strip()
)

PREDICTION_FEATURES = [
    "CGPA",
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "ExtracurricularActivities",
    "PlacementTraining",
    "SSC_Marks",
    "HSC_Marks",
]


# ============================================================
# CLOUD STORAGE HELPERS
# ============================================================

def cloud_select(table_name):
    """Return records owned by the current student."""
    return (
        supabase.table(table_name)
        .select("*")
        .eq("user_id", student_id)
        .execute()
    )


def cloud_save(table_name, record):
    """Insert a record belonging to the current student."""
    payload = {
        **record,
        "user_id": student_id,
    }

    return (
        supabase.table(table_name)
        .insert(payload)
        .execute()
    )


def get_student_profile():
    response = (
        supabase.table("placementai_student_profiles")
        .select("*")
        .eq("user_id", student_id)
        .limit(1)
        .execute()
    )

    if response.data:
        return response.data[0]

    return {}


def save_student_profile(
    full_name,
    course,
    graduation_year,
    skills,
):
    existing = get_student_profile()

    payload = {
        "user_id": student_id,
        "full_name": full_name.strip(),
        "course": course.strip(),
        "graduation_year": int(graduation_year),
        "skills": skills.strip(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    if existing:
        return (
            supabase.table("placementai_student_profiles")
            .update(payload)
            .eq("user_id", student_id)
            .execute()
        )

    return (
        supabase.table("placementai_student_profiles")
        .insert(payload)
        .execute()
    )


def save_history(action, result, details, confidence=None):
    return cloud_save(
        "placementai_prediction_history",
        {
            "action": action,
            "prediction": str(result),
            "confidence": confidence,
            "input_data": details,
        },
    )


def get_sorted_history():
    """Return this student's saved history with newest records first."""
    response = cloud_select("placementai_prediction_history")
    records = response.data or []

    return sorted(
        records,
        key=lambda item: str(item.get("created_at") or ""),
        reverse=True,
    )


def normalize_skills(value):
    return {
        skill.strip().casefold()
        for skill in str(value).split(",")
        if skill.strip()
    }


def display_value(value):
    if isinstance(value, dict):
        return ", ".join(
            f"{str(key).replace('_', ' ').title()}: "
            f"{display_value(item)}"
            for key, item in value.items()
        )

    if isinstance(value, list):
        return ", ".join(display_value(item) for item in value)

    if value is None:
        return "Not available"

    return str(value)


def get_prediction_input(raw_data):
    """
    Align the input DataFrame with the model's expected feature names
    when feature_names_in_ is available.
    """
    frame = pd.DataFrame([raw_data])

    expected_features = getattr(model, "feature_names_in_", None)

    if expected_features is not None:
        expected_features = list(expected_features)

        missing = [
            feature
            for feature in expected_features
            if feature not in frame.columns
        ]

        extra = [
            feature
            for feature in frame.columns
            if feature not in expected_features
        ]

        if missing or extra:
            raise ValueError(
                f"Model expects features: {expected_features}. "
                f"Missing: {missing}. Extra: {extra}. "
                "Update the prediction form to match the model's "
                "training features."
            )

        frame = frame[expected_features]

    return frame


def safe_error(error):
    """Show a useful error message without exposing credentials."""
    st.caption(f"{type(error).__name__}: {error}")


def get_latest_display_result(history):
    """
    Prefer the latest unsaved placement prediction from this session.
    Otherwise, show the newest saved record from Supabase.
    """
    current_prediction = st.session_state.get("last_prediction")

    if current_prediction:
        return str(
            current_prediction.get("prediction") or "Result unavailable"
        )

    if history:
        return str(
            history[0].get("prediction") or "Result unavailable"
        )

    return "Not started"


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

PAGES = [
    "🏠 Dashboard",
    "👤 Student Profile",
    "📊 Placement Prediction",
    "💼 Job Role Recommendations",
    "🧩 Skill Gap Analysis",
    "☁️ Cloud Storage",
]

pending_page = st.session_state.pop("pending_page", None)

if pending_page in PAGES:
    st.session_state["dashboard_page"] = pending_page

if st.session_state.get("dashboard_page") not in PAGES:
    st.session_state["dashboard_page"] = PAGES[0]

with st.sidebar:
    st.markdown(
        """
        <div style="padding:12px 0 24px">
            <div style="font-size:34px">🎓</div>
            <div style="font-size:25px;font-weight:800">
                PlacementAI
            </div>
            <div style="color:#9cafd3;font-size:12px">
                YOUR CAREER COMPANION
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("WORKSPACE")

    page = st.radio(
        "Dashboard Menu",
        PAGES,
        key="dashboard_page",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**SIGNED-IN STUDENT**")
    st.caption(student_email or student_id)

    st.caption(
        "Private student workspace · Streamlit and Supabase"
    )

    if st.button("🚪 Log out", use_container_width=True):
        try:
            supabase.auth.sign_out()
        except Exception:
            pass

        # Clear session state, including cached page-level results.
        st.session_state.clear()
        st.rerun()


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">STUDENT SUCCESS PLATFORM</div>
        <h1>Build your future with PlacementAI 🚀</h1>
        <p>
            Predict placement outcomes, discover suitable IT careers,
            identify skill gaps, and securely store your progress.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HOME DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    welcome = (
        "Welcome back, " + html.escape(student_email) + "!"
        if student_email
        else "Welcome back!"
    )

    st.markdown(
        f"""
        <div class="section-heading">{welcome}</div>
        <p style="color:#aebcda">
            Your academic progress and career preparation in one place.
        </p>
        """,
        unsafe_allow_html=True,
    )

    try:
        profile = get_student_profile()
    except Exception as error:
        profile = {}
        st.warning("Could not load your profile.")
        safe_error(error)

    try:
        history = get_sorted_history()
    except Exception as error:
        history = []
        st.warning("Could not load your saved history.")
        safe_error(error)

    # FIX: Do not truncate the result to 35 characters.
    # FIX: Use sorted history so an older record is not displayed
    # just because Supabase returned it first.
    latest = get_latest_display_result(history)

    col1, col2, col3 = st.columns(3)

    with col1:
        profile_status = (
            "Complete" if profile.get("full_name") else "Get started"
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">👤 PROFILE</div>
                <div class="metric-value">
                    {html.escape(profile_status)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">📈 SAVED ANALYSES</div>
                <div class="metric-value">{len(history)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">🎯 LATEST RESULT</div>
                <div class="metric-value" style="font-size:18px;">
                    {html.escape(latest)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-heading">Explore your workspace</div>',
        unsafe_allow_html=True,
    )

    cards = [
        (
            "📊",
            "Placement Prediction",
            "Estimate your placement status using academic details.",
            "📊 Placement Prediction",
        ),
        (
            "💼",
            "Job Recommendations",
            "Explore IT roles and their required skills.",
            "💼 Job Role Recommendations",
        ),
        (
            "🧩",
            "Skill Gap Analysis",
            "Compare your current skills with job requirements.",
            "🧩 Skill Gap Analysis",
        ),
        (
            "👤",
            "Student Profile",
            "Save your academic profile and skills.",
            "👤 Student Profile",
        ),
        (
            "☁️",
            "Cloud Storage",
            "View your profile and career history.",
            "☁️ Cloud Storage",
        ),
    ]

    for start in range(0, len(cards), 2):
        row_cards = cards[start:start + 2]
        columns = st.columns(2)

        for column, card in zip(columns, row_cards):
            icon, title, description, target = card

            with column:
                st.markdown(
                    f"""
                    <div class="feature-card">
                        <div class="feature-icon">{icon}</div>
                        <div class="feature-title">
                            {html.escape(title)}
                        </div>
                        <div class="feature-description">
                            {html.escape(description)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(
                    f"Open {title} →",
                    key=f"open_{start}_{title}",
                    use_container_width=True,
                ):
                    st.session_state["pending_page"] = target
                    st.rerun()

    st.markdown(
        """
        <div class="info-banner">
            💡 Keep your profile updated and use skill-gap analysis
            to plan what to learn next. Predictions are estimates,
            not guarantees of employment.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# STUDENT PROFILE
# ============================================================

elif page == "👤 Student Profile":

    st.header("👤 Student Profile")
    st.write(
        "Maintain your academic information and current skills."
    )

    try:
        profile = get_student_profile()
    except Exception as error:
        profile = {}
        st.error("Could not load the cloud profile.")
        safe_error(error)

    with st.form("student_profile_form"):
        full_name = st.text_input(
            "Full name",
            value=str(profile.get("full_name") or ""),
        )

        course = st.text_input(
            "Course / Degree",
            value=str(profile.get("course") or ""),
            placeholder="Example: B.E. Computer Science",
        )

        try:
            existing_year = int(
                profile.get("graduation_year") or 2027
            )
        except (ValueError, TypeError):
            existing_year = 2027

        graduation_year = st.number_input(
            "Graduation year",
            min_value=2000,
            max_value=2100,
            value=max(2000, min(2100, existing_year)),
            step=1,
        )

        skills = st.text_area(
            "Current skills",
            value=str(profile.get("skills") or ""),
            placeholder="Python, SQL, HTML, CSS",
        )

        submitted = st.form_submit_button(
            "☁️ Save Profile to Cloud",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        if not full_name.strip():
            st.warning("Please enter your full name.")
        else:
            try:
                save_student_profile(
                    full_name,
                    course,
                    graduation_year,
                    skills,
                )
                st.success("Your profile was saved.")
                st.rerun()
            except Exception as error:
                st.error("Could not save your profile.")
                safe_error(error)


# ============================================================
# PLACEMENT PREDICTION
# ============================================================

elif page == "📊 Placement Prediction":

    st.header("📊 Placement Prediction")

    st.write(
        "Enter your academic details to obtain an estimate "
        "from your trained machine-learning model."
    )

    st.info(
        "A model prediction is an estimate, not a guarantee "
        "of real-world placement."
    )

    with st.form("placement_prediction_form"):
        left, right = st.columns(2)

        with left:
            cgpa = st.number_input(
                "CGPA",
                min_value=0.0,
                max_value=10.0,
                value=7.5,
                step=0.1,
            )

            internships = st.number_input(
                "Number of internships",
                min_value=0,
                max_value=20,
                value=1,
                step=1,
            )

            projects = st.number_input(
                "Number of projects",
                min_value=0,
                max_value=50,
                value=2,
                step=1,
            )

            workshops = st.number_input(
                "Workshops / Certifications",
                min_value=0,
                max_value=50,
                value=2,
                step=1,
            )

            aptitude = st.slider(
                "Aptitude test score",
                min_value=0,
                max_value=100,
                value=70,
            )

        with right:
            soft_skills = st.slider(
                "Soft skills rating",
                min_value=0.0,
                max_value=5.0,
                value=3.5,
                step=0.5,
            )

            extracurricular = st.selectbox(
                "Extracurricular activities",
                ["Yes", "No"],
            )

            training = st.selectbox(
                "Placement training",
                ["Yes", "No"],
            )

            ssc_marks = st.slider(
                "SSC marks (%)",
                min_value=0,
                max_value=100,
                value=80,
            )

            hsc_marks = st.slider(
                "HSC marks (%)",
                min_value=0,
                max_value=100,
                value=80,
            )

        predict_button = st.form_submit_button(
            "🚀 Predict Placement Status",
            use_container_width=True,
            type="primary",
        )

    if predict_button:
        raw_data = {
            "CGPA": cgpa,
            "Internships": internships,
            "Projects": projects,
            "Workshops/Certifications": workshops,
            "AptitudeTestScore": aptitude,
            "SoftSkillsRating": soft_skills,
            "ExtracurricularActivities": extracurricular,
            "PlacementTraining": training,
            "SSC_Marks": ssc_marks,
            "HSC_Marks": hsc_marks,
        }

        try:
            student_data = get_prediction_input(raw_data)
            prediction = model.predict(student_data)[0]

            confidence = None

            if hasattr(model, "predict_proba"):
                try:
                    probabilities = model.predict_proba(student_data)[0]
                    classes = list(model.classes_)

                    matching_class = next(
                        (
                            index
                            for index, label in enumerate(classes)
                            if label == prediction
                        ),
                        None,
                    )

                    if matching_class is not None:
                        confidence = round(
                            float(probabilities[matching_class]) * 100,
                            2,
                        )

                except (
                    AttributeError,
                    ValueError,
                    TypeError,
                    IndexError,
                ):
                    confidence = None

            clean_inputs = {
                str(key): (
                    value.item()
                    if hasattr(value, "item")
                    else value
                )
                for key, value in student_data.iloc[0].to_dict().items()
            }

            # FIX: Update the result immediately in session state.
            # The dashboard can now show it without requiring a
            # separate page reload or a database save first.
            st.session_state["last_prediction"] = {
                "prediction": str(prediction),
                "confidence": confidence,
                "input_data": clean_inputs,
            }

        except Exception as error:
            st.error(
                "Prediction failed. Verify the model's feature names, "
                "input types, and training-time preprocessing."
            )
            safe_error(error)

    # Render the result outside the submit-button condition.
    last_prediction = st.session_state.get("last_prediction")

    if last_prediction:
        prediction = last_prediction["prediction"]
        confidence = last_prediction["confidence"]

        st.markdown("### Your prediction")

        if prediction.strip().casefold() in {
            "placed",
            "yes",
            "1",
            "true",
        }:
            st.success(f"Model prediction: {prediction}")
        else:
            st.warning(f"Model prediction: {prediction}")

        if confidence is not None:
            st.metric(
                "Model confidence for predicted class",
                f"{confidence:.2f}%",
            )
            st.caption(
                "This model score is not the actual probability "
                "that you will get a job."
            )

        st.markdown("#### Academic details used")

        st.dataframe(
            pd.DataFrame(
                [
                    {"Field": key, "Value": value}
                    for key, value in last_prediction["input_data"].items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        if st.button(
            "☁️ Save Prediction to Cloud",
            key="save_prediction",
            use_container_width=True,
            type="primary",
        ):
            try:
                save_history(
                    "placement_prediction",
                    prediction,
                    last_prediction["input_data"],
                    confidence,
                )

                # Keep the result available in session state.
                # The dashboard will display it immediately.
                st.success(
                    "Placement prediction saved to Cloud Storage."
                )
                st.rerun()

            except Exception as error:
                st.error("Could not save the prediction.")
                safe_error(error)


# ============================================================
# JOB ROLE RECOMMENDATIONS
# ============================================================

elif page == "💼 Job Role Recommendations":

    st.header("💼 Job Role Recommendations")
    st.write("Explore roles using your job-role dataset.")

    if not job_titles:
        st.warning("No job titles were found in job_roles.csv.")
    else:
        selected_job = st.selectbox(
            "Choose an IT job role",
            job_titles,
            key="recommendation_job",
        )

        selected_rows = jobs_df[
            jobs_df["Job Title"].str.strip() == selected_job
        ]

        if not selected_rows.empty:
            job = selected_rows.iloc[0]

            st.markdown(f"### {html.escape(job['Job Title'])}")
            st.markdown("#### Job description")
            st.write(
                job["Job Description"] or "No description available."
            )

            required_skills = [
                item.strip()
                for item in job["Skills"].split(",")
                if item.strip()
            ]

            certifications = [
                item.strip()
                for item in job["Certifications"].split(",")
                if item.strip()
            ]

            left, right = st.columns(2)

            with left:
                st.markdown("#### Required skills")

                if required_skills:
                    for skill in required_skills:
                        st.write(f"• {skill}")
                else:
                    st.info("No skills listed.")

            with right:
                st.markdown("#### Suggested certifications")

                if certifications:
                    for certification in certifications:
                        st.write(f"• {certification}")
                else:
                    st.info("No certifications listed.")

            if st.button(
                "☁️ Save Job Role to Cloud History",
                key="save_job_role",
                use_container_width=True,
                type="primary",
            ):
                try:
                    save_history(
                        "job_recommendation",
                        selected_job,
                        {
                            "job_title": selected_job,
                            "skills": required_skills,
                            "certifications": certifications,
                            "job_description": job["Job Description"],
                        },
                    )

                    st.success(
                        "Job recommendation saved to Cloud Storage."
                    )

                except Exception as error:
                    st.error("Could not save this recommendation.")
                    safe_error(error)


# ============================================================
# SKILL GAP ANALYSIS
# ============================================================

elif page == "🧩 Skill Gap Analysis":

    st.header("🧩 Skill Gap Analysis")

    st.write(
        "Compare your existing skills with those required by a job role."
    )

    if not job_titles:
        st.warning("No job roles are available.")
    else:
        selected_job = st.selectbox(
            "Choose your target job role",
            job_titles,
            key="gap_job",
        )

        job_rows = jobs_df[
            jobs_df["Job Title"].str.strip() == selected_job
        ]

        # Clear stale analysis when the user switches target roles.
        existing_analysis = st.session_state.get("last_gap_analysis")

        if (
            existing_analysis
            and existing_analysis.get("job_title") != selected_job
        ):
            st.session_state.pop("last_gap_analysis", None)

        if job_rows.empty:
            st.warning("The selected job role could not be found.")
        else:
            job = job_rows.iloc[0]

            required_skills = [
                item.strip()
                for item in job["Skills"].split(",")
                if item.strip()
            ]

            try:
                profile = get_student_profile()
                default_skills = str(profile.get("skills") or "")
            except Exception:
                default_skills = ""

            student_skills_text = st.text_area(
                "Your current skills (separate with commas)",
                value=default_skills,
                placeholder="Python, SQL, HTML, CSS",
                key="gap_student_skills",
            )

            if st.button(
                "🔍 Analyze My Skill Gap",
                key="analyze_skill_gap",
                type="primary",
                use_container_width=True,
            ):
                student_skills = normalize_skills(
                    student_skills_text
                )

                required_lookup = {
                    skill.casefold(): skill
                    for skill in required_skills
                }

                matched_skills = [
                    original
                    for normalized, original in required_lookup.items()
                    if normalized in student_skills
                ]

                missing_skills = [
                    original
                    for normalized, original in required_lookup.items()
                    if normalized not in student_skills
                ]

                match_percentage = (
                    len(matched_skills) / len(required_lookup) * 100
                    if required_lookup
                    else 0.0
                )

                st.session_state["last_gap_analysis"] = {
                    "job_title": selected_job,
                    "student_skills": student_skills_text,
                    "required_skills": list(
                        required_lookup.values()
                    ),
                    "matched_skills": matched_skills,
                    "missing_skills": missing_skills,
                    "match_percentage": round(match_percentage, 1),
                }

            analysis = st.session_state.get("last_gap_analysis")

            if analysis and analysis.get("job_title") == selected_job:
                st.markdown("### Your analysis")
                st.caption(
                    f"Target job role: {analysis['job_title']}"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Required skills",
                        len(analysis["required_skills"]),
                    )

                with col2:
                    st.metric(
                        "Skills matched",
                        len(analysis["matched_skills"]),
                    )

                with col3:
                    st.metric(
                        "Skills to develop",
                        len(analysis["missing_skills"]),
                    )

                st.progress(
                    int(round(analysis["match_percentage"]))
                )

                st.write(
                    f"**Exact skill-name match: "
                    f"{analysis['match_percentage']:.1f}%**"
                )

                left, right = st.columns(2)

                with left:
                    st.markdown("#### ✅ Skills you have")

                    if analysis["matched_skills"]:
                        for skill in analysis["matched_skills"]:
                            st.write(f"• {skill}")
                    else:
                        st.info("No exact skill matches found.")

                with right:
                    st.markdown("#### 📚 Skills to learn")

                    if analysis["missing_skills"]:
                        for skill in analysis["missing_skills"]:
                            st.write(f"• {skill}")
                    else:
                        st.success(
                            "All listed skills match your input."
                        )

                st.caption(
                    "This uses exact skill-name matching and does not "
                    "measure proficiency or certify expertise."
                )

                if st.button(
                    "☁️ Save Skill Gap Analysis",
                    key="save_gap_analysis",
                    use_container_width=True,
                    type="primary",
                ):
                    try:
                        save_history(
                            "skill_gap_analysis",
                            (
                                f"{analysis['match_percentage']:.1f}% "
                                f"match for {analysis['job_title']}"
                            ),
                            analysis,
                        )

                        st.session_state.pop(
                            "last_gap_analysis", None
                        )

                        st.success(
                            "Skill gap analysis saved to Cloud Storage."
                        )
                        st.rerun()

                    except Exception as error:
                        st.error(
                            "Could not save the skill gap analysis."
                        )
                        safe_error(error)


# ============================================================
# CLOUD STORAGE
# ============================================================

elif page == "☁️ Cloud Storage":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">YOUR PERSONAL WORKSPACE</div>
            <h1>☁️ Cloud Storage</h1>
            <p>
                Review your student profile, placement predictions,
                job recommendations, and skill-gap analyses.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-banner">
            🔐 Your saved records are associated with your student account.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_profile, tab_history = st.tabs(
        ["👤 My Profile", "📚 Saved Analyses & History"]
    )

    # --------------------------------------------------------
    # PROFILE TAB
    # --------------------------------------------------------

    with tab_profile:
        st.markdown("### 👤 My Student Profile")

        try:
            profile = get_student_profile()

            if not profile:
                st.info(
                    "Your profile has not been saved yet. "
                    "Open Student Profile to add your information."
                )
            else:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">FULL NAME</div>
                            <div class="metric-value">
                                {html.escape(str(profile.get("full_name") or "Not provided"))}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">COURSE</div>
                            <div class="metric-value">
                                {html.escape(str(profile.get("course") or "Not provided"))}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                col3, col4 = st.columns(2)

                with col3:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">GRADUATION YEAR</div>
                            <div class="metric-value">
                                {html.escape(str(profile.get("graduation_year") or "Not provided"))}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col4:
                    st.markdown(
                        """
                        <div class="metric-card">
                            <div class="metric-label">ACCOUNT STATUS</div>
                            <div class="metric-value">Active</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("### 🛠️ Your Skills")

                skills = [
                    item.strip()
                    for item in str(
                        profile.get("skills") or ""
                    ).split(",")
                    if item.strip()
                ]

                if skills:
                    skill_columns = st.columns(3)

                    for index, skill in enumerate(skills):
                        with skill_columns[index % 3]:
                            st.markdown(
                                f"""
                                <div class="feature-card">
                                    <div class="feature-icon">✅</div>
                                    <div class="feature-title">
                                        {html.escape(skill)}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                else:
                    st.info("No skills have been added yet.")

                st.caption(
                    "Last updated: "
                    + str(
                        profile.get("updated_at") or "Not available"
                    )
                )

        except Exception as error:
            st.error("Unable to load your saved profile.")
            safe_error(error)

    # --------------------------------------------------------
    # HISTORY TAB
    # --------------------------------------------------------

    with tab_history:
        st.markdown("### 📚 Your Career History")

        if st.button(
            "🔄 Refresh History",
            key="refresh_cloud_history",
        ):
            st.rerun()

        try:
            # FIX: Use the same newest-first ordering as the dashboard.
            history = get_sorted_history()

            if not history:
                st.info(
                    "No saved records yet. Save a prediction, "
                    "job recommendation, or skill-gap analysis "
                    "to see it here."
                )
            else:
                st.caption(f"☁️ {len(history)} saved record(s)")

                actions = [
                    str(item.get("action") or "").lower()
                    for item in history
                ]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "📊 Predictions",
                        actions.count("placement_prediction"),
                    )

                with col2:
                    st.metric(
                        "💼 Job Recommendations",
                        actions.count("job_recommendation"),
                    )

                with col3:
                    st.metric(
                        "🧩 Skill Gap Reports",
                        actions.count("skill_gap_analysis"),
                    )

                filter_choice = st.selectbox(
                    "Filter saved records",
                    [
                        "All Records",
                        "Placement Predictions",
                        "Job Recommendations",
                        "Skill Gap Analyses",
                    ],
                    key="cloud_history_filter",
                )

                action_filters = {
                    "Placement Predictions": "placement_prediction",
                    "Job Recommendations": "job_recommendation",
                    "Skill Gap Analyses": "skill_gap_analysis",
                }

                filtered_history = [
                    item
                    for item in history
                    if (
                        filter_choice == "All Records"
                        or str(item.get("action") or "").lower()
                        == action_filters[filter_choice]
                    )
                ]

                if not filtered_history:
                    st.info("No records match this filter.")

                for index, item in enumerate(filtered_history):
                    action = str(
                        item.get("action") or "saved_record"
                    )
                    result = str(
                        item.get("prediction") or "Result unavailable"
                    )
                    created_at = str(
                        item.get("created_at") or "Date unavailable"
                    )

                    action_info = {
                        "placement_prediction": (
                            "📊",
                            "Placement Prediction",
                        ),
                        "job_recommendation": (
                            "💼",
                            "Job Recommendation",
                        ),
                        "skill_gap_analysis": (
                            "🧩",
                            "Skill Gap Analysis",
                        ),
                    }

                    icon, title = action_info.get(
                        action.lower(),
                        (
                            "📁",
                            action.replace("_", " ").title(),
                        ),
                    )

                    with st.container(border=True):
                        left, right = st.columns([3, 1])

                        with left:
                            st.markdown(f"### {icon} {title}")
                            st.markdown(f"**Result:** {result}")
                            st.caption(f"🕒 {created_at}")

                            if item.get("confidence") is not None:
                                st.caption(
                                    "Model confidence: "
                                    f"{item['confidence']}%"
                                )

                        with right:
                            st.markdown("**Cloud status**")
                            st.success("Saved")

                        details = item.get("input_data", {})

                        if isinstance(details, str):
                            try:
                                details = json.loads(details)
                            except (
                                json.JSONDecodeError,
                                TypeError,
                            ):
                                pass

                        with st.expander("🔍 View Analysis Details"):
                            if isinstance(details, dict):
                                if not details:
                                    st.caption(
                                        "No additional details saved."
                                    )
                                else:
                                    st.json(details)

                            elif isinstance(details, list):
                                st.write(details)

                            elif details:
                                st.write(display_value(details))

                            else:
                                st.caption(
                                    "No additional details available."
                                )

        except Exception as error:
            st.error(
                "Unable to load cloud history. Check your Supabase "
                "table, permissions, and column names."
            )
            safe_error(error)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        PlacementAI · Student Placement & Career Support Platform<br>
        Machine learning · Career exploration · Cloud-based storage
    </div>
    """,
    unsafe_allow_html=True,
)