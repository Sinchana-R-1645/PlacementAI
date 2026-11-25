from pathlib import Path
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
import re
import requests
import streamlit as st
from jooble_api import search_jooble
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
# MODERN DASHBOARD STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0b1020;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, 0.95);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111a32 0%, #101426 100%);
        border-right: 1px solid #273452;
    }

    [data-testid="stSidebar"] * {
        color: #edf2ff;
    }

    .hero {
        background: linear-gradient(120deg, #18284d, #25245a 60%, #173e55);
        padding: 34px;
        border: 1px solid #394875;
        border-radius: 22px;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 36px;
        font-weight: 800;
        margin: 0 0 10px 0;
    }

    .hero p {
        color: #d0dcff;
        font-size: 15px;
        margin: 0;
        line-height: 1.8;
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
        font-size: 23px;
        font-weight: 750;
        color: #f3f6ff;
        margin: 24px 0 14px 0;
    }

    .metric-card {
        background: linear-gradient(145deg, #17223c, #151b30);
        border: 1px solid #2c3b5b;
        border-radius: 16px;
        padding: 20px;
        min-height: 120px;
        margin-bottom: 12px;
    }

    .metric-label {
        color: #a9b8d7;
        font-size: 13px;
        margin-bottom: 12px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 27px;
        font-weight: 800;
    }

    .feature-card {
        background: linear-gradient(145deg, #17223c, #12182a);
        border: 1px solid #2b3b5d;
        border-radius: 17px;
        padding: 21px;
        min-height: 170px;
        margin-bottom: 10px;
    }

    .feature-icon {
        font-size: 28px;
        margin-bottom: 12px;
    }

    .feature-title {
        color: #f2f5ff;
        font-weight: 750;
        font-size: 17px;
        margin-bottom: 8px;
    }

    .feature-description {
        color: #aebcda;
        font-size: 13px;
        line-height: 1.7;
    }

    .info-banner {
        background: #172743;
        border: 1px solid #344b73;
        border-radius: 12px;
        padding: 15px;
        color: #dce8ff;
        margin: 12px 0;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 650;
        border: 1px solid #526ca4;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: #8fb6ff;
        color: #ffffff;
    }

    div[data-testid="stForm"] {
        background: #121a2d;
        border: 1px solid #293956;
        border-radius: 15px;
        padding: 20px;
    }

    h1, h2, h3 {
        color: #f4f6ff;
    }

    p, label {
        color: #d5def1;
    }

    [data-testid="stMetric"] {
        background: #151e33;
        border: 1px solid #293956;
        padding: 16px;
        border-radius: 13px;
    }

    .footer {
        color: #8191b1;
        font-size: 12px;
        text-align: center;
        padding: 25px 0 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# AUTHENTICATION
# ============================================================

if "student_id" not in st.session_state:
    show_auth_page()
    st.stop()

try:
    supabase = get_supabase_client()

    access_token = st.session_state.get("access_token")
    refresh_token = st.session_state.get("refresh_token")

    if not access_token or not refresh_token:
        raise RuntimeError("Your login session is missing.")

    supabase.auth.set_session(access_token, refresh_token)

except Exception:
    for key in (
        "student_id",
        "student_email",
        "access_token",
        "refresh_token",
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
    bundle = joblib.load(MODEL_PATH)

    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"]

    return bundle


@st.cache_data
def load_datasets():
    placement_df = pd.read_csv(PLACEMENT_DATA_PATH)
    jobs_df = pd.read_csv(JOB_DATA_PATH)
    return placement_df, jobs_df


try:
    model = load_model()
    placement_df, jobs_df = load_datasets()

except FileNotFoundError as error:
    st.error(f"Required project file not found: {error}")
    st.info(
        "Check models/placement_model.pkl and the CSV files "
        "inside the data folder."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not load the model or datasets: {error}")
    st.stop()


jobs_df.columns = jobs_df.columns.astype(str).str.strip()

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
    st.error(f"Missing columns in job_roles.csv: {missing_columns}")
    st.stop()

for column in required_job_columns:
    jobs_df[column] = jobs_df[column].fillna("")

jobs_df["Job Title"] = jobs_df["Job Title"].astype(str)

job_titles = sorted(
    title
    for title in jobs_df["Job Title"].unique().tolist()
    if title.strip()
)

def get_grouped_feature_importance(model, input_columns):
    """Return Random Forest importance grouped by original input feature."""
    try:
        if not hasattr(model, "named_steps"):
            return None

        preprocessor = model.named_steps.get("preprocessor")
        classifier = model.named_steps.get("classifier")

        if preprocessor is None or classifier is None:
            return None

        if not hasattr(classifier, "feature_importances_"):
            return None

        transformed_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_

        if len(transformed_names) != len(importances):
            return None

        grouped = {column: 0.0 for column in input_columns}

        for transformed_name, importance in zip(
            transformed_names, importances
        ):
            # Remove the transformer prefix, such as "num__".
            clean_name = transformed_name.split("__", 1)[-1]

            # Group one-hot encoded columns under their original feature.
            matches = [
                column
                for column in input_columns
                if clean_name == column
                or clean_name.startswith(f"{column}_")
            ]

            if matches:
                original_column = max(matches, key=len)
                grouped[original_column] += float(importance)

        result = pd.DataFrame(
            [
                {"Feature": feature, "Importance": importance}
                for feature, importance in grouped.items()
            ]
        )

        result = result.sort_values(
            "Importance", ascending=False
        ).reset_index(drop=True)

        return result

    except (AttributeError, KeyError, TypeError, ValueError):
        return None


# ============================================================
# CLOUD STORAGE HELPERS
# ============================================================

def cloud_select(table_name):
    """Return rows belonging to the currently authenticated student."""
    return (
        supabase.table(table_name)
        .select("*")
        .eq("user_id", student_id)
        .execute()
    )


def cloud_save(table_name, record):
    """Save a record associated with the authenticated student."""
    payload = {
        **record,
        "user_id": student_id,
    }

    return supabase.table(table_name).insert(payload).execute()


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
    experience_level="Fresher",
    work_experience_years=0,
    internship_experience_months=0,
    project_experience="",
):
    # IMPORTANT:
    # Reuse the globally authenticated Supabase client.
    # Do not create a new client inside this function.

    payload = {
        "user_id": student_id,
        "full_name": full_name.strip(),
        "course": course.strip(),
        "graduation_year": int(graduation_year),
        "skills": skills.strip(),
        "experience_level": experience_level,
        "work_experience_years": float(work_experience_years),
        "internship_experience_months": int(
            internship_experience_months
        ),
        "project_experience": project_experience.strip(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Check whether this student's profile already exists.
    existing_profile = (
        supabase.table("placementai_student_profiles")
        .select("user_id")
        .eq("user_id", student_id)
        .limit(1)
        .execute()
    )

    if existing_profile.data:
        # Update the existing profile.
        response = (
            supabase.table("placementai_student_profiles")
            .update(payload)
            .eq("user_id", student_id)
            .execute()
        )
    else:
        # Insert a new profile.
        response = (
            supabase.table("placementai_student_profiles")
            .insert(payload)
            .execute()
        )

    return response


def save_history(action, result, details):
    """Persist prediction or analysis history in Supabase."""
    return cloud_save(
        "placementai_prediction_history",
        {
            "action": action,
            "prediction": str(result),
            "confidence": None,
            "input_data": details,
        },
    )


def normalize_skills(skills_text):
    """Convert comma-separated skills into normalized skill names."""
    return {
        skill.strip().casefold()
        for skill in str(skills_text).split(",")
        if skill.strip()
    }


def render_saved_details(details):
    """Display saved analysis details in a readable format."""
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except (json.JSONDecodeError, TypeError):
            st.write(details)
            return

    if not isinstance(details, dict):
        st.write(details)
        return

    for key, value in details.items():
        label = key.replace("_", " ").title()

        if isinstance(value, (list, dict)):
            st.write(f"**{label}:**")
            st.json(value)
        else:
            st.write(f"**{label}:** {value}")


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 22px 0 18px 0;">
            <div style="font-size: 34px;">🎓</div>
            <h1 style="font-size: 30px; margin-bottom: 4px;">
                PlacementAI
            </h1>
            <div style="color: #8fb6ff; font-size: 13px;">
                YOUR CAREER COMPANION
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### WORKSPACE")

    page = st.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "👤 Student Profile",
            "📊 Placement Prediction",
            "💼 Job Role Recommendations",
            "🧩 Skill Gap Analysis",
             "🌐 Live Job Vacancies",
            "☁️ Cloud Storage",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### SIGNED-IN STUDENT")
    st.write(student_email or "Student")

    if st.button("🚪 Log Out", use_container_width=True):
        try:
            supabase.auth.sign_out()
        except Exception:
            pass

        for key in (
            "student_id",
            "student_email",
            "access_token",
            "refresh_token",
        ):
            st.session_state.pop(key, None)

        st.rerun()



# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">YOUR CAREER COMPANION</div>
            <h1>Welcome to PlacementAI 🎓</h1>
            <p>
                Predict placement readiness, explore career paths,
                identify skill gaps, and follow a personalized learning plan.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # All 7 workspace sections
    dashboard_cards = [
        (
            "👤",
            "Student Profile",
            "Manage your academic details, skills, and experience.",
        ),
        (
            "📊",
            "Placement Prediction",
            "Estimate placement readiness from your profile.",
        ),
        (
            "💼",
            "Job Role Recommendations",
            "Explore suitable job roles, required skills, and certifications.",
        ),
        (
            "🧩",
            "Skill Gap Analysis",
            "Identify missing skills and follow your learning roadmap.",
        ),
        (
            "🌐",
            "Live Job Vacancies",
            "Explore job listings and compare them with your skills.",
        ),
        (
            "☁️",
            "Cloud Storage",
            "Access your saved student information and history.",
        ),
        (
            "📈",
            "Career Overview",
            "Review your placement readiness and career progress.",
        ),
    ]

    # Display 3 cards per row
    for start in range(0, len(dashboard_cards), 3):
        card_columns = st.columns(3)

        for column, card in zip(
            card_columns,
            dashboard_cards[start:start + 3],
        ):
            icon, title, description = card

            with column:
                st.markdown(
                    f"""
                    <div class="feature-card">
                        <div class="feature-icon">{icon}</div>
                        <div class="feature-title">{title}</div>
                        <div class="feature-description">
                            {description}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Student profile overview
    try:
        profile = get_student_profile()

        st.markdown("### Your profile")

        if profile:
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Course",
                profile.get("course") or "Not specified",
            )

            col2.metric(
                "Graduation year",
                profile.get("graduation_year") or "Not specified",
            )

            col3.metric(
                "Skills listed",
                len(normalize_skills(profile.get("skills", ""))),
            )
        else:
            st.info(
                "Complete your Student Profile to personalize your experience."
            )

    except Exception as error:
        st.warning("Could not load your profile.")
        st.caption(str(error))



# ============================================================
# STUDENT PROFILE
# ============================================================


elif page == "👤 Student Profile":

    st.header("👤 Student Profile")

    st.write(
        "Maintain your academic details, skills, work experience, "
        "internships, and projects to personalize job recommendations."
    )

    try:
        profile = get_student_profile() or {}
    except Exception as error:
        profile = {}
        st.error("Could not load your student profile.")
        st.caption(str(error))

    experience_levels = [
        "Fresher",
        "Internship",
        "Entry-level",
        "Experienced",
    ]

    saved_level = profile.get("experience_level", "Fresher")
    if saved_level not in experience_levels:
        saved_level = "Fresher"

    with st.form("student_profile_form"):

        st.subheader("Academic Information")

        full_name = st.text_input(
            "Full name",
            value=profile.get("full_name", "") or "",
        )

        course = st.text_input(
            "Course / Degree",
            value=profile.get("course", "") or "",
        )

        graduation_year = st.number_input(
            "Graduation year",
            min_value=2020,
            max_value=2040,
            value=int(profile.get("graduation_year") or 2027),
            step=1,
        )

        skills = st.text_area(
            "Current skills (comma-separated)",
            value=profile.get("skills", "") or "",
            placeholder="Python, SQL, Data Structures, Cloud Computing",
        )

        st.divider()
        st.subheader("Work Experience")

        experience_level = st.selectbox(
            "Experience level",
            experience_levels,
            index=experience_levels.index(saved_level),
        )

        work_experience_years = st.number_input(
            "Full-time work experience (years)",
            min_value=0.0,
            max_value=50.0,
            value=float(profile.get("work_experience_years") or 0),
            step=0.5,
            help="Enter 0 if you have no full-time work experience.",
        )

        internship_experience_months = st.number_input(
            "Internship experience (total months)",
            min_value=0,
            max_value=120,
            value=int(
                profile.get("internship_experience_months") or 0
            ),
            step=1,
        )

        project_experience = st.text_area(
            "Project experience",
            value=profile.get("project_experience", "") or "",
            placeholder=(
                "Example: Built a cloud-based student placement "
                "prediction system using Python, Streamlit, "
                "Scikit-learn, and Supabase."
            ),
            help=(
                "Describe your projects, technologies used, and "
                "your contribution."
            ),
        )

        submitted = st.form_submit_button(
            "💾 Save Student Profile",
            use_container_width=True,
        )

    if submitted:

        if not full_name.strip() or not course.strip():
            st.error("Please enter your full name and course.")

        else:
            try:
                save_student_profile(
                    full_name=full_name,
                    course=course,
                    graduation_year=graduation_year,
                    skills=skills,
                    experience_level=experience_level,
                    work_experience_years=work_experience_years,
                    internship_experience_months=(
                        internship_experience_months
                    ),
                    project_experience=project_experience,
                )

                st.success("Student profile saved successfully!")
                st.rerun()

            except Exception as error:
                st.error("Could not save your student profile.")
                st.caption(str(error))



# ============================================================
# PLACEMENT PREDICTION
# ============================================================

elif page == "📊 Placement Prediction":

    st.header("📊 Placement Prediction")

    st.write(
        "Enter your academic and placement-preparation details "
        "to estimate your placement readiness."
    )

    feature_columns = list(placement_df.columns)

    target_candidates = [
        column
        for column in feature_columns
        if column.casefold() in {
            "placement",
            "placed",
            "placement_status",
            "status",
            "target",
        }
    ]

    target_column = target_candidates[-1] if target_candidates else None

    if target_column:
        feature_columns = [
            column
            for column in feature_columns
            if column != target_column
        ]

    profile = {}
    try:
        profile = get_student_profile()
    except Exception:
        pass

    # --------------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------------

    with st.form("placement_prediction_form"):

        input_data = {}

        columns = st.columns(2)

        for index, column in enumerate(feature_columns):

            series = placement_df[column]

            with columns[index % 2]:

                if pd.api.types.is_numeric_dtype(series):

                    median_value = pd.to_numeric(
                        series,
                        errors="coerce",
                    ).median()

                    if pd.isna(median_value):
                        median_value = 0.0

                    # Keep CGPA's existing decimal increment
                    if "cgpa" in column.casefold():

                        input_data[column] = st.number_input(
                            column.replace("_", " ").title(),
                            value=float(median_value),
                        )

                    # Other numeric fields increase by 1
                    else:

                        input_data[column] = st.number_input(
                            column.replace("_", " ").title(),
                            value=float(median_value),
                            step=1.0,
                            format="%.0f",
                        )

                else:

                    choices = [
                        str(value)
                        for value in series.dropna().unique().tolist()
                    ]

                    if not choices:
                        choices = ["Unknown"]

                    input_data[column] = st.selectbox(
                        column.replace("_", " ").title(),
                        choices,
                    )

        predict_clicked = st.form_submit_button(
            "🔮 Predict Placement Readiness",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # GENERATE PREDICTION
    # --------------------------------------------------------

    if predict_clicked:

        try:

            prediction_frame = pd.DataFrame(
                [input_data],
                columns=feature_columns,
            )

            prediction = model.predict(prediction_frame)[0]

            confidence = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    prediction_frame
                )[0]

                confidence = round(
                    float(max(probabilities)) * 100,
                    2,
                )

            # Store the latest prediction in session state.
            # This keeps it available when Streamlit reruns.
            st.session_state["latest_placement_prediction"] = {
                "prediction": str(prediction),
                "confidence": confidence,
                "input_data": input_data.copy(),
            }

            # Reset save status for the new prediction
            st.session_state["placement_prediction_saved"] = False

        except Exception as error:

            st.error("Could not generate a placement prediction.")
            st.caption(str(error))

            st.session_state.pop(
                "latest_placement_prediction",
                None,
            )

            st.session_state.pop(
                "placement_prediction_saved",
                None,
            )

    # --------------------------------------------------------
    # DISPLAY RESULT AND SAVE BUTTON
    # Keep this section OUTSIDE if predict_clicked
    # --------------------------------------------------------

    latest = st.session_state.get(
        "latest_placement_prediction"
    )

    if latest:

        st.markdown("### Prediction Result")

        st.success(
            f"Predicted result: **{latest['prediction']}**"
        )

        if latest["confidence"] is not None:

            st.metric(
                "Model confidence",
                f"{latest['confidence']:.1f}%",
            )

        # Save only when the user clicks the button
        if st.session_state.get("placement_prediction_saved", False):

            st.success("Prediction saved to your cloud history.")

        else:

            if st.button(
                "☁️ Save Prediction to Cloud",
                key="save_placement_prediction",
                use_container_width=True,
            ):

                try:

                    save_history(
                        "placement_prediction",
                        latest["prediction"],
                        {
                            "input_data": latest["input_data"],
                            "confidence": latest["confidence"],
                        },
                    )

                    st.session_state["placement_prediction_saved"] = True

                    st.success(
                        "Prediction saved to your cloud history!"
                    )

                except Exception as error:

                    st.error(
                        "Prediction was made, but could not be saved to cloud."
                    )

                    st.caption(str(error))


    # ============================================================
    # PREDICTION EXPLAINABILITY
    # ============================================================

    st.markdown("---")
    st.subheader("🔍 Understanding Placement Predictions")

    st.write(
        "Explore which academic and preparation-related factors "
        "the trained model relies on when making predictions."
    )

    importance_df = get_grouped_feature_importance(
        model, feature_columns
    )

    if importance_df is not None and not importance_df.empty:
        importance_df["Importance (%)"] = (
            importance_df["Importance"] * 100
        )

        st.markdown("#### Overall Feature Importance")

        chart_df = importance_df.set_index("Feature")[
            "Importance (%)"
        ]

        st.bar_chart(chart_df.head(10))

        st.caption(
            "Higher values mean the model relied more on that feature "
            "across its trees. This does not mean the feature causes "
            "placement or guarantees a particular outcome."
        )

        st.markdown("#### Top Factors Used by the Model")

        top_features = importance_df.head(5)

        for _, row in top_features.iterrows():
            st.write(
                f"- **{row['Feature'].replace('_', ' ').title()}**: "
                f"{row['Importance (%)']:.2f}% importance"
            )

    else:
        st.info(
            "Feature importance is unavailable for the loaded model. "
            "The prediction feature will continue to work normally."
        )

    latest_explanation = st.session_state.get(
        "latest_placement_prediction"
    )

    if latest_explanation:
        st.markdown("#### Explanation of Your Latest Prediction")

        st.write(
            f"The model predicted **{latest_explanation['prediction']}** "
            "using the details entered in the prediction form."
        )

        if latest_explanation.get("confidence") is not None:
            st.write(
                f"Model confidence score: "
                f"**{latest_explanation['confidence']:.2f}%**"
            )

        with st.expander("View the details used for this prediction"):
            details_df = pd.DataFrame(
                list(latest_explanation["input_data"].items()),
                columns=["Input Feature", "Entered Value"],
            )
            st.dataframe(details_df, use_container_width=True)

        st.caption(
            "The feature-importance chart explains overall model "
            "behaviour; it does not calculate each feature's individual "
            "contribution to this particular prediction. The confidence "
            "score is not a guarantee of actual placement."
        )

# ============================================================
# JOB ROLE RECOMMENDATIONS
# ============================================================

elif page == "💼 Job Role Recommendations":

    st.header("💼 Job Role Recommendations")

    st.write(
        "Explore job roles, their descriptions, required skills, "
        "and suggested certifications."
    )

    if not job_titles:
        st.warning("No job roles are available in job_roles.csv.")

    else:
        selected_job = st.selectbox(
            "Choose a job role",
            job_titles,
            key="recommendation_job",
        )

        job = jobs_df[
            jobs_df["Job Title"] == selected_job
        ].iloc[0]

        st.markdown(f"### {selected_job}")

        description = str(job["Job Description"]).strip()
        st.write(description or "No description is available.")

        required_skills = [
            item.strip()
            for item in str(job["Skills"]).split(",")
            if item.strip()
        ]

        certifications = [
            item.strip()
            for item in str(job["Certifications"]).split(",")
            if item.strip()
        ]

        left, right = st.columns(2)

        with left:
            st.markdown("#### Required skills")
            if required_skills:
                for skill in required_skills:
                    st.write(f"• {skill}")
            else:
                st.info("No skills listed for this role.")

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
        ):
            try:
                save_history(
                    "job_recommendation",
                    selected_job,
                    {
                        "job_title": selected_job,
                        "skills": required_skills,
                        "certifications": certifications,
                    },
                )
                st.success("Job recommendation saved.")
            except Exception as error:
                st.error("Could not save this recommendation.")
                st.caption(str(error))
 # ============================================================
# SKILL GAP ANALYSIS + GUIDED LEARNING ROADMAP
# ============================================================

elif page == "🧩 Skill Gap Analysis":

    st.header("🧩 Skill Gap Analysis & Guided Learning Roadmap")
    st.write(
        "Compare your skills with a target job role, learn missing skills "
        "through daily tasks, practise, build projects, and take a quiz."
    )

    ROADMAP_TABLE = "placementai_skill_roadmaps"

    SKILL_PLANS = {
        "python": {
            "url": "https://docs.python.org/3/tutorial/",
            "hours": 8,
            "lessons": [
                ("Day 1: Python basics", "Learn variables, data types, print(), and input().", "Write a program that asks for your name and displays a greeting.", 60),
                ("Day 2: Conditions and loops", "Practise if/else, for loops, while loops, and range().", "Print a multiplication table and classify numbers.", 60),
                ("Day 3: Lists and dictionaries", "Learn lists, tuples, sets, dictionaries, and iteration.", "Store five marks and calculate their average.", 60),
                ("Day 4: Functions", "Learn parameters, return values, and reusable functions.", "Write functions to calculate average marks and grades.", 60),
                ("Day 5: Files and errors", "Read files and handle exceptions using try/except.", "Read marks from a file and handle invalid input.", 60),
                ("Day 6: Practice", "Solve beginner programming problems.", "Solve five problems using loops, collections, and functions.", 90),
                ("Day 7: Mini-project", "Combine the concepts learned throughout the week.", "Build a Student Marks Analyser that calculates grades and identifies the topper.", 90),
            ],
            "quiz": [
                ("Which data structure stores ordered, changeable items?", ["List", "Tuple", "Set", "Integer"], 0),
                ("What does return do in a function?", ["Repeats a loop", "Returns a result to the caller", "Imports a module", "Creates a variable"], 1),
                ("Which statement handles exceptions?", ["if", "for", "try/except", "print"], 2),
            ],
            "project": [
                "Ask the user for student names and marks.",
                "Validate marks and store student records.",
                "Write functions to calculate averages and grades.",
                "Display the class average and highest-scoring student.",
                "Test invalid inputs and optionally export results to CSV.",
            ],
        },
        "sql": {
            "url": "https://sqlbolt.com/",
            "hours": 6,
            "lessons": [
                ("Day 1: SELECT and FROM", "Understand tables, rows, columns, SELECT, and FROM.", "Select specific columns from a sample student table.", 45),
                ("Day 2: Filtering", "Practise WHERE, AND, OR, LIKE, IN, and ORDER BY.", "Find students above a selected mark and sort the results.", 60),
                ("Day 3: Aggregates", "Learn COUNT, SUM, AVG, MIN, MAX, and GROUP BY.", "Calculate average marks by department.", 60),
                ("Day 4: JOIN operations", "Understand primary keys, foreign keys, and INNER/LEFT JOIN.", "Join students with their department information.", 60),
                ("Day 5: Data changes", "Practise INSERT, UPDATE, DELETE, and CREATE TABLE.", "Create a courses table and insert sample records.", 60),
                ("Day 6: Practice", "Combine filters, aggregates, and joins.", "Write five queries for a student placement report.", 60),
                ("Day 7: Mini-project", "Create a small SQL reporting project.", "Produce a placement report showing student names, scores, and department averages.", 75),
            ],
            "quiz": [
                ("Which clause filters individual rows?", ["WHERE", "ORDER BY", "SELECT", "LIMIT"], 0),
                ("Which function counts rows?", ["TOTAL()", "COUNT()", "NUMBER()", "ROWS()"], 1),
                ("Which JOIN preserves every row from the left table?", ["CROSS JOIN", "INNER JOIN", "LEFT JOIN", "SELF JOIN"], 2),
            ],
            "project": [
                "Create Students, Departments, and Scores tables.",
                "Insert at least five sample students.",
                "Write queries to filter students by score.",
                "Calculate average scores by department.",
                "Document each query and test the results.",
            ],
        },
        "html": {
            "url": "https://developer.mozilla.org/en-US/docs/Learn_web_development/Core/Structuring_content",
            "hours": 4,
            "lessons": [
                ("Day 1: HTML structure", "Learn doctype, html, head, title, and body.", "Create a valid webpage with a heading and paragraph.", 30),
                ("Day 2: Text and links", "Practise headings, lists, links, and emphasis.", "Create an About Me webpage.", 35),
                ("Day 3: Images and semantic HTML", "Learn image alt text, header, main, section, and footer.", "Organize your webpage into semantic sections.", 35),
                ("Day 4: Forms and tables", "Learn labels, inputs, buttons, rows, and table headers.", "Build a contact form and education table.", 40),
                ("Day 5: Mini-project", "Combine the HTML concepts you learned.", "Build a portfolio page with About, Skills, Projects, and Contact sections.", 60),
            ],
            "quiz": [
                ("What is the purpose of image alt text?", ["Set width", "Describe the image accessibly", "Change format", "Set page title"], 1),
                ("Which element contains the main page content?", ["main", "meta", "head", "title"], 0),
                ("How is a label associated with an input?", ["Matching for and id", "Matching page titles", "Using image alt", "Using a table caption"], 0),
            ],
            "project": [
                "Create an index.html file.",
                "Add About, Skills, Projects, and Contact sections.",
                "Add links to your projects.",
                "Use meaningful image alt text if images are included.",
                "Open the page in a browser and fix any layout or markup issues.",
            ],
        },
        "css": {
            "url": "https://developer.mozilla.org/en-US/docs/Learn_web_development/Core/Styling_basics",
            "hours": 5,
            "lessons": [
                ("Day 1: CSS basics", "Learn selectors, properties, values, and stylesheets.", "Change the colors and fonts of a webpage.", 45),
                ("Day 2: Box model", "Practise margin, padding, border, and width.", "Create three cards with consistent spacing.", 45),
                ("Day 3: Flexbox", "Learn display:flex, alignment, and gaps.", "Create a responsive navigation bar.", 45),
                ("Day 4: Responsive design", "Use media queries and flexible widths.", "Make a page work on mobile and desktop.", 60),
                ("Day 5: Mini-project", "Combine CSS layout and responsive design.", "Style your portfolio with cards, navigation, and a mobile layout.", 75),
            ],
            "quiz": [
                ("Which property changes text color?", ["font-size", "color", "padding", "display"], 1),
                ("Which CSS layout is useful for one-dimensional alignment?", ["Flexbox", "SQL", "HTML", "JSON"], 0),
                ("What does padding control?", ["Space inside an element's border", "Text content", "Database rows", "Page title"], 0),
            ],
            "project": [
                "Create a separate styles.css file.",
                "Style headings, text, buttons, and links.",
                "Use Flexbox to arrange project cards.",
                "Add a media query for small screens.",
                "Test the page at desktop and mobile widths.",
            ],
        },
        "javascript": {
            "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide",
            "hours": 8,
            "lessons": [
                ("Day 1: Variables and values", "Learn let, const, strings, numbers, and booleans.", "Create a calculator for two numbers.", 60),
                ("Day 2: Conditions and loops", "Practise if/else, for, and while.", "Print numbers and classify even and odd values.", 60),
                ("Day 3: Arrays and objects", "Store and access data in arrays and objects.", "Represent a student using a JavaScript object.", 60),
                ("Day 4: Functions", "Write reusable functions and arrow functions.", "Create functions for addition, subtraction, and multiplication.", 60),
                ("Day 5: DOM basics", "Select HTML elements and update their content.", "Make a button change a heading when clicked.", 60),
                ("Day 6: Practice", "Solve small coding exercises.", "Complete five array, string, and function exercises.", 60),
                ("Day 7: Mini-project", "Combine JavaScript with HTML and CSS.", "Build an interactive to-do list.", 90),
            ],
            "quiz": [
                ("Which keyword declares a block-scoped variable that can be reassigned?", ["let", "const", "class", "import"], 0),
                ("Which method selects an element using a CSS selector?", ["document.querySelector()", "console.log()", "JSON.parse()", "Math.round()"], 0),
                ("Which array method creates a new array by transforming items?", ["map()", "push()", "pop()", "shift()"], 0),
            ],
            "project": [
                "Create an HTML page with a text input and Add button.",
                "Use JavaScript to read the input value.",
                "Add new tasks to a list.",
                "Allow users to mark tasks complete or remove them.",
                "Test empty inputs and multiple tasks.",
            ],
        },
        "machine learning": {
            "url": "https://scikit-learn.org/stable/tutorial/index.html",
            "hours": 10,
            "lessons": [
                ("Day 1: ML concepts", "Learn features, labels, training data, and testing data.", "Identify features and labels in a sample dataset.", 60),
                ("Day 2: Data preparation", "Explore missing values, duplicates, and categorical data.", "Clean a small CSV dataset using pandas.", 75),
                ("Day 3: Train/test split", "Learn why data is separated for training and evaluation.", "Split a dataset into training and test sets.", 60),
                ("Day 4: Classification", "Train a simple classification model.", "Train a decision tree classifier.", 75),
                ("Day 5: Evaluation", "Learn accuracy, precision, recall, and confusion matrices.", "Evaluate a model and explain its errors.", 75),
                ("Day 6: Practice", "Compare models and avoid data leakage.", "Compare two classifiers using the same test set.", 90),
                ("Day 7: Mini-project", "Build an end-to-end ML workflow.", "Train and evaluate a model on a public dataset and document its limitations.", 120),
            ],
            "quiz": [
                ("Why keep a test dataset?", ["To evaluate generalization on unseen data", "To increase column names", "To remove all labels", "To replace training"], 0),
                ("Which metric is the fraction of predictions that are correct?", ["Accuracy", "Recall", "Support", "Feature count"], 0),
                ("What is data leakage?", ["Using information unavailable at prediction time", "Removing duplicates", "Saving a CSV", "Plotting a chart"], 0),
            ],
            "project": [
                "Choose a public, labelled dataset.",
                "Inspect missing values and clean the data.",
                "Split the data into training and test sets.",
                "Train a baseline classifier.",
                "Report metrics and describe possible limitations.",
            ],
        },
    }

    def make_generic_plan(skill):
        return {
            "url": "https://infyspringboard.onwingspan.com/",
            "hours": 5,
            "lessons": [
                ("Day 1: Understand the fundamentals", f"Find an introductory lesson covering the core concepts of {skill}.", f"Write down five important concepts related to {skill}.", 45),
                ("Day 2: Learn the basic tools", f"Identify the common tools and workflow used for {skill}.", f"Complete one beginner exercise involving {skill}.", 45),
                ("Day 3: Guided practice", f"Follow a beginner tutorial for {skill}.", f"Repeat the tutorial without copying every step.", 60),
                ("Day 4: Independent practice", f"Apply {skill} to a small realistic problem.", f"Solve two exercises and note any mistakes.", 60),
                ("Day 5: Mini-project", f"Build a small project demonstrating {skill}.", f"Document your work and explain what you learned.", 90),
            ],
            "quiz": [
                (f"What is a good first step when learning {skill}?", ["Understand the fundamentals", "Skip all practice", "Avoid examples", "Memorize without understanding"], 0),
                ("Why should you practise independently?", ["To check whether you can apply concepts", "To avoid learning", "To remove feedback", "To skip fundamentals"], 0),
                ("What should a project demonstrate?", ["Practical application of the skill", "Only a title", "No working output", "Copied text alone"], 0),
            ],
            "project": [
                f"Define a small problem that can be solved using {skill}.",
                "Write down the requirements and expected result.",
                "Complete a beginner tutorial related to the problem.",
                "Build your own solution and test it.",
                "Document the result and what you would improve.",
            ],
        }

    def get_plan(skill):
        return SKILL_PLANS.get(skill.casefold(), make_generic_plan(skill))

    def load_skill_roadmap(job_title):
        result = (
            supabase.table(ROADMAP_TABLE)
            .select("*")
            .eq("user_id", student_id)
            .eq("job_title", job_title)
            .execute()
        )
        return result.data or []

    def save_skill_roadmap_item(job_title, item):
        payload = {
            "user_id": student_id,
            "job_title": job_title,
            "skill": item["skill"],
            "priority": item.get("priority", "Medium"),
            "learning_stage": item.get("learning_stage", "Beginner"),
            "estimated_hours": int(item.get("estimated_hours", 5)),
            "status": item.get("status", "Not Started"),
            "resource_url": item.get("resource_url", ""),
            "project_idea": item.get("project_idea", ""),
            "lessons": item.get("lessons", []),
            "quiz": item.get("quiz", []),
            "project_steps": item.get("project_steps", []),
            "completed_lessons": item.get("completed_lessons", []),
            "quiz_score": item.get("quiz_score"),
            "quiz_passed": item.get("quiz_passed", False),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        return (
            supabase.table(ROADMAP_TABLE)
            .upsert(
                payload,
                on_conflict="user_id,job_title,skill",
            )
            .execute()
        )

    def delete_skill_roadmap(job_title):
        return (
            supabase.table(ROADMAP_TABLE)
            .delete()
            .eq("user_id", student_id)
            .eq("job_title", job_title)
            .execute()
        )

    if not job_titles:
        st.warning("No job roles were found in data/job_roles.csv.")

    else:
        selected_job = st.selectbox(
            "Choose your target job role",
            job_titles,
            key="skill_gap_job",
        )

        job_rows = jobs_df[jobs_df["Job Title"] == selected_job]

        if job_rows.empty:
            st.error("The selected job role could not be found.")
        else:
            job = job_rows.iloc[0]

            required_skills = [
                skill.strip()
                for skill in str(job["Skills"]).split(",")
                if skill.strip()
            ]

            try:
                profile = get_student_profile()
                default_skills = profile.get("skills", "")
            except Exception:
                default_skills = ""

            current_skills = st.text_area(
                "Enter your current skills, separated by commas",
                value=default_skills,
                placeholder="Python, SQL, HTML, Communication",
                key="skill_gap_current_skills",
            )

            if st.button(
                "🔍 Analyze My Skill Gap",
                type="primary",
                use_container_width=True,
            ):
                student_skills = normalize_skills(current_skills)

                matched = [
                    skill for skill in required_skills
                    if skill.casefold() in student_skills
                ]

                missing = [
                    skill for skill in required_skills
                    if skill.casefold() not in student_skills
                ]

                percentage = (
                    len(matched) / len(required_skills) * 100
                    if required_skills else 0
                )

                st.session_state["last_gap_analysis"] = {
                    "job_title": selected_job,
                    "current_skills": current_skills,
                    "required_skills": required_skills,
                    "matched": matched,
                    "missing": missing,
                    "percentage": percentage,
                }

            analysis = st.session_state.get("last_gap_analysis")

            if analysis and analysis.get("job_title") == selected_job:
                matched = analysis["matched"]
                missing = analysis["missing"]
                percentage = analysis["percentage"]

                st.subheader("Your Skill Gap Results")
                c1, c2, c3 = st.columns(3)
                c1.metric("Required skills", len(required_skills))
                c2.metric("Skills matched", len(matched))
                c3.metric("Skills to learn", len(missing))

                st.progress(int(percentage))
                st.write(f"**Skill match: {percentage:.1f}%**")

                left, right = st.columns(2)

                with left:
                    st.markdown("#### ✅ Skills you have")
                    if matched:
                        for skill in matched:
                            st.write(f"• {skill}")
                    else:
                        st.info("No exact matches found yet.")

                with right:
                    st.markdown("#### 📚 Skills to learn")
                    if missing:
                        for skill in missing:
                            st.write(f"• {skill}")
                    else:
                        st.success("You have listed all required skills.")

                if st.button(
                    "☁️ Save Skill Gap Analysis",
                    key="save_gap_analysis",
                ):
                    try:
                        save_history(
                            "skill_gap_analysis",
                            f"{percentage:.1f}% match for {selected_job}",
                            {
                                "job_title": selected_job,
                                "current_skills": current_skills,
                                "matched_skills": matched,
                                "missing_skills": missing,
                                "match_percentage": round(percentage, 1),
                            },
                        )
                        st.success("Skill-gap analysis saved.")
                    except Exception as error:
                        st.error("Could not save the analysis.")
                        st.caption(str(error))

                st.divider()
                st.subheader("🗺️ Your Guided Learning Roadmap")

                if missing:
                    st.write(
                        "Generate a guided plan for each missing skill. "
                        "Each plan includes daily topics, exercises, a project, and a quiz."
                    )

                    if st.button(
                        "✨ Generate / Update My Roadmap",
                        type="primary",
                        use_container_width=True,
                    ):
                        try:
                            old_items = load_skill_roadmap(selected_job)
                            old_by_skill = {
                                item["skill"].casefold(): item
                                for item in old_items
                            }

                            for index, skill in enumerate(missing):
                                plan = get_plan(skill)
                                old = old_by_skill.get(skill.casefold(), {})

                                item = {
                                    "skill": skill,
                                    "priority": "High" if index == 0 else "Medium",
                                    "learning_stage": "Beginner" if index == 0 else "Intermediate",
                                    "estimated_hours": plan["hours"],
                                    "status": old.get("status", "Not Started"),
                                    "resource_url": plan["url"],
                                    "project_idea": plan["project"][-1] if plan.get("project") else f"Build a small project using {skill}.",
                                    "lessons": plan["lessons"],
                                    "quiz": plan["quiz"],
                                    "project_steps": plan["project"],
                                    "completed_lessons": old.get("completed_lessons", []),
                                    "quiz_score": old.get("quiz_score"),
                                    "quiz_passed": old.get("quiz_passed", False),
                                }

                                save_skill_roadmap_item(selected_job, item)

                            st.success("Your guided roadmap has been saved.")
                            st.rerun()

                        except Exception as error:
                            st.error(
                                "Could not generate the roadmap. Check the "
                                "Supabase table schema and permissions."
                            )
                            st.caption(str(error))

                try:
                    roadmap_items = load_skill_roadmap(selected_job)

                    if not roadmap_items:
                        st.info(
                            "Click Generate / Update My Roadmap to create "
                            "your daily learning plans."
                        )
                    else:
                        total = len(roadmap_items)
                        completed = sum(
                            item.get("status") == "Completed"
                            for item in roadmap_items
                        )
                        total_hours = sum(
                            int(item.get("estimated_hours") or 0)
                            for item in roadmap_items
                        )

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Roadmap skills", total)
                        m2.metric("Completed skills", f"{completed}/{total}")
                        m3.metric("Study hours", total_hours)

                        completion = completed / total * 100 if total else 0
                        st.progress(int(completion))
                        st.write(f"**Roadmap completion: {completion:.1f}%**")

                        for item_index, item in enumerate(roadmap_items):
                            skill = item.get("skill", "Skill")
                            plan = get_plan(skill)

                            with st.expander(
                                f"📘 {skill} — {item.get('priority', 'Medium')} priority",
                                expanded=False,
                            ):
                                st.write(
                                    f"**Estimated time:** "
                                    f"{item.get('estimated_hours', plan['hours'])} hours"
                                )
                                st.write(
                                    f"**Learning stage:** "
                                    f"{item.get('learning_stage', 'Beginner')}"
                                )

                                resource_url = "https://infyspringboard.onwingspan.com/"
                                st.link_button(
                                    "Open Infosys Springboard",
                                     resource_url,
                                )

                                st.markdown("### Daily learning schedule")

                                lessons = item.get("lessons") or plan["lessons"]
                                if isinstance(lessons, str):
                                    try:
                                        lessons = json.loads(lessons)
                                    except json.JSONDecodeError:
                                        lessons = plan["lessons"]

                                completed_lessons = item.get("completed_lessons") or []
                                if isinstance(completed_lessons, str):
                                    try:
                                        completed_lessons = json.loads(completed_lessons)
                                    except json.JSONDecodeError:
                                        completed_lessons = []

                                updated_completed = []

                                for lesson_index, lesson in enumerate(lessons):
                                    title, topic, exercise, minutes = lesson

                                    is_done = st.checkbox(
                                        f"{title} ({minutes} minutes)",
                                        value=lesson_index in completed_lessons,
                                        key=f"lesson_{selected_job}_{item_index}_{lesson_index}",
                                    )

                                    st.write(f"**Study:** {topic}")
                                    st.write(f"**Practice:** {exercise}")

                                    if is_done:
                                        updated_completed.append(lesson_index)

                                st.markdown("### Mini-project instructions")
                                project_steps = item.get("project_steps") or plan["project"]

                                if isinstance(project_steps, str):
                                    try:
                                        project_steps = json.loads(project_steps)
                                    except json.JSONDecodeError:
                                        project_steps = plan["project"]

                                for step_index, step in enumerate(project_steps, 1):
                                    st.write(f"{step_index}. {step}")

                               
                                st.markdown("### Save lesson progress")

                                status_options = [
                                    "Not Started",
                                    "In Progress",
                                    "Completed",
                                ]

                                current_status = item.get("status", "Not Started")
                                status = st.selectbox(
                                    "Overall skill status",
                                    status_options,
                                    index=(
                                        status_options.index(current_status)
                                        if current_status in status_options else 0
                                    ),
                                    key=f"status_{selected_job}_{item_index}",
                                )

                                if st.button(
                                    "💾 Save lesson checkboxes and status",
                                    key=f"save_lessons_{selected_job}_{item_index}",
                                    use_container_width=True,
                                ):
                                    try:
                                        updated = {
                                            **item,
                                            "completed_lessons": updated_completed,
                                            "status": status,
                                        }
                                        save_skill_roadmap_item(selected_job, updated)
                                        st.success("Your progress has been saved.")
                                        st.rerun()
                                    except Exception as error:
                                        st.error("Could not save your progress.")
                                        st.caption(str(error))

                        if st.button(
                            "🗑️ Reset roadmap for this job role",
                            key="reset_skill_roadmap",
                        ):
                            try:
                                delete_skill_roadmap(selected_job)
                                st.success("Roadmap reset.")
                                st.rerun()
                            except Exception as error:
                                st.error("Could not reset the roadmap.")
                                st.caption(str(error))

                except Exception as error:
                    st.error(
                        "Could not load the roadmap from Supabase. "
                        "Check the table columns, permissions, and RLS policies."
                    )
                    st.caption(str(error))










# ============================================================
# LIVE JOB VACANCIES
# Adzuna + Jooble
# Fixes:
# 1. Accurate old-listing hidden counts
# 2. Separate undated listings
# 3. Consistent date and experience categories
# ============================================================


elif page == "🌐 Live Job Vacancies":

    import re
    import requests
    import streamlit as st
    from datetime import datetime, timezone, timedelta
    from urllib.parse import urlparse
    from difflib import SequenceMatcher

    from jooble_api import search_jooble

    st.header("🌐 Live Job Vacancies")
    st.write(
        "Search job listings from Adzuna and Jooble, compare them with "
        "your saved skills, and review experience requirements."
    )
    st.caption(
        "Skill matching and experience classifications are estimates. "
        "Always verify the requirements on the original vacancy page."
    )

    # ------------------------------------------------------------
    # 1. GENERAL HELPERS
    # ------------------------------------------------------------

    def clean_text(value):
        """Convert text or nested API fields into readable text."""
        if value is None:
            return ""

        if isinstance(value, dict):
            for key in (
                "display_name", "name", "label", "value",
                "description", "title"
            ):
                if value.get(key):
                    return clean_text(value[key])

            return ""

        if isinstance(value, list):
            return ", ".join(
                clean_text(item) for item in value if item is not None
            )

        return re.sub(r"\s+", " ", str(value)).strip()

    def normalized_text(value):
        value = clean_text(value).lower()
        value = value.replace("node.js", "nodejs")
        value = value.replace("c++", "cplusplus")
        value = value.replace("c#", "csharp")
        value = value.replace("&", " and ")
        return re.sub(r"\s+", " ", value).strip()

    def get_first(data, keys, default=""):
        if not isinstance(data, dict):
            return default

        for key in keys:
            value = data.get(key)
            if value is not None and value != "":
                return value

        return default

    def normalize_url(url):
        url = clean_text(url)
        if not url:
            return ""

        if url.startswith("//"):
            url = "https:" + url
        elif url.startswith("www."):
            url = "https://" + url
        elif url and not url.startswith(("http://", "https://")):
            return ""

        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                return ""

            # Remove tracking query strings and fragments for deduplication.
            return (
                parsed.scheme.lower()
                + "://"
                + parsed.netloc.lower()
                + parsed.path.rstrip("/")
            )
        except Exception:
            return ""

    def parse_posted_date(value):
        """Return a timezone-aware datetime or None."""
        if value is None or value == "":
            return None

        if isinstance(value, (int, float)):
            try:
                # Handle Unix timestamps in seconds or milliseconds.
                if value > 10_000_000_000:
                    value = value / 1000
                return datetime.fromtimestamp(value, tz=timezone.utc)
            except (ValueError, OverflowError, OSError):
                return None

        text_value = clean_text(value)
        if not text_value:
            return None

        try:
            parsed = datetime.fromisoformat(
                text_value.replace("Z", "+00:00")
            )
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except (ValueError, TypeError):
            pass

        formats = [
            "%d %b %Y", "%d %B %Y",
            "%Y-%m-%d", "%Y/%m/%d",
            "%d-%m-%Y", "%m/%d/%Y",
        ]

        for date_format in formats:
            try:
                return datetime.strptime(
                    text_value, date_format
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        # Relative dates such as "2 days ago".
        match = re.search(
            r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago",
            text_value.lower()
        )

        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            now = datetime.now(timezone.utc)

            if unit == "minute":
                return now - timedelta(minutes=amount)
            if unit == "hour":
                return now - timedelta(hours=amount)
            if unit == "day":
                return now - timedelta(days=amount)
            if unit == "week":
                return now - timedelta(weeks=amount)
            if unit == "month":
                return now - timedelta(days=30 * amount)
            if unit == "year":
                return now - timedelta(days=365 * amount)

        if text_value.lower() in ("today", "just now"):
            return datetime.now(timezone.utc)

        if text_value.lower() == "yesterday":
            return datetime.now(timezone.utc) - timedelta(days=1)

        return None

    def normalize_adzuna_job(job):
        """Convert Adzuna nested fields to plain readable strings."""
        company = clean_text(job.get("company"))
        location = clean_text(job.get("location"))

        return {
            "title": clean_text(job.get("title")) or "Untitled vacancy",
            "company": company or "Company not provided",
            "location": location or "Location not provided",
            "description": clean_text(job.get("description")),
            "url": clean_text(
                get_first(job, ["redirect_url", "url", "link"])
            ),
            "posted": get_first(
                job, ["created", "posted", "posted_at", "date"]
            ),
            "source": "Adzuna",
            "job_type": clean_text(job.get("contract_type")),
            "salary_min": job.get("salary_min"),
            "salary_max": job.get("salary_max"),
        }

    def normalize_jooble_job(job):
        """Normalize results from the existing Jooble helper."""
        return {
            "title": clean_text(
                get_first(job, ["title", "position"])
            ) or "Untitled vacancy",
            "company": clean_text(
                get_first(job, ["company", "company_name"])
            ) or "Company not provided",
            "location": clean_text(
                get_first(job, ["location", "city"])
            ) or "Location not provided",
            "description": clean_text(
                get_first(job, ["description", "snippet"])
            ),
            "url": clean_text(
                get_first(job, ["url", "link", "job_url"])
            ),
            "posted": get_first(
                job, ["updated", "updated_at", "date", "posted"]
            ),
            "source": "Jooble",
            "job_type": clean_text(
                get_first(job, ["type", "job_type"])
            ),
            "salary_min": None,
            "salary_max": None,
        }

    # ------------------------------------------------------------
    # 2. SKILL MATCHING
    # Do not match single-letter "c": it creates false positives.
    # ------------------------------------------------------------

    SKILL_ALIASES = {
        "python": [r"\bpython\b"],
        "java": [r"\bjava\b"],
        "sql": [r"\bsql\b"],
        "data structures": [
            r"\bdata structures\b",
            r"\bdata structure and algorithms\b",
            r"\bdsa\b",
        ],
        "data analytics": [
            r"\bdata analytics\b",
            r"\bdata analysis\b",
            r"\bdata analyst\b",
            r"\bdata analytics tools\b",
        ],
        "cloud computing": [
            r"\bcloud computing\b",
            r"\bcloud services\b",
            r"\baws\b",
            r"\bazure\b",
            r"\bgcp\b",
            r"\bgoogle cloud\b",
        ],
        "machine learning": [
            r"\bmachine learning\b",
            r"\bml\b",
        ],
        "rest api": [
            r"\brest api\b",
            r"\brestful api\b",
            r"\brestful services\b",
        ],
        "html": [r"\bhtml5?\b"],
        "css": [r"\bcss3?\b"],
        "javascript": [r"\bjavascript\b"],
        "typescript": [r"\btypescript\b"],
        "react": [r"\breact(?:\.js)?\b"],
        "nodejs": [r"\bnodejs\b", r"\bnode\.js\b"],
        "django": [r"\bdjango\b"],
        "flask": [r"\bflask\b"],
        "spring boot": [r"\bspring boot\b"],
        "c++": [r"\bc\+\+\b", r"\bcplusplus\b"],
        "c#": [r"\bc#\b", r"\bcsharp\b"],
        "golang": [r"\bgolang\b", r"\bgo language\b"],
        "testing": [
            r"\bsoftware testing\b",
            r"\btest automation\b",
            r"\btesting\b",
        ],
        "database management": [
            r"\bdatabase management\b",
            r"\bdatabase design\b",
            r"\bdbms\b",
            r"\bmysql\b",
            r"\bpostgresql\b",
            r"\bsql server\b",
        ],
        "data science": [r"\bdata science\b"],
        "power bi": [r"\bpower bi\b"],
        "excel": [r"\bexcel\b", r"\bmicrosoft excel\b"],
        "communication": [r"\bcommunication skills\b"],
        "git": [r"\bgit\b", r"\bgithub\b"],
        "docker": [r"\bdocker\b"],
        "kubernetes": [r"\bkubernetes\b"],
        "cybersecurity": [
            r"\bcybersecurity\b",
            r"\binformation security\b",
        ],
    }

    def find_skill_matches(text, skills):
        text = normalized_text(text)
        matched = []

        for skill in skills:
            skill_name = normalized_text(skill)
            if not skill_name:
                continue

            patterns = SKILL_ALIASES.get(skill_name)

            if patterns is None:
                # Never use substring matching for a skill.
                patterns = [
                    r"(?<!\w)" + re.escape(skill_name) + r"(?!\w)"
                ]

            if any(re.search(pattern, text) for pattern in patterns):
                matched.append(skill)

        return matched

    def detected_technologies(text):
        found = []

        for skill_name in SKILL_ALIASES:
            patterns = SKILL_ALIASES[skill_name]
            if any(
                re.search(pattern, normalized_text(text))
                for pattern in patterns
            ):
                found.append(skill_name)

        return found

    # ------------------------------------------------------------
    # 3. EXPERIENCE CLASSIFICATION
    # ------------------------------------------------------------

    def classify_experience(text):
        text = normalized_text(text)

        fresher_patterns = [
            r"\bfresher\b",
            r"\bfreshers\b",
            r"\bentry.level\b",
            r"\bgraduate trainee\b",
            r"\brecent graduate\b",
            r"\bno experience required\b",
            r"\b0\s*(?:to|-)\s*1\s*years?\b",
            r"\b0\s*years?\s+of\s+experience\b",
        ]

        if any(re.search(p, text) for p in fresher_patterns):
            return {
                "category": "Fresher-friendly",
                "requirement": "Fresher or entry-level wording found",
                "min_years": 0,
            }

        senior_patterns = [
            r"\bsenior software engineer\b",
            r"\bstaff engineer\b",
            r"\bprincipal engineer\b",
            r"\bengineering manager\b",
            r"\bsoftware development manager\b",
            r"\bteam lead\b",
            r"\btechnical lead\b",
            r"\btech lead\b",
            r"\barchitect\b",
            r"\b\d+\s*\+\s*years?\b",
        ]

        if any(re.search(p, text) for p in senior_patterns):
            # A specific minimum is still extracted below if available.
            category = "Review"
        else:
            category = "Unknown"

        range_match = re.search(
            r"\b(\d+(?:\.\d+)?)\s*(?:-|to)\s*"
            r"(\d+(?:\.\d+)?)\s*\+?\s*years?\b",
            text
        )

        if range_match:
            minimum = float(range_match.group(1))
            maximum = float(range_match.group(2))
            return {
                "category": "Fresher-friendly" if minimum <= 1 else "Review",
                "requirement": (
                    f"Requires approximately {minimum:g}–{maximum:g} years"
                ),
                "min_years": minimum,
            }

        plus_match = re.search(
            r"\b(\d+(?:\.\d+)?)\s*\+\s*years?\b", text
        )
        if plus_match:
            minimum = float(plus_match.group(1))
            return {
                "category": "Fresher-friendly" if minimum <= 1 else "Review",
                "requirement": f"Requires {minimum:g}+ years",
                "min_years": minimum,
            }

        single_match = re.search(
            r"\b(?:minimum|min\.?|at least|over|more than)\s*"
            r"(\d+(?:\.\d+)?)\s*years?\b",
            text
        )
        if single_match:
            minimum = float(single_match.group(1))
            return {
                "category": "Fresher-friendly" if minimum <= 1 else "Review",
                "requirement": f"Requires at least {minimum:g} years",
                "min_years": minimum,
            }

        if category == "Review":
            return {
                "category": "Review",
                "requirement": "Possible senior-level role; verify requirements",
                "min_years": None,
            }

        return {
            "category": "Unknown",
            "requirement": "Experience requirement not verified",
            "min_years": None,
        }

    # ------------------------------------------------------------
    # 4. LOAD PROFILE SKILLS
    # ------------------------------------------------------------

    try:
        profile = get_student_profile() or {}
    except Exception:
        profile = {}

    if not isinstance(profile, dict):
        profile = {}

    saved_skills = get_first(
        profile,
        ["skills", "technical_skills", "student_skills"],
        []
    )

    if isinstance(saved_skills, str):
        # Support comma-separated skills from the profile.
        saved_skills = [
            item.strip()
            for item in re.split(r"[,;\n]", saved_skills)
            if item.strip()
        ]
    elif not isinstance(saved_skills, list):
        saved_skills = []

    saved_skills = list(dict.fromkeys(
        clean_text(skill)
        for skill in saved_skills
        if clean_text(skill)
    ))

    experience_value = get_first(
        profile,
        ["experience_years", "work_experience", "experience"],
        0
    )

    try:
        saved_experience = float(
            re.search(r"\d+(?:\.\d+)?", clean_text(experience_value)).group()
        )
    except (AttributeError, ValueError, TypeError):
        saved_experience = 0.0

    st.write("**Your saved skills:**", ", ".join(saved_skills) or "No skills saved")
    st.write(
        "**Full-time work experience saved in your profile:**",
        f"{saved_experience:g} years"
    )

    # ------------------------------------------------------------
    # 5. SEARCH FORM
    # ------------------------------------------------------------

    with st.form("live_job_search_form"):
        query = st.text_input(
            "Job title or keywords",
            value="software developer"
        )
        location = st.text_input("Location", value="Bengaluru")
        max_results = st.selectbox(
            "Maximum listings to retrieve per provider",
            options=[5, 10, 20],
            index=0
        )
        fresher_priority = st.checkbox(
            "Prioritize fresher and entry-level jobs",
            value=True
        )
        include_old = st.checkbox(
            "Show listings older than 90 days in a separate section",
            value=False
        )

        search_clicked = st.form_submit_button(
            "Search Jobs",
            type="primary"
        )

    # ------------------------------------------------------------
    # 6. FETCH AND NORMALIZE PROVIDER RESULTS
    # ------------------------------------------------------------

    if search_clicked:
        if not query.strip() or not location.strip():
            st.warning("Enter both a job title and location.")
        else:
            all_jobs = []
            adzuna_total = 0
            jooble_total = 0
            errors = []

            # Adzuna
            app_id = st.secrets.get("ADZUNA_APP_ID", "")
            app_key = st.secrets.get("ADZUNA_APP_KEY", "")

            if app_id and app_key:
                try:
                    response = requests.get(
                        "https://api.adzuna.com/v1/api/jobs/in/search/1",
                        params={
                            "app_id": app_id,
                            "app_key": app_key,
                            "what": query.strip(),
                            "where": location.strip(),
                            "results_per_page": max_results,
                            "content-type": "application/json",
                        },
                        timeout=20,
                    )
                    response.raise_for_status()
                    data = response.json()
                    adzuna_total = data.get("count", 0)

                    for job in data.get("results", []):
                        all_jobs.append(normalize_adzuna_job(job))

                except Exception as exc:
                    errors.append(f"Adzuna: {exc}")
            else:
                errors.append(
                    "Adzuna credentials are missing from Streamlit Secrets."
                )

            # Jooble
            try:
                jooble_response = search_jooble(
                    query.strip(), location.strip(), max_results
                )

                if isinstance(jooble_response, dict):
                    jooble_total = jooble_response.get(
                        "totalCount",
                        jooble_response.get(
                            "total", jooble_response.get("count", 0)
                        )
                    )

                    jooble_jobs = jooble_response.get(
                        "jobs",
                        jooble_response.get(
                            "data", jooble_response.get("results", [])
                        )
                    )

                    if not isinstance(jooble_jobs, list):
                        jooble_jobs = []

                    for job in jooble_jobs:
                        if isinstance(job, dict):
                            all_jobs.append(normalize_jooble_job(job))

                elif isinstance(jooble_response, list):
                    for job in jooble_response:
                        if isinstance(job, dict):
                            all_jobs.append(normalize_jooble_job(job))

            except Exception as exc:
                errors.append(f"Jooble: {exc}")

            # Deduplicate across providers.
            unique_jobs = []
            seen_urls = set()
            seen_fingerprints = set()

            for job in all_jobs:
                job["url"] = normalize_url(job.get("url", ""))

                title_key = normalized_text(job.get("title", ""))
                company_key = normalized_text(job.get("company", ""))
                location_key = normalized_text(job.get("location", ""))

                fingerprint = (
                    title_key,
                    company_key,
                    location_key,
                )

                if job["url"] and job["url"] in seen_urls:
                    continue

                if fingerprint in seen_fingerprints:
                    continue

                if job["url"]:
                    seen_urls.add(job["url"])

                seen_fingerprints.add(fingerprint)
                unique_jobs.append(job)

            st.session_state["live_job_results"] = unique_jobs
            st.session_state["live_job_search_info"] = {
                "query": query.strip(),
                "location": location.strip(),
                "adzuna_total": adzuna_total,
                "jooble_total": jooble_total,
                "errors": errors,
                "fresher_priority": fresher_priority,
                "include_old": include_old,
                "max_results": max_results,
            }

    # ------------------------------------------------------------
    # 7. DISPLAY RESULTS AND CONSISTENT COUNTS
    # ------------------------------------------------------------

    if "live_job_results" in st.session_state:
        jobs = st.session_state["live_job_results"]
        info = st.session_state.get("live_job_search_info", {})

        # Use the settings from the latest search.
        fresher_priority = info.get("fresher_priority", True)
        include_old = info.get("include_old", False)

        st.subheader("Search Results")
        st.write(
            f"Search: **{info.get('query', '')}** in "
            f"**{info.get('location', '')}**"
        )

        for error in info.get("errors", []):
            st.warning(error)

        st.write("### Provider totals")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Adzuna reported total",
            f"{info.get('adzuna_total', 0):,}"
        )
        col2.metric(
            "Jooble reported total",
            f"{info.get('jooble_total', 0):,}"
        )
        col3.metric("Unique listings retrieved", len(jobs))

        now = datetime.now(timezone.utc)
        recent_jobs = []
        undated_jobs = []
        old_jobs = []
        experience_filtered = []

        # Classify date and experience independently.
        for job in jobs:
            posted = parse_posted_date(job.get("posted"))
            job["parsed_posted"] = posted

            if posted is None:
                job["date_group"] = "undated"
                job["date_label"] = "Date not verified"
                undated_jobs.append(job)
            else:
                age_days = (now - posted).total_seconds() / 86400

                if age_days < 0:
                    job["date_group"] = "undated"
                    job["date_label"] = "Posting date needs verification"
                    undated_jobs.append(job)
                elif age_days > 90:
                    job["date_group"] = "old"
                    job["date_label"] = posted.strftime("%d %b %Y")
                    old_jobs.append(job)
                else:
                    job["date_group"] = "recent"
                    job["date_label"] = posted.strftime("%d %b %Y")
                    recent_jobs.append(job)

            text_for_analysis = (
                clean_text(job.get("title", ""))
                + " "
                + clean_text(job.get("description", ""))
            )

            exp = classify_experience(text_for_analysis)
            job["experience_category"] = exp["category"]
            job["experience_requirement"] = exp["requirement"]
            job["minimum_experience"] = exp["min_years"]

            job["matching_skills"] = find_skill_matches(
                text_for_analysis, saved_skills
            )

            job["missing_skills"] = [
                skill for skill in saved_skills
                if skill not in job["matching_skills"]
            ]

            job["technologies"] = detected_technologies(text_for_analysis)

            if saved_skills:
                job["skill_overlap"] = round(
                    100 * len(job["matching_skills"]) / len(saved_skills)
                )
            else:
                job["skill_overlap"] = 0

            # Experience filtering does not alter date classification.
            is_excluded = False

            if fresher_priority:
                if job["experience_category"] == "Review":
                    is_excluded = True
                elif (
                    job["minimum_experience"] is not None
                    and job["minimum_experience"] > 1
                ):
                    is_excluded = True

            job["experience_excluded"] = is_excluded

            if is_excluded:
                experience_filtered.append(job)

        # Count eligible listings by date group.
        def eligible(items):
            return [
                job for job in items
                if not job.get("experience_excluded", False)
            ]

        eligible_recent = eligible(recent_jobs)
        eligible_undated = eligible(undated_jobs)
        eligible_old = eligible(old_jobs)

        visible_old = eligible_old if include_old else []

        displayed_jobs = (
            eligible_recent
            + eligible_undated
            + visible_old
        )

        st.subheader("Result Summary")

        m1, m2, m3 = st.columns(3)
        m1.metric("Recent listings", len(recent_jobs))
        m2.metric("Undated listings", len(undated_jobs))
        m3.metric("Displayed listings", len(displayed_jobs))

        m4, m5, m6 = st.columns(3)
        m4.metric("Old listings found", len(old_jobs))
        m5.metric("Experience-filtered listings", len(experience_filtered))
        m6.metric(
            "Old listings hidden",
            len(eligible_old) if not include_old else 0
        )

        st.caption(
            "Date counts classify every unique listing independently of "
            "experience filtering. Experience-filtered listings may overlap "
            "with recent, undated, or old date groups."
        )

        old_experience_excluded = sum(
            1 for job in old_jobs if job.get("experience_excluded")
        )

        if old_experience_excluded:
            st.info(
                f"{old_experience_excluded} old listing(s) are also excluded "
                "by the experience filter. They are counted in the "
                "experience-filtered group, not the old-listings-hidden count."
            )

        if fresher_priority:
            st.info(
                f"Fresher prioritization is on. "
                f"{len(experience_filtered)} listing(s) are excluded from "
                "the main results because their experience classification "
                "suggests they may not suit entry-level applicants."
            )

        
        # --------------------------------------------------------
        # 8. JOB CARD DISPLAY
        # --------------------------------------------------------

        st.subheader("Available Job Listings")

        if not displayed_jobs:
            st.info(
                "No job listings are available with the current filters. "
                "Try disabling fresher prioritization, including older "
                "listings, or searching for another job title or location."
            )

        else:
            st.write(
                f"Displaying **{len(displayed_jobs)}** eligible job listing(s)."
            )

            for index, job in enumerate(displayed_jobs, start=1):

                title = (
                    clean_text(job.get("title", ""))
                    or "Job title not provided"
                )

                company_value = job.get("company", "")
                if isinstance(company_value, dict):
                    company_value = (
                        company_value.get("display_name")
                        or company_value.get("name")
                        or ""
                    )

                company = (
                    clean_text(company_value)
                    or "Company not specified"
                )

                location_value = job.get("location", "")
                if isinstance(location_value, dict):
                    location_value = (
                        location_value.get("display_name")
                        or location_value.get("name")
                        or ""
                    )

                location = (
                    clean_text(location_value)
                    or "Location not specified"
                )

                description = clean_text(
                    job.get("description", "")
                )

                job_url = (
                    job.get("url")
                    or job.get("redirect_url")
                    or job.get("link")
                    or job.get("job_url")
                    or ""
                )

                source = (
                    clean_text(job.get("source", ""))
                    or "Job provider not specified"
                )

                date_label = (
                    job.get("date_label")
                    or "Date not verified"
                )

                experience_category = (
                    job.get("experience_category")
                    or "Not classified"
                )

                experience_requirement = (
                    job.get("experience_requirement")
                    or "Check the original job listing"
                )

                matching_skills = job.get("matching_skills", [])
                missing_skills = job.get("missing_skills", [])
                technologies = job.get("technologies", [])
                skill_overlap = job.get("skill_overlap", 0)

                with st.container(border=True):

                    st.markdown(f"### {index}. {title}")

                    st.write(f"**Company:** {company}")
                    st.write(f"**Location:** {location}")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Skill overlap",
                        f"{skill_overlap}%"
                    )

                    col2.metric(
                        "Experience category",
                        experience_category
                    )

                    col3.metric(
                        "Posting date",
                        date_label
                    )

                    st.write(
                        f"**Experience requirement:** "
                        f"{experience_requirement}"
                    )

                    if matching_skills:
                        st.write(
                            "**Matching skills:** "
                            + ", ".join(map(str, matching_skills))
                        )
                    else:
                        st.write(
                            "**Matching skills:** No matches identified"
                        )

                    if missing_skills:
                        st.write(
                            "**Skills to strengthen:** "
                            + ", ".join(map(str, missing_skills))
                        )

                    if technologies:
                        st.write(
                            "**Technologies mentioned:** "
                            + ", ".join(map(str, technologies))
                        )

                    if description:
                        with st.expander("View job description"):
                            st.write(description)

                    st.caption(
                        f"Source: {source} | "
                        f"Listing date: {date_label}"
                    )

                    if job_url:
                        st.link_button(
                            "View Original Job / Apply",
                            str(job_url),
                            use_container_width=True
                        )
                    else:
                        st.caption(
                            "An application link was not provided by "
                            "the job provider."
                        )


# ============================================================
# CLOUD STORAGE
# ============================================================

elif page == "☁️ Cloud Storage":

    st.header("☁️ Cloud Storage")
    st.write("View your saved profile and prediction/analysis history.")

    tab_profile, tab_history = st.tabs(
        ["👤 Saved Profile", "📚 Prediction & Analysis History"]
    )

    with tab_profile:
        try:
            profile = get_student_profile()

            if profile:
                st.write("**Full name:**", profile.get("full_name", ""))
                st.write("**Course:**", profile.get("course", ""))
                st.write("**Graduation year:**", profile.get("graduation_year", ""))
                st.write("**Skills:**", profile.get("skills", ""))
                st.write(
                    f"**Experience level:** "
                    f"{profile.get('experience_level') or 'Not specified'}"
                )
                st.write(
                    f"**Work experience:** "
                    f"{profile.get('work_experience_years', 0)} years"
                )
                st.write(
                         f"**Internship experience:** "
                          f"{profile.get('internship_experience_months', 0)} months"
                        )

                st.write(
                        f"**Project experience:** "
                         f"{profile.get('project_experience') or 'Not specified'}"
                        )
            else:
                st.info("No profile saved yet.")
        except Exception as error:
            st.error("Could not load your saved profile.")
            st.caption(str(error))

    with tab_history:
        if st.button("🔄 Refresh Cloud History"):
            st.rerun()

        try:
            response = cloud_select("placementai_prediction_history")
            history = response.data or []

            if not history:
                st.info("No saved predictions or analyses yet.")

            for index, record in enumerate(history):
                action = str(record.get("action", "Saved record"))
                result = str(record.get("prediction", ""))
                created_at = str(record.get("created_at", ""))

                with st.expander(
                    f"{action.replace('_', ' ').title()} — {result} — {created_at}"
                ):
                    render_saved_details(record.get("input_data", {}))

        except Exception as error:
            st.error("Could not load cloud history.")
            st.caption(str(error))


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        PlacementAI · Student Placement & Career Support Platform<br>
        Machine learning · Career exploration · Guided learning roadmap
    </div>
    """,
    unsafe_allow_html=True,
)