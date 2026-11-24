from pathlib import Path
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
import re
import requests
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
    supabase = get_supabase_client()

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

    existing_profile = (
        supabase.table("placementai_student_profiles")
        .select("user_id")
        .eq("user_id", student_id)
        .limit(1)
        .execute()
    )

    if existing_profile.data:
        supabase.table("placementai_student_profiles").update(
            payload
        ).eq("user_id", student_id).execute()
    else:
        supabase.table("placementai_student_profiles").insert(
            payload
        ).execute()



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

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Placement Prediction</div>
                <div class="feature-description">
                    Estimate placement readiness from your profile.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">💼</div>
                <div class="feature-title">Career Recommendations</div>
                <div class="feature-description">
                    Explore job roles, required skills, and certifications.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🧩</div>
                <div class="feature-title">Skill Gap Roadmap</div>
                <div class="feature-description">
                    Discover missing skills and follow a guided study plan.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
# Adzuna API + Skill Matching + Skill Gaps
# + Related Technologies + Experience Analysis
# + Fresher / Entry-Level Filter
# ============================================================

elif page == "🌐 Live Job Vacancies":

    st.header("🌐 Live Job Vacancies")

    st.write(
        "Search live Adzuna listings, compare student skills, "
        "identify related technologies to explore, and estimate "
        "experience compatibility."
    )

    st.info(
        "Skill matching and experience comparisons are estimates. "
        "They do not guarantee eligibility, interview calls, or selection."
    )

    # --------------------------------------------------------
    # 1. LOAD STUDENT PROFILE
    # --------------------------------------------------------

    profile = {}

    try:
        profile = get_student_profile() or {}
    except Exception:
        st.warning(
            "Could not load your saved profile. "
            "You can still search for vacancies."
        )

    saved_skills = profile.get("skills", "")

    if isinstance(saved_skills, list):
        student_skills = list(dict.fromkeys(
            str(skill).strip()
            for skill in saved_skills
            if str(skill).strip()
        ))
    else:
        student_skills = list(dict.fromkeys(
            skill.strip()
            for skill in str(saved_skills or "").split(",")
            if skill.strip()
        ))

    try:
        saved_work_years = max(
            0.0,
            min(
                50.0,
                float(profile.get("work_experience_years") or 0)
            )
        )
    except (TypeError, ValueError):
        saved_work_years = 0.0

    if student_skills:
        st.write(
            "**Your saved skills:** " + ", ".join(student_skills)
        )
    else:
        st.warning(
            "No saved skills found. Add skills in Student Profile "
            "for personalized matching."
        )

    st.caption(
        f"Full-time work experience saved in your profile: "
        f"{saved_work_years:g} years"
    )

    # --------------------------------------------------------
    # 2. SEARCH FORM
    # --------------------------------------------------------

    with st.form("live_job_search_form"):

        search_title = st.text_input(
            "Job title or keywords",
            value="software developer",
            placeholder="Python developer, data analyst, AI engineer",
        )

        search_location = st.text_input(
            "Location",
            value="Bengaluru",
            placeholder="Mangalore, Mysuru, Chennai, Hyderabad",
        )

        search_experience = st.number_input(
            "Your full-time work experience (years)",
            min_value=0.0,
            max_value=50.0,
            value=float(saved_work_years),
            step=0.5,
            help=(
                "Used for this search only. "
                "It does not update your saved profile."
            ),
        )

        results_limit = st.selectbox(
            "Number of vacancies to retrieve",
            [5, 10, 20],
            index=1,
        )

        fresher_only = st.checkbox(
            "Show only likely fresher/entry-level jobs",
            value=True,
            help=(
                "Shows listings with evidence of entry-level suitability. "
                "Excludes senior/management titles, unclear experience "
                "levels, and requirements exceeding one year."
            ),
        )

        search_clicked = st.form_submit_button(
            "🔍 Search Live Vacancies",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # 3. FETCH LIVE ADZUNA RESULTS
    # --------------------------------------------------------

    if search_clicked:

        st.session_state.pop("live_job_results", None)
        st.session_state.pop("live_job_query", None)

        if not search_title.strip() or not search_location.strip():
            st.error("Enter both a job title and a location.")

        else:
            try:
                app_id = st.secrets["ADZUNA_APP_ID"]
                app_key = st.secrets["ADZUNA_APP_KEY"]

                api_url = (
                    "https://api.adzuna.com/v1/api/jobs/in/search/1"
                )

                params = {
                    "app_id": app_id,
                    "app_key": app_key,
                    "what": search_title.strip(),
                    "where": search_location.strip(),
                    "results_per_page": int(results_limit),
                    "content-type": "application/json",
                }

                with st.spinner("Fetching live vacancies from Adzuna..."):
                    response = requests.get(
                        api_url,
                        params=params,
                        timeout=25,
                    )

                if response.status_code == 200:

                    data = response.json()

                    st.session_state["live_job_results"] = (
                        data.get("results", [])
                    )

                    st.session_state["live_job_query"] = {
                        "title": search_title.strip(),
                        "location": search_location.strip(),
                        "total": data.get("count", 0),
                        "experience_years": float(search_experience),
                        "fresher_only": bool(fresher_only),
                        "results_limit": int(results_limit),
                    }

                else:
                    st.error(
                        f"Adzuna returned HTTP {response.status_code}. "
                        "Check your API access and try again."
                    )

            except KeyError:
                st.error(
                    "Adzuna credentials are missing. Check "
                    "ADZUNA_APP_ID and ADZUNA_APP_KEY in "
                    ".streamlit/secrets.toml."
                )

            except requests.RequestException:
                st.error(
                    "Could not connect to Adzuna. Check your internet "
                    "connection and try again."
                )

            except ValueError:
                st.error(
                    "Could not process the API response. "
                    "Please try the search again."
                )

            except Exception as exc:
                st.error(
                    "An unexpected error occurred while fetching "
                    "vacancies. Check your configuration."
                )
                st.caption(f"Error details: {exc}")

    # --------------------------------------------------------
    # 4. SKILL CATALOGUE AND ALIASES
    # --------------------------------------------------------

    # Each canonical skill has common alternative spellings.
    # A related technology is NOT automatically treated as a
    # skill the student already knows.

    skill_aliases = {
        "python": [
            "python",
            "python programming",
            "python developer",
        ],
        "cloud computing": [
            "cloud computing",
            "cloud services",
            "cloud infrastructure",
            "cloud platform",
            "cloud technology",
        ],
        "data analytics": [
            "data analytics",
            "data analysis",
            "data analyst",
            "business analytics",
        ],
        "data structures": [
            "data structures",
            "data structure",
            "data structures and algorithms",
            "data structure and algorithms",
            "dsa",
        ],
        "java": [
            "java",
            "core java",
            "java programming",
        ],
        "machine learning": [
            "machine learning",
            "machine learning models",
        ],
        "sql": [
            "sql",
            "sql queries",
            "structured query language",
        ],
        "web development": [
            "web development",
            "web developer",
            "frontend development",
            "front-end development",
            "backend development",
            "back-end development",
            "full stack development",
            "full-stack development",
        ],
        "aws": [
            "aws",
            "amazon web services",
        ],
        "microsoft azure": [
            "microsoft azure",
            "azure",
        ],
        "google cloud": [
            "google cloud",
            "google cloud platform",
            "gcp",
        ],
        "mysql": ["mysql"],
        "postgresql": ["postgresql", "postgres"],
        "pandas": ["pandas"],
        "numpy": ["numpy"],
        "power bi": ["power bi"],
        "tableau": ["tableau"],
        "excel": ["microsoft excel", "excel"],
        "deep learning": ["deep learning"],
        "natural language processing": [
            "natural language processing",
            "nlp",
        ],
        "html": ["html5", "html"],
        "css": ["css3", "css"],
        "javascript": ["javascript", "java script"],
        "react": ["react.js", "reactjs", "react"],
        "node.js": ["node.js", "nodejs"],
        "django": ["django"],
        "flask": ["flask"],
        "rest apis": [
            "restful apis",
            "restful api",
            "rest apis",
            "rest api",
        ],
        "git": ["git", "git version control"],
        "linux": ["linux"],
        "docker": ["docker"],
        "kubernetes": ["kubernetes"],
        "cybersecurity": ["cybersecurity", "cyber security"],
        "computer networks": [
            "computer networks",
            "computer networking",
        ],
        "statistics": ["statistics", "statistical analysis"],
        "data visualization": ["data visualization"],
        "communication": [
            "communication skills",
            "verbal communication",
        ],
        "agile": ["agile methodology", "agile"],
        "c": ["c programming", "c language"],
        "c++": ["c++", "cpp"],
        "algorithms": ["algorithms", "algorithm design"],
    }

    # These are related areas worth exploring, not proof of existing
    # knowledge of each specific tool or technology.

    related_skill_groups = {
        "cloud computing": [
            "aws",
            "microsoft azure",
            "google cloud",
        ],
        "data analytics": [
            "pandas",
            "numpy",
            "power bi",
            "tableau",
            "excel",
            "data visualization",
            "sql",
        ],
        "data structures": [
            "algorithms",
        ],
        "web development": [
            "html",
            "css",
            "javascript",
            "react",
            "node.js",
            "django",
            "flask",
            "rest apis",
        ],
        "machine learning": [
            "deep learning",
            "natural language processing",
        ],
        "sql": [
            "mysql",
            "postgresql",
        ],
    }

    # --------------------------------------------------------
    # 5. TEXT NORMALIZATION AND SKILL MATCHING
    # --------------------------------------------------------

    def normalize_skill_text(value):
        """Normalize text for consistent keyword matching."""

        value = str(value or "").casefold()

        # Remove HTML tags that may be present in descriptions.
        value = re.sub(r"<[^>]*>", " ", value)

        # Standardize common dash characters.
        value = value.replace("–", "-").replace("—", "-")

        # Preserve C++, C#, and common technology punctuation.
        value = re.sub(r"[^a-z0-9+#.]+", " ", value)

        return " ".join(value.split())

    def phrase_is_present(phrase, text):
        """Match a complete phrase instead of part of a larger word."""

        phrase = normalize_skill_text(phrase)
        text = normalize_skill_text(text)

        if not phrase or not text:
            return False

        phrase_pattern = r"\s+".join(
            re.escape(part) for part in phrase.split()
        )

        return bool(
            re.search(
                r"(?<!\w)" + phrase_pattern + r"(?!\w)",
                text,
            )
        )

    def canonical_skill_name(skill):
        """Convert known skill aliases to canonical catalogue names."""

        normalized = normalize_skill_text(skill)

        for canonical, aliases in skill_aliases.items():

            if normalized == normalize_skill_text(canonical):
                return canonical

            for alias in aliases:
                if normalized == normalize_skill_text(alias):
                    return canonical

        return normalized

    def skill_is_in_text(skill, text):
        """Check a skill against the listing and its known aliases."""

        canonical = canonical_skill_name(skill)
        aliases = skill_aliases.get(canonical, [skill])

        candidates = list(dict.fromkeys([canonical] + aliases))

        return any(
            phrase_is_present(alias, text)
            for alias in candidates
        )

    def get_profile_skill_names(skills):
        """Return the canonical names of saved profile skills."""

        return {
            canonical_skill_name(skill)
            for skill in skills
            if normalize_skill_text(skill)
        }

    def detect_skills_in_listing(text):
        """Find known catalogue skills in the available job text."""

        detected = set()

        for canonical, aliases in skill_aliases.items():

            candidates = list(dict.fromkeys([canonical] + aliases))

            if any(
                phrase_is_present(alias, text)
                for alias in candidates
            ):
                detected.add(canonical)

        return detected

    def analyze_job_skills(job_text, profile_skills):
        """
        Compare saved skills with the job listing.

        The score is a keyword-match percentage, not a hiring
        probability or a guarantee of eligibility.
        """

        matched = []
        unmatched = []

        for skill in profile_skills:

            if skill_is_in_text(skill, job_text):
                matched.append(skill)
            else:
                unmatched.append(skill)

        if profile_skills:
            score = len(matched) / len(profile_skills) * 100
        else:
            score = 0.0

        detected = detect_skills_in_listing(job_text)
        profile_names = get_profile_skill_names(profile_skills)

        # Catalogue skills detected in the listing but not already
        # represented by a canonical saved profile skill.
        additional = detected - profile_names

        # Related technologies are shown separately. For example,
        # cloud computing can be related to AWS without proving
        # that the student already knows AWS.
        related = set()

        for profile_skill in profile_names:

            for related_skill in related_skill_groups.get(
                profile_skill, []
            ):

                if (
                    related_skill in detected
                    and related_skill not in profile_names
                ):
                    related.add(related_skill)

        additional = sorted(additional - related)

        return {
            "matched": matched,
            "unmatched": unmatched,
            "score": score,
            "additional": additional,
            "related": sorted(related),
        }

    # --------------------------------------------------------
    # 6. EXPERIENCE REQUIREMENT DETECTION
    # --------------------------------------------------------

    entry_level_terms = [
        "fresher",
        "freshers",
        "fresh graduate",
        "recent graduate",
        "graduate trainee",
        "entry level",
        "entry-level",
        "junior developer",
        "junior software",
        "intern",
        "internship",
        "trainee",
        "graduate program",
        "graduate programme",
        "campus hiring",
    ]

    senior_title_terms = [
        "senior",
        "staff engineer",
        "principal engineer",
        "lead developer",
        "tech lead",
        "team lead",
        "engineering manager",
        "development manager",
        "software development manager",
        "director",
        "vice president",
        "head of",
        "architect",
    ]

    def detect_experience_requirement(title, description):
        """
        Detect common experience formats.

        This is an estimate because job descriptions may mention
        multiple roles, different experience levels, or unrelated
        dates. Always verify the original vacancy.
        """

        text = normalize_skill_text(f"{title} {description}")

        # First check for explicit ranges, such as 2-3 years.
        range_matches = list(re.finditer(
            r"(?<!\w)(\d+(?:\.\d+)?)\s*"
            r"(?:-|to)\s*(\d+(?:\.\d+)?)\s*"
            r"(?:years?|yrs?)(?!\w)",
            text,
        ))

        if range_matches:

            # Prefer a range that appears near an experience keyword.
            selected_match = range_matches[0]

            for candidate in range_matches:
                surrounding = text[
                    max(0, candidate.start() - 65):
                    min(len(text), candidate.end() + 65)
                ]

                if any(term in surrounding for term in [
                    "experience",
                    "work experience",
                    "relevant experience",
                    "professional experience",
                    "years of experience",
                ]):
                    selected_match = candidate
                    break

            minimum = float(selected_match.group(1))
            maximum = float(selected_match.group(2))

            if maximum < minimum:
                minimum, maximum = maximum, minimum

            return {
                "requirement": f"{minimum:g}-{maximum:g} years",
                "minimum_years": minimum,
                "maximum_years": maximum,
                "detected": True,
                "broad_range": maximum > 1.0,
            }

        # Examples: 2+ years, 3+ yrs.
        plus_match = re.search(
            r"(?<!\w)(\d+(?:\.\d+)?)\s*\+\s*"
            r"(?:years?|yrs?)(?!\w)",
            text,
        )

        if plus_match:
            minimum = float(plus_match.group(1))

            return {
                "requirement": f"{minimum:g}+ years",
                "minimum_years": minimum,
                "maximum_years": None,
                "detected": True,
                "broad_range": False,
            }

        # Examples: 2 years of experience, 1 year relevant experience.
        experience_match = re.search(
            r"(?<!\w)(\d+(?:\.\d+)?)\s*"
            r"(?:years?|yrs?)\s*(?:of\s+)?"
            r"(?:(?:relevant|professional|full[- ]time|work)\s+)*"
            r"experience(?!\w)",
            text,
        )

        if experience_match:
            minimum = float(experience_match.group(1))

            return {
                "requirement": f"{minimum:g} years",
                "minimum_years": minimum,
                "maximum_years": None,
                "detected": True,
                "broad_range": False,
            }

        # Examples: minimum 2 years, at least 1 year.
        minimum_match = re.search(
            r"(?:minimum|min\.?|at least|requires?|requiring)"
            r"\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*"
            r"(?:years?|yrs?)(?!\w)",
            text,
        )

        if minimum_match:
            minimum = float(minimum_match.group(1))

            return {
                "requirement": f"{minimum:g}+ years",
                "minimum_years": minimum,
                "maximum_years": None,
                "detected": True,
                "broad_range": False,
            }

        # Entry-level language is evidence, not a guarantee.
        if any(
            phrase_is_present(term, text)
            for term in entry_level_terms
        ):
            return {
                "requirement": "Entry-level wording detected",
                "minimum_years": 0.0,
                "maximum_years": 1.0,
                "detected": True,
                "broad_range": False,
            }

        return {
            "requirement": "Not specified in listing text",
            "minimum_years": None,
            "maximum_years": None,
            "detected": False,
            "broad_range": False,
        }

    def classify_job_level(title, experience_info):
        """Classify the likely level and determine fresher eligibility."""

        title_text = normalize_skill_text(title)

        is_senior = any(
            phrase_is_present(term, title_text)
            for term in senior_title_terms
        )

        is_entry_title = any(
            phrase_is_present(term, title_text)
            for term in entry_level_terms
        )

        minimum = experience_info["minimum_years"]
        maximum = experience_info["maximum_years"]

        # Senior/management titles are excluded from fresher results.
        if is_senior:
            return {
                "label": "Likely senior/management role",
                "rank": 3,
                "fresher_eligible": False,
                "reason": "Senior or management wording in title",
            }

        # A stated minimum greater than one year is not treated
        # as a likely fresher role.
        if (
            experience_info["detected"]
            and minimum is not None
            and minimum > 1
        ):
            return {
                "label": "Likely experienced role",
                "rank": 2,
                "fresher_eligible": False,
                "reason": "Minimum experience exceeds one year",
            }

        # A clearly stated range within 0-1 years is a candidate.
        if (
            experience_info["detected"]
            and maximum is not None
            and maximum <= 1
        ):
            return {
                "label": "Potential fresher/entry-level role",
                "rank": 0,
                "fresher_eligible": True,
                "reason": "Stated experience range is within 0-1 years",
            }

        # A clearly entry-level title with no detected experience
        # requirement can be included for manual verification.
        if is_entry_title and not experience_info["detected"]:
            return {
                "label": "Potential fresher/entry-level role",
                "rank": 0,
                "fresher_eligible": True,
                "reason": "Entry-level wording; verify description",
            }

        # A broad range such as 0-5 years is not proof that a
        # vacancy is suitable for freshers.
        if (
            experience_info["detected"]
            and minimum is not None
            and maximum is not None
            and minimum <= 1
            and maximum > 1
        ):
            return {
                "label": "Broad experience range - verify",
                "rank": 1,
                "fresher_eligible": False,
                "reason": "Range includes experience beyond one year",
            }

        return {
            "label": "Experience level not confirmed",
            "rank": 1,
            "fresher_eligible": False,
            "reason": "Insufficient evidence of fresher eligibility",
        }

    # --------------------------------------------------------
    # 7. LOAD THE LATEST SEARCH RESULTS
    # --------------------------------------------------------

    live_jobs = st.session_state.get("live_job_results", [])
    query_info = st.session_state.get("live_job_query", {})

    if query_info:

        st.subheader("Search Results")

        st.write(
            f"Search: **{query_info['title']}** "
            f"in **{query_info['location']}**"
        )

        st.caption(
            f"Total results reported by Adzuna: "
            f"{query_info['total']}"
        )

        st.caption(
            f"Listings retrieved from Adzuna: {len(live_jobs)}"
        )

        student_work_years = float(
            query_info.get(
                "experience_years",
                saved_work_years,
            )
        )

        # ----------------------------------------------------
        # 8. ANALYZE EACH LISTING
        # ----------------------------------------------------

        prepared_jobs = []

        for job in live_jobs:

            title = job.get("title") or "Job title not provided"
            description = job.get("description") or ""

            company_data = job.get("company") or {}
            company = company_data.get(
                "display_name",
                "Company not listed",
            )

            location_data = job.get("location") or {}
            location = location_data.get(
                "display_name",
                "Location not listed",
            )

            # Analyze only text returned by Adzuna.
            job_text = f"{title} {description}"

            skill_analysis = analyze_job_skills(
                job_text,
                student_skills,
            )

            experience_info = detect_experience_requirement(
                title,
                description,
            )

            level_info = classify_job_level(
                title,
                experience_info,
            )

            minimum = experience_info["minimum_years"]
            maximum = experience_info["maximum_years"]

            # Compare the user's entered years with the detected
            # minimum/maximum, where available.
            if not experience_info["detected"]:

                experience_comparison = (
                    "Experience requirement unknown"
                )

                experience_category = "unknown"

            elif (
                maximum is not None
                and student_work_years > maximum
            ):

                experience_comparison = (
                    "Your experience is above the stated range; verify"
                )

                experience_category = "above_range"

            elif (
                minimum is not None
                and student_work_years >= minimum
            ):

                experience_comparison = (
                    "Stated minimum experience appears to match"
                )

                experience_category = "match"

            else:

                experience_comparison = (
                    "May need more work experience"
                )

                experience_category = "more"

            prepared_jobs.append({
                "job": job,
                "title": title,
                "company": company,
                "location": location,
                "description": description,
                "skill_analysis": skill_analysis,
                "experience_info": experience_info,
                "experience_comparison": experience_comparison,
                "experience_category": experience_category,
                "level_info": level_info,
            })

        # ----------------------------------------------------
        # 9. APPLY FRESHER FILTER
        # ----------------------------------------------------

        filter_is_active = query_info.get("fresher_only", True)

        if filter_is_active:

            displayed_jobs = [
                item for item in prepared_jobs
                if (
            item["level_info"]["fresher_eligible"]
            or (
                item["experience_info"]["detected"]
                and item["experience_info"]["minimum_years"] is not None
                and item["experience_info"]["minimum_years"] <= 2
                and item["level_info"]["label"]
                not in [
                    "Likely senior/management role",
                    "Likely experienced role",
                ]
            )
        )
            ]

        else:
            displayed_jobs = prepared_jobs

        # Show stronger keyword matches first within the eligible
        # set. This is a display order, not a selection prediction.
        displayed_jobs.sort(
            key=lambda item: (
                item["level_info"]["rank"],
                -item["skill_analysis"]["score"],
            )
        )

        st.write(
            "**Fresher-only filter:** "
            + ("ON" if filter_is_active else "OFF")
        )

        st.caption(
            f"{len(prepared_jobs) - len(displayed_jobs)} "
            "retrieved listings excluded by the active filter."
        )

        if filter_is_active:
            st.caption(
                "The filter excludes senior/management titles, "
                "unknown-level listings, and roles whose detected "
                "experience range exceeds one year. It may miss "
                "suitable jobs when descriptions are incomplete."
            )

        # ----------------------------------------------------
        # 10. EXPERIENCE SUMMARY
        # ----------------------------------------------------

        st.subheader("Experience-Based Vacancy Summary")

        st.write(
            f"**Your experience used for comparison:** "
            f"{student_work_years:g} years"
        )

        st.caption(
            "Counts below refer to the listings shown after filtering."
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Listings Retrieved",
            len(prepared_jobs),
        )

        col2.metric(
            "Listings Shown",
            len(displayed_jobs),
        )

        col3, col4 = st.columns(2)

        col3.metric(
            "Minimum Experience Appears to Match",
            sum(
                item["experience_category"] == "match"
                for item in displayed_jobs
            ),
        )

        col4.metric(
            "May Need More Experience",
            sum(
                item["experience_category"] == "more"
                for item in displayed_jobs
            ),
        )

        # ----------------------------------------------------
        # 11. EMPTY RESULTS HANDLING
        # ----------------------------------------------------

        if not displayed_jobs and filter_is_active:

            st.warning(
                "No retrieved listings passed the strict fresher "
                "filter. Try a more specific entry-level job title, "
                "retrieve more vacancies, or turn off the fresher "
                "filter to inspect all retrieved listings."
            )

        elif not displayed_jobs:

            st.warning(
                "No listings were returned. Try broader job keywords "
                "or check the location spelling."
            )

        # ----------------------------------------------------
        # 12. DISPLAY JOB CARDS
        # ----------------------------------------------------

        for item in displayed_jobs:

            job = item["job"]
            title = item["title"]
            company = item["company"]
            location = item["location"]
            description = item["description"]

            analysis = item["skill_analysis"]

            matched_skills = analysis["matched"]
            unmatched_skills = analysis["unmatched"]
            additional_skills = analysis["additional"]
            related_skills = analysis["related"]
            score = analysis["score"]

            experience_info = item["experience_info"]
            level_info = item["level_info"]

            job_url = job.get("redirect_url") or ""

            with st.container(border=True):

                st.markdown(f"### {title}")

                st.write(f"**Company:** {company}")
                st.write(f"**Location:** {location}")

                st.write(
                    "**Role classification:** "
                    + level_info["label"]
                )

                st.write(
                    "**Experience requirement:** "
                    + experience_info["requirement"]
                )

                st.write(
                    "**Experience comparison:** "
                    + item["experience_comparison"]
                )

                # --------------------------------------------
                # A. STUDENT SKILL MATCH SCORE
                # --------------------------------------------

                if student_skills:

                    st.progress(
                        min(max(score / 100.0, 0.0), 1.0)
                    )

                    st.write(
                        f"**Student skill keyword-match score:** "
                        f"{score:.1f}%"
                    )

                    st.caption(
                        "This score measures exact or alias-based "
                        "keyword matches between your saved skills "
                        "and the job title and description returned "
                        "by Adzuna. Missing keywords do not prove that "
                        "you lack a skill or are unqualified. This is "
                        "not a probability of getting selected."
                    )

                    if matched_skills:

                        st.success(
                            "**Student skills found in listing:** "
                            + ", ".join(matched_skills)
                        )

                    else:

                        st.write(
                            "**Student skills found in listing:** "
                            "None detected"
                        )

                    if unmatched_skills:

                        st.write(
                            "**Student skills not found in listing:** "
                            + ", ".join(unmatched_skills)
                        )

                    else:

                        st.write(
                            "**Student skills not found in listing:** "
                            "None"
                        )

                else:

                    st.info(
                        "Add skills to your student profile to "
                        "calculate a personalized keyword-match score."
                    )

                # --------------------------------------------
                # B. POTENTIAL SKILLS TO LEARN
                # --------------------------------------------

                st.markdown("**Potential skills to learn**")

                if additional_skills:

                    st.write(", ".join(additional_skills))

                    st.caption(
                        "These catalogue skills were detected in "
                        "the listing but are not represented by a "
                        "matching skill in your saved profile. They "
                        "are possible learning topics, not confirmed "
                        "mandatory requirements."
                    )

                else:

                    st.write(
                        "No additional catalogued skills were detected."
                    )

                # --------------------------------------------
                # C. RELATED TECHNOLOGIES
                # --------------------------------------------

                if related_skills:

                    st.markdown(
                        "**Related technologies to explore**"
                    )

                    st.write(", ".join(related_skills))

                    st.caption(
                        "These technologies are related to a broader "
                        "skill in your profile. For example, cloud "
                        "computing is related to AWS, but that does "
                        "not prove that you already know AWS."
                    )

                # --------------------------------------------
                # D. ORIGINAL JOB DESCRIPTION
                # --------------------------------------------

                with st.expander("View job description"):

                    st.write(
                        description
                        or "No description was provided by the API."
                    )

                # --------------------------------------------
                # E. ORIGINAL ADZUNA VACANCY
                # --------------------------------------------

                if job_url:

                    st.link_button(
                        "View vacancy on Adzuna",
                        job_url,
                        use_container_width=True,
                    )

                st.caption(
                    "Check the original listing for mandatory skills, "
                    "education, graduation year, experience, and "
                    "application eligibility before applying."
                )

    # --------------------------------------------------------
    # 13. INITIAL EMPTY STATE
    # --------------------------------------------------------

    else:

        st.caption(
            "Enter a job title and location, set your work "
            "experience, and select 'Search Live Vacancies' "
            "to retrieve listings."
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