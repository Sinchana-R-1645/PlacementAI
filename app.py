
from pathlib import Path
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


def save_student_profile(full_name, course, graduation_year, skills):
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


def normalize_skills(text):
    return {
        skill.strip().casefold()
        for skill in text.split(",")
        if skill.strip()
    }


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

# Apply pending navigation BEFORE creating the radio widget.
if "pending_page" in st.session_state:
    pending_page = st.session_state.pop("pending_page")

    if pending_page in PAGES:
        st.session_state["dashboard_page"] = pending_page

if st.session_state.get("dashboard_page") not in PAGES:
    st.session_state["dashboard_page"] = PAGES[0]

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 24px 0;">
            <div style="font-size: 34px;">🎓</div>
            <div style="font-size: 25px; font-weight: 800;">
                PlacementAI
            </div>
            <div style="color: #9cafd3; font-size: 12px;">
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

    st.markdown(
        """
        <div style="padding: 12px 0; color: #8496ba; font-size: 12px;">
            Private student workspace<br>
            Powered by Streamlit and Supabase
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚪 Log out", use_container_width=True):
        try:
            supabase.auth.sign_out()
        except Exception:
            pass

        for key in (
            "student_id",
            "student_email",
            "access_token",
            "refresh_token",
            "dashboard_page",
            "pending_page",
        ):
            st.session_state.pop(key, None)

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
            identify skill gaps, and keep your student progress
            securely stored in the cloud.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD HOME
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        f"""
        <div class="section-heading">
            Welcome back{", " + student_email if student_email else ""}!
        </div>
        <p style="color:#aebcda;">
            Your academic progress and career preparation, all in one place.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Read profile and history to populate dashboard metrics.
    saved_profile = {}
    saved_history = []

    try:
        saved_profile = get_student_profile()
    except Exception:
        pass

    try:
        response = cloud_select("placementai_prediction_history")
        saved_history = response.data or []
    except Exception:
        pass

    latest_result = (
        saved_history[0].get("prediction", "Not started")
        if saved_history
        else "Not started"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">👤 PROFILE</div>
                <div class="metric-value">
                    {"Complete" if saved_profile.get("full_name") else "Get started"}
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
                <div class="metric-value">{len(saved_history)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">🎯 LATEST RESULT</div>
                <div class="metric-value" style="font-size:20px;">
                    {latest_result[:35]}
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
            "Estimate your placement status using your academic details.",
            "📊 Placement Prediction",
        ),
        (
            "💼",
            "Job Recommendations",
            "Explore IT roles, their responsibilities, and required skills.",
            "💼 Job Role Recommendations",
        ),
        (
            "🧩",
            "Skill Gap Analysis",
            "Compare your current skills with the skills a role requires.",
            "🧩 Skill Gap Analysis",
        ),
        (
            "👤",
            "Student Profile",
            "Save your academic profile and current skills.",
            "👤 Student Profile",
        ),
        (
            "☁️",
            "Cloud Storage",
            "View your saved profile and prediction history.",
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
                        <div class="feature-title">{title}</div>
                        <div class="feature-description">
                            {description}
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
            💡 <b>Tip:</b> Keep your profile updated and use skill-gap
            analysis to plan what to learn next. A prediction is an estimate,
            not a guarantee of actual placement.
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
        "Maintain your academic information and current skills. "
        "Your profile is stored in your Supabase account."
    )

    try:
        profile = get_student_profile()
    except Exception as error:
        profile = {}
        st.error("Could not load the cloud profile.")
        st.caption(str(error))

    with st.form("student_profile_form"):

        full_name = st.text_input(
            "Full name",
            value=profile.get("full_name", ""),
        )

        course = st.text_input(
            "Course / Degree",
            value=profile.get("course", ""),
            placeholder="Example: B.E. Computer Science",
        )

        graduation_year = st.number_input(
            "Graduation year",
            min_value=2000,
            max_value=2100,
            value=int(profile.get("graduation_year") or 2027),
            step=1,
        )

        skills = st.text_area(
            "Current skills",
            value=profile.get("skills", ""),
            placeholder="Python, SQL, HTML, CSS",
        )

        save_profile = st.form_submit_button(
            "☁️ Save Profile to Cloud",
            use_container_width=True,
            type="primary",
        )

    if save_profile:
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
                st.success("Your profile was saved to Supabase.")
                st.rerun()
            except Exception as error:
                st.error("Could not save the profile.")
                st.caption(str(error))


# ============================================================
# PLACEMENT PREDICTION
# ============================================================

elif page == "📊 Placement Prediction":

    st.header("📊 Placement Prediction")

    st.write(
        "Enter your academic details to obtain an estimate from "
        "your trained scikit-learn model."
    )

    st.info(
        "The result is based on the model and its training data. "
        "It is not a guarantee of real-world placement."
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
            )

            projects = st.number_input(
                "Number of projects",
                min_value=0,
                max_value=50,
                value=2,
            )

            workshops = st.number_input(
                "Workshops / Certifications",
                min_value=0,
                max_value=50,
                value=2,
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

        student_data = pd.DataFrame(
            [{
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
            }]
        )

        try:
            prediction = model.predict(student_data)[0]

            st.markdown("### Your prediction")

            if str(prediction).strip().casefold() == "placed":
                st.success(
                    "The model estimates that you may be placed."
                )
            else:
                st.warning(
                    "The model estimates that additional preparation "
                    "may be helpful."
                )

            confidence = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(student_data)[0]
                classes = list(model.classes_)

                if prediction in classes:
                    index = classes.index(prediction)
                    confidence = round(
                        float(probabilities[index]) * 100,
                        2,
                    )

                    st.metric(
                        "Model confidence for predicted class",
                        f"{confidence:.2f}%",
                    )

                    st.caption(
                        "This probability is not your actual probability "
                        "of getting a job."
                    )

            details = student_data.iloc[0].to_dict()

            st.session_state["last_prediction"] = {
                "prediction": str(prediction),
                "confidence": confidence,
                "input_data": details,
            }

            if st.button(
                "☁️ Save Prediction to Cloud",
                key="save_prediction",
                use_container_width=True,
            ):
                try:
                    cloud_save(
                        "placementai_prediction_history",
                        {
                            "action": "placement_prediction",
                            "prediction": str(prediction),
                            "confidence": confidence,
                            "input_data": details,
                            "user_id": student_id,
                        },
                    )
                    st.success("Prediction saved to cloud storage.")
                except Exception as error:
                    st.error("Could not save prediction.")
                    st.caption(str(error))

        except Exception as error:
            st.error(
                "Prediction failed. Verify that the model expects "
                "the same input columns used in this application."
            )
            st.caption(str(error))


# ============================================================
# JOB ROLE RECOMMENDATIONS
# ============================================================

elif page == "💼 Job Role Recommendations":

    st.header("💼 Job Role Recommendations")

    st.write(
        "Explore IT job roles using your local job-role dataset."
    )

    if not job_titles:
        st.warning("No job titles were found in job_roles.csv.")

    else:
        selected_job = st.selectbox(
            "Choose an IT job role",
            job_titles,
            key="recommendation_job",
        )

        selected_rows = jobs_df[
            jobs_df["Job Title"] == selected_job
        ]

        if not selected_rows.empty:
            job = selected_rows.iloc[0]

            st.markdown(f"### {job['Job Title']}")

            st.markdown("#### Job description")
            st.write(
                job["Job Description"]
                or "No description available."
            )

            required_skills = [
                skill.strip()
                for skill in str(job["Skills"]).split(",")
                if skill.strip()
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
# SKILL GAP ANALYSIS
# ============================================================

elif page == "🧩 Skill Gap Analysis":

    st.header("🧩 Skill Gap Analysis")

    st.write(
        "Compare your existing skills against the skills listed "
        "for a selected job role."
    )

    if not job_titles:
        st.warning("No job roles are available.")
    else:
        selected_job = st.selectbox(
            "Choose your target job role",
            job_titles,
            key="gap_job",
        )

        job = jobs_df[
            jobs_df["Job Title"] == selected_job
        ].iloc[0]

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

        student_skills_text = st.text_area(
            "Your current skills (separate with commas)",
            value=default_skills,
            placeholder="Python, SQL, HTML, CSS",
        )

        if st.button(
            "🔍 Analyze My Skill Gap",
            type="primary",
            use_container_width=True,
        ):
            student_skills = normalize_skills(student_skills_text)

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

            st.markdown("### Your analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Required skills", len(required_skills))

            with col2:
                st.metric("Skills matched", len(matched_skills))

            with col3:
                st.metric("Skills to develop", len(missing_skills))

            if required_skills:
                match_percentage = (
                    len(matched_skills) / len(required_skills)
                ) * 100
            else:
                match_percentage = 0

            st.progress(int(match_percentage))
            st.write(f"**Exact skill-name match: {match_percentage:.1f}%**")

            left, right = st.columns(2)

            with left:
                st.markdown("#### ✅ Skills you have")

                if matched_skills:
                    for skill in matched_skills:
                        st.write(f"• {skill}")
                else:
                    st.info("No exact skill matches found.")

            with right:
                st.markdown("#### 📚 Skills to learn")

                if missing_skills:
                    for skill in missing_skills:
                        st.write(f"• {skill}")
                else:
                    st.success("You listed every skill in this dataset entry.")

            st.caption(
                "This analysis uses exact skill-name matching. "
                "It does not measure proficiency or certify expertise."
            )

            if st.button(
                "☁️ Save Skill Gap Analysis",
                key="save_gap_analysis",
                use_container_width=True,
            ):
                try:
                    save_history(
                        "skill_gap_analysis",
                        f"{match_percentage:.1f}% match for {selected_job}",
                        {
                            "job_title": selected_job,
                            "student_skills": student_skills_text,
                            "matched_skills": matched_skills,
                            "missing_skills": missing_skills,
                            "match_percentage": round(match_percentage, 1),
                        },
                    )
                    st.success("Skill-gap analysis saved.")
                except Exception as error:
                    st.error("Could not save analysis.")
                    st.caption(str(error))


# ============================================================
# CLOUD STORAGE
# ============================================================

elif page == "☁️ Cloud Storage":

    st.header("☁️ Cloud Storage")

    st.write(
        "Access your saved student profile and analysis history "
        "stored in Supabase."
    )

    st.markdown(
        """
        <div class="info-banner">
            🔐 Your records are associated with your authenticated
            student account. Configure Supabase Row Level Security
            to enforce access restrictions at the database level.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_profile, tab_history = st.tabs(
        ["👤 Saved Profile", "📚 Prediction & Analysis History"]
    )

    with tab_profile:
        try:
            profile = get_student_profile()

            if profile:
                st.markdown("### Your saved profile")

                st.write("**Full name:**", profile.get("full_name", ""))
                st.write("**Course:**", profile.get("course", ""))
                st.write(
                    "**Graduation year:**",
                    profile.get("graduation_year", ""),
                )
                st.write("**Skills:**", profile.get("skills", ""))
                st.caption(
                    f"Last updated: {profile.get('updated_at', 'Not available')}"
                )
            else:
                st.info(
                    "No profile saved yet. Open Student Profile "
                    "to add your details."
                )

        except Exception as error:
            st.error(
                "Could not load your cloud profile. "
                "Check the Supabase table and its permissions."
            )
            st.caption(str(error))

    with tab_history:
        if st.button("🔄 Refresh Cloud History"):
            st.rerun()

        try:
            response = cloud_select("placementai_prediction_history")
            history = response.data or []

            if not history:
                st.info("No saved predictions or analyses yet.")

            else:
                st.caption(f"{len(history)} saved record(s)")

                for index, item in enumerate(history):
                    action = item.get("action", "Saved record")
                    result = item.get("prediction", "")
                    created_at = item.get("created_at", "")

                    with st.expander(
                        f"{action.replace('_', ' ').title()} — "
                        f"{result} — {created_at}",
                        key=f"history_{index}",
                    ):
                        st.write("**Result:**", result)

                        if item.get("confidence") is not None:
                            st.write(
                                "**Model confidence:**",
                                f"{item['confidence']}%",
                            )

                        details = item.get("input_data", {})

                        if isinstance(details, str):
                            try:
                                details = json.loads(details)
                            except (json.JSONDecodeError, TypeError):
                                pass

                        st.write("**Details**")
                        st.json(details)

        except Exception as error:
            st.error(
                "Could not load history. Check the table name, "
                "columns, grants, and Row Level Security policies."
            )
            st.caption(str(error))


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