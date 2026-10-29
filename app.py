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
        "Maintain your academic information and current skills. "
        "Your profile is saved to your authenticated Supabase account."
    )

    try:
        profile = get_student_profile()
    except Exception as error:
        profile = {}
        st.warning("Could not load your saved profile.")
        st.caption(str(error))

    with st.form("student_profile_form"):
        full_name = st.text_input(
            "Full name",
            value=profile.get("full_name", ""),
        )

        course = st.text_input(
            "Course / Degree",
            value=profile.get("course", ""),
            placeholder="B.Tech Computer Science",
        )

        current_year = datetime.now().year
        saved_year = profile.get("graduation_year", current_year + 1)

        try:
            saved_year = int(saved_year)
        except (ValueError, TypeError):
            saved_year = current_year + 1

        graduation_year = st.number_input(
            "Graduation year",
            min_value=current_year,
            max_value=current_year + 10,
            value=min(max(saved_year, current_year), current_year + 10),
            step=1,
        )

        skills = st.text_area(
            "Current skills (comma-separated)",
            value=profile.get("skills", ""),
            placeholder="Python, SQL, Communication, HTML",
        )

        save_profile = st.form_submit_button(
            "💾 Save Student Profile",
            use_container_width=True,
        )

    if save_profile:
        if not full_name.strip():
            st.error("Please enter your full name.")
        else:
            try:
                save_student_profile(
                    full_name,
                    course,
                    graduation_year,
                    skills,
                )
                st.success("Your profile has been saved.")
                st.rerun()
            except Exception as error:
                st.error("Could not save your profile.")
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

                    input_data[column] = st.number_input(
                        column.replace("_", " ").title(),
                        value=float(median_value),
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

    if predict_clicked:
        try:
            prediction_frame = pd.DataFrame(
                [input_data],
                columns=feature_columns,
            )

            prediction = model.predict(prediction_frame)[0]

            confidence = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(prediction_frame)[0]
                confidence = round(float(max(probabilities)) * 100, 2)

            st.markdown("### Prediction Result")
            st.success(f"Predicted result: **{prediction}**")

            if confidence is not None:
                st.metric("Model confidence", f"{confidence:.1f}%")

            try:
                save_history(
                    "placement_prediction",
                    prediction,
                    {
                        "input_data": input_data,
                        "confidence": confidence,
                    },
                )
                st.caption("Prediction saved to your cloud history.")
            except Exception as error:
                st.warning("Prediction was made but could not be saved.")
                st.caption(str(error))

        except Exception as error:
            st.error("Could not generate a placement prediction.")
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
            "url": "https://www.coursera.org/",
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

                                resource_url = item.get("resource_url") or plan["url"]
                                st.markdown(
                                    f"[Open learning reference]({resource_url})"
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

                                st.markdown("### Final quiz")

                                quiz = item.get("quiz") or plan["quiz"]
                                if isinstance(quiz, str):
                                    try:
                                        quiz = json.loads(quiz)
                                    except json.JSONDecodeError:
                                        quiz = plan["quiz"]

                                quiz_answers = []

                                for question_index, question in enumerate(quiz):
                                    question_text, options, correct_index = question[:3]

                                    answer = st.radio(
                                        question_text,
                                        options,
                                        key=f"quiz_{selected_job}_{item_index}_{question_index}",
                                    )
                                    quiz_answers.append(options.index(answer))

                                if st.button(
                                    "🧪 Submit quiz and save progress",
                                    key=f"submit_{selected_job}_{item_index}",
                                    use_container_width=True,
                                ):
                                    score = sum(
                                        answer == question[2]
                                        for answer, question in zip(quiz_answers, quiz)
                                    )
                                    passed = score >= max(1, (len(quiz) * 2 + 2) // 3)

                                    updated = {
                                        **item,
                                        "completed_lessons": updated_completed,
                                        "quiz_score": score,
                                        "quiz_passed": passed,
                                        "status": (
                                            "Completed"
                                            if passed and len(updated_completed) == len(lessons)
                                            else "In Progress"
                                        ),
                                    }

                                    try:
                                        save_skill_roadmap_item(selected_job, updated)
                                        st.success(
                                            f"Quiz score: {score}/{len(quiz)}. "
                                            + ("Passed!" if passed else "Keep practising and try again.")
                                        )
                                        st.rerun()
                                    except Exception as error:
                                        st.error("Could not save your quiz progress.")
                                        st.caption(str(error))

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