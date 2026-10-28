
import streamlit as st


def get_supabase_client():
    from supabase import create_client

    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"],
    )


def show_auth_page():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, #202650, transparent 40%),
                radial-gradient(circle at 90% 90%, #103b4b, transparent 40%),
                #0b1020;
            color: #f8fafc;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1250px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        .auth-hero {
            box-sizing: border-box;
            width: 100%;
            min-height: 530px;
            padding: 38px 30px;
            border-radius: 24px;
            background: linear-gradient(
                145deg,
                #20294e,
                #29265c 55%,
                #15465a
            );
            border: 1px solid #39476d;
            box-shadow: 0 20px 60px rgba(0,0,0,.22);
            color: #ffffff;
            font-family: sans-serif;
        }

        .auth-hero .eyebrow {
            color: #8ccaff;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 3px;
            margin-bottom: 25px;
        }

        .auth-hero h1 {
            color: #ffffff;
            font-size: 43px;
            font-weight: 850;
            line-height: 1.2;
            margin: 0 0 22px 0;
        }

        .auth-hero .description {
            color: #dce7ff;
            font-size: 16px;
            line-height: 1.8;
            margin-bottom: 26px;
        }

        .auth-feature {
            background: rgba(255,255,255,.07);
            border: 1px solid rgba(255,255,255,.09);
            border-radius: 12px;
            padding: 13px;
            margin: 12px 0;
            color: #f1f5f9;
            font-size: 14px;
        }

        .auth-footer {
            color: #a7c8ff;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1.5px;
            margin-top: 30px;
        }

        .auth-heading {
            color: #ffffff;
            font-size: 31px;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .auth-subheading {
            color: #aab8d0;
            font-size: 14px;
            line-height: 1.7;
            margin-bottom: 24px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            border-bottom: 1px solid #293753;
        }

        .stTabs [data-baseweb="tab"] {
            color: #cbd5e1;
            font-weight: 700;
            padding: 12px 15px;
        }

        .stTabs [aria-selected="true"] {
            color: #ffffff !important;
        }

        .stTextInput label {
            color: #e2e8f0 !important;
            font-weight: 600;
        }

        .stTextInput input {
            background-color: #151e32 !important;
            color: #ffffff !important;
            border: 1px solid #344463 !important;
            border-radius: 10px !important;
            min-height: 44px;
        }

        .stForm {
            background: rgba(21,30,50,.75);
            padding: 22px;
            border: 1px solid #293753;
            border-radius: 16px;
        }

        .stFormSubmitButton button {
            background: linear-gradient(90deg, #ff4b5c, #e83e70);
            color: white;
            border: none;
            border-radius: 10px;
            min-height: 47px;
            font-weight: 750;
        }

        .stFormSubmitButton button:hover {
            filter: brightness(1.1);
            border: 1px solid #ff8090;
        }

        @media (max-width: 768px) {
            .block-container {
                padding: 1rem;
            }

            .auth-hero {
                min-height: auto;
                padding: 26px 22px;
                margin-bottom: 24px;
            }

            .auth-hero h1 {
                font-size: 34px;
            }

            .auth-heading {
                font-size: 26px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        supabase = get_supabase_client()
    except Exception as error:
        st.error(
            "Could not connect to Supabase. "
            "Check your .streamlit/secrets.toml configuration."
        )
        st.caption(str(error))
        return

    left, right = st.columns(
        [1, 1],
        gap="large",
        vertical_alignment="center",
    )

    with left:
        # st.html renders HTML directly instead of treating it as Markdown.
        st.html("""
        <div class="auth-hero">
            <div class="eyebrow">STUDENT SUCCESS PLATFORM</div>
            <h1>🎓 PlacementAI</h1>

            <div class="description">
                Your journey to a successful career starts here.
                Prepare smarter, discover your strengths, and get ready
                for your dream job.
            </div>

            <div class="auth-feature">
                📊 &nbsp; AI-powered placement prediction
            </div>

            <div class="auth-feature">
                💼 &nbsp; Personalized job role recommendations
            </div>

            <div class="auth-feature">
                🧩 &nbsp; Identify and improve your skill gaps
            </div>

            <div class="auth-feature">
                ☁️ &nbsp; Secure cloud-based student storage
            </div>

            <div class="auth-footer">
                YOUR CAREER. YOUR SKILLS. YOUR FUTURE.
            </div>
        </div>
        """)

    with right:
        st.markdown(
            '<div class="auth-heading">Welcome to PlacementAI</div>'
            '<div class="auth-subheading">'
            'Sign in to continue or create your free student account.'
            '</div>',
            unsafe_allow_html=True,
        )

        login_tab, signup_tab = st.tabs(
            ["🔐 Login", "✨ Register"]
        )

        # LOGIN
        with login_tab:
            st.subheader("Welcome Back")
            st.write("Log in to continue to your student dashboard.")

            with st.form("login_form"):
                login_email = st.text_input(
                    "Email address",
                    placeholder="you@example.com",
                    key="login_email",
                )

                login_password = st.text_input(
                    "Password",
                    placeholder="Enter your password",
                    type="password",
                    key="login_password",
                )

                login_submitted = st.form_submit_button(
                    "Login to Your Account →",
                    use_container_width=True,
                    type="primary",
                )

            if login_submitted:
                if not login_email.strip() or not login_password:
                    st.warning("Enter both your email and password.")
                else:
                    try:
                        response = supabase.auth.sign_in_with_password({
                            "email": login_email.strip(),
                            "password": login_password,
                        })

                        if response.user and response.session:
                            st.session_state["access_token"] = (
                                response.session.access_token
                            )
                            st.session_state["refresh_token"] = (
                                response.session.refresh_token
                            )
                            st.session_state["student_id"] = (
                                str(response.user.id)
                            )
                            st.session_state["student_email"] = (
                                response.user.email or login_email.strip()
                            )

                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("Login did not return a valid session.")

                    except Exception:
                        st.error(
                            "Login failed. Check your email and password, "
                            "and verify your Supabase authentication settings."
                        )

        # REGISTRATION
        with signup_tab:
            st.subheader("Create Your Account")
            st.write(
                "Join PlacementAI and start preparing for your career."
            )

            with st.form("registration_form"):
                full_name = st.text_input(
                    "Full name",
                    placeholder="Enter your full name",
                    key="signup_name",
                )

                signup_email = st.text_input(
                    "Email address",
                    placeholder="you@example.com",
                    key="signup_email",
                )

                signup_password = st.text_input(
                    "Create password",
                    placeholder="At least 8 characters",
                    type="password",
                    key="signup_password",
                )

                confirm_password = st.text_input(
                    "Confirm password",
                    placeholder="Re-enter your password",
                    type="password",
                    key="confirm_password",
                )

                signup_submitted = st.form_submit_button(
                    "Create My Account →",
                    use_container_width=True,
                    type="primary",
                )

            if signup_submitted:
                if not full_name.strip() or not signup_email.strip():
                    st.warning("Enter your name and email address.")

                elif len(signup_password) < 8:
                    st.warning(
                        "Your password must contain at least 8 characters."
                    )

                elif signup_password != confirm_password:
                    st.warning("The passwords do not match.")

                else:
                    try:
                        response = supabase.auth.sign_up({
                            "email": signup_email.strip(),
                            "password": signup_password,
                            "options": {
                                "data": {
                                    "full_name": full_name.strip()
                                }
                            },
                        })

                        if response.user and response.session:
                            st.session_state["access_token"] = (
                                response.session.access_token
                            )
                            st.session_state["refresh_token"] = (
                                response.session.refresh_token
                            )
                            st.session_state["student_id"] = (
                                str(response.user.id)
                            )
                            st.session_state["student_email"] = (
                                response.user.email or signup_email.strip()
                            )

                            st.success("Registration successful!")
                            st.rerun()

                        elif response.user:
                            st.success(
                                "Your account was created, but Supabase "
                                "did not return a login session."
                            )
                            st.info(
                                "If email confirmation is enabled, "
                                "confirm your email before logging in."
                            )

                        else:
                            st.error(
                                "Supabase did not return a user. "
                                "Please try again."
                            )

                    except Exception as error:
                        st.error("Registration failed.")
                        st.caption(str(error))

            st.caption(
                "Create your account to access your personal "
                "placement preparation workspace."
            )
