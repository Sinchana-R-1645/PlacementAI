
import streamlit as st


def get_supabase_client():
    """Create a Supabase client using Streamlit secrets."""
    from supabase import create_client

    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"],
    )


def show_auth_page():
    """Display the PlacementAI login and registration page."""

    st.markdown(
        """
        <div style="text-align:center; padding:25px 0 10px 0;">
            <h1>🎓 PlacementAI</h1>
            <p style="font-size:18px;">
                Your personal placement preparation platform
            </p>
            <p>
                Placement Prediction • Job Recommendations • Skill Gap Analysis
            </p>
        </div>
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

    login_tab, signup_tab = st.tabs(
        ["🔐 Login", "✨ Register"]
    )

    # ---------------- LOGIN ----------------
    with login_tab:
        st.subheader("Welcome Back")
        st.write("Log in to continue to your student dashboard.")

        with st.form("login_form"):
            login_email = st.text_input(
                "Email address",
                key="login_email",
            )

            login_password = st.text_input(
                "Password",
                type="password",
                key="login_password",
            )

            login_submitted = st.form_submit_button(
                "Login",
                use_container_width=True,
                type="primary",
            )

        if login_submitted:
            if not login_email.strip() or not login_password:
                st.warning("Enter both your email and password.")
            else:
                try:
                    response = supabase.auth.sign_in_with_password(
                        {
                            "email": login_email.strip(),
                            "password": login_password,
                        }
                    )

                    if not response.user or not response.session:
                        st.error(
                            "Login did not return a valid session. "
                            "Please try again."
                        )
                    else:
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

                except Exception as error:
                    st.error(
                        "Login failed. Check your email and password "
                        "and confirm that your account is eligible to sign in."
                    )
                    st.caption(str(error))

    # ---------------- REGISTRATION ----------------
    with signup_tab:
        st.subheader("Create Your Account")
        st.write("Register to save your placement preparation progress.")

        with st.form("registration_form"):
            full_name = st.text_input(
                "Full name",
                key="signup_name",
            )

            signup_email = st.text_input(
                "Email address",
                key="signup_email",
            )

            signup_password = st.text_input(
                "Create password",
                type="password",
                key="signup_password",
            )

            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                key="confirm_password",
            )

            signup_submitted = st.form_submit_button(
                "Create Account",
                use_container_width=True,
                type="primary",
            )

        if signup_submitted:
            if not full_name.strip() or not signup_email.strip():
                st.warning("Enter your name and email address.")

            elif len(signup_password) < 8:
                st.warning("Your password must contain at least 8 characters.")

            elif signup_password != confirm_password:
                st.warning("The passwords do not match.")

            else:
                try:
                    response = supabase.auth.sign_up(
                        {
                            "email": signup_email.strip(),
                            "password": signup_password,
                            "options": {
                                "data": {
                                    "full_name": full_name.strip(),
                                }
                            },
                        }
                    )

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
                            "Your account was created, but no login session "
                            "was returned. If email confirmation is enabled "
                            "in Supabase, confirm your email and then log in."
                        )

                    else:
                        st.error(
                            "Supabase did not return a user. Please try again."
                        )

                except Exception as error:
                    st.error("Registration failed.")
                    st.caption(str(error))