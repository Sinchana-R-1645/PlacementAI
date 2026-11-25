
import requests
import streamlit as st


JOOBLE_API_URL = "https://in.jooble.org/api/"


def search_jooble(keywords, location, limit=5):
    """Fetch real job listings from Jooble India."""

    api_key = st.secrets.get("JOOBLE_API_KEY", "")

    if not api_key:
        return {
            "success": False,
            "jobs": [],
            "total": 0,
            "error": "Jooble API key is not configured.",
        }

    payload = {
        "keywords": keywords,
        "location": location,
        "page": "1",
        "ResultOnPage": str(limit),
    }

    try:
        response = requests.post(
            f"{JOOBLE_API_URL}{api_key}",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=20,
        )

        if response.status_code == 403:
            return {
                "success": False,
                "jobs": [],
                "total": 0,
                "error": "Jooble rejected the API key. Check your key and region.",
            }

        if response.status_code == 429:
            return {
                "success": False,
                "jobs": [],
                "total": 0,
                "error": "Jooble request limit reached. Try again later.",
            }

        response.raise_for_status()
        data = response.json()

        jobs = []

        for job in data.get("jobs", []):
            jobs.append({
                "source": "Jooble",
                "title": job.get("title", "Title unavailable"),
                "company": job.get("company", "Company not listed"),
                "location": job.get("location", location),
                "description": job.get("snippet", ""),
                "url": job.get("link", ""),
                "salary": job.get("salary", ""),
                "job_type": job.get("type", ""),
                "updated": job.get("updated", ""),
                "id": str(job.get("id", "")),
            })

        return {
            "success": True,
            "jobs": jobs,
            "total": data.get("totalCount", len(jobs)),
            "error": None,
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "jobs": [],
            "total": 0,
            "error": "Jooble request timed out. Please try again.",
        }

    except requests.exceptions.RequestException:
        return {
            "success": False,
            "jobs": [],
            "total": 0,
            "error": "Could not connect to Jooble. Please try again later.",
        }

    except (ValueError, TypeError):
        return {
            "success": False,
            "jobs": [],
            "total": 0,
            "error": "Jooble returned an unexpected response.",
        }
