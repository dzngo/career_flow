import ast
import json

import pandas as pd
import streamlit as st
from streamlit_tags import st_tags

# Page settings must be first Streamlit command
st.set_page_config(page_title="CareerFlow JD Annotation Tool", layout="wide")

# --- improve contrast of disabled text areas ---
st.markdown(
    """
    <style>
    /* Light theme: match normal input style */
    @media (prefers-color-scheme: light) {
      .stTextArea textarea:disabled {
        color: #31333F !important;            /* dark text */
        background-color: #F0F2F6 !important; /* input bg */
        border: 1px solid #D3D3D3 !important; /* same border */
        opacity: 1 !important;               /* unfaded */
        -webkit-text-fill-color: #31333F !important;
      }
    }

    /* Dark theme: match normal input style */
    @media (prefers-color-scheme: dark) {
      .stTextArea textarea:disabled {
        color: #FAFAFA !important;            /* light text */
        background-color: #586E75 !important; /* input bg */
        border: 1px solid #AAAAAA !important; /* subtle border */
        opacity: 1 !important;
        -webkit-text-fill-color: #FAFAFA !important;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- compatibility helper: Streamlit 1.31 renamed experimental_rerun -> rerun ---
def _rerun():
    """Call the correct rerun function regardless of Streamlit version."""
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:  # Streamlit ≥ 1.31
        st.rerun()


def _safe_json_loads(val):
    """Return Python object from JSON/str/dict; fallback to empty dict."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            pass
    return {}


def _handle_nan_str(val):
    """Return an empty string for None or NaN, otherwise cast to string."""
    if isinstance(val, float) and pd.isna(val):
        return ""
    if val is None:
        return ""
    return str(val)


def _to_list(val):
    """Ensure the returned value is a list of strings."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return [v.strip() for v in val.split(",") if v.strip()]
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    return [str(val)]


st.title("CareerFlow JD Annotation Tool")

# Upload CSV
uploaded_file = st.file_uploader("Upload AI-annotated JD CSV", type=["csv"])
if uploaded_file:
    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_file)
    df = st.session_state.df
    idx = st.session_state.get("idx", 0)

    # Current row
    row = df.iloc[idx]

    st.subheader(f"JD {idx+1} of {len(df)}")
    # ---- two‑column layout ----
    col_left, col_right = st.columns([5, 3])

    # ---------------- LEFT COLUMN : read‑only raw JD ----------------
    with col_left:
        st.markdown("### Raw Job Text")
        st.text_area(
            "Raw Job Text",
            value=row["raw_job_text"],
            height=800,
            disabled=True,
        )

    # --------------- RIGHT COLUMN : editable annotations ------------
    with col_right:
        st.markdown("### Annotation")
        # ---- Title & Industry ----
        title_col, industry_col = st.columns(2)
        with title_col:
            row["title"] = st.text_input("Job Title", value=_handle_nan_str(row.get("title", "")))
        with industry_col:
            row["industry"] = st.text_input("Industry", value=_handle_nan_str(row.get("industry", "")))

        # ---- Employment Type & Contract in two columns ----
        emp_type_col, emp_contract_col = st.columns(2)
        with emp_type_col:
            row["employment_type"] = st.text_input(
                "Employment Type", value=_handle_nan_str(row.get("employment_type", ""))
            )
        with emp_contract_col:
            row["employment_contract"] = st.text_input(
                "Employment Contract", value=_handle_nan_str(row.get("employment_contract", ""))
            )

        # ---- nested: Skills ----------------------------------------------
        st.markdown("##### Skills")
        skills_data = _safe_json_loads(row.get("skills", {}))

        hard_list = _to_list(skills_data.get("hard_skills", []))
        soft_list = _to_list(skills_data.get("soft_skills", []))
        lang_list = _to_list(skills_data.get("required_languages", []))
        nice_list = _to_list(skills_data.get("nice_to_have", []))

        # ---- Skills in two rows of two columns -----------------------
        col_hard, col_soft = st.columns(2)
        with col_hard:
            hard = st_tags(
                label="Hard skills",
                text="Press enter to add",
                value=hard_list,
                suggestions=hard_list,
                key=f"hard_tags_{idx}",
            )
        with col_soft:
            soft = st_tags(
                label="Soft skills",
                text="Press enter to add",
                value=soft_list,
                suggestions=soft_list,
                key=f"soft_tags_{idx}",
            )

        col_langs, col_nice = st.columns(2)
        with col_langs:
            langs = st_tags(
                label="Required languages",
                text="Press enter to add",
                value=lang_list,
                suggestions=lang_list,
                key=f"lang_tags_{idx}",
            )
        with col_nice:
            nice = st_tags(
                label="Nice-to-have skills",
                text="Press enter to add",
                value=nice_list,
                suggestions=nice_list,
                key=f"nice_tags_{idx}",
            )

        # ---- nested: Required Experience ---------------------------
        st.markdown("##### Required Experience")
        req_exp = _safe_json_loads(row.get("required_experience", {}))
        years_dict = req_exp.get("years", {})

        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            years_min = st.number_input("Min years", min_value=-1, value=int(years_dict.get("min", 0)), step=1)
        with exp_col2:
            years_max = st.number_input("Max years", min_value=-1, value=int(years_dict.get("max", 0)), step=1)
        with exp_col3:
            level = st.text_input("Experience level", value=req_exp.get("level", ""))

        # ---- nested: Salary ----------------------------------------
        st.markdown("##### Salary")
        salary_data = _safe_json_loads(row.get("salary", {}))

        sal_col1, sal_col2, sal_col3 = st.columns(3)
        with sal_col1:
            sal_min = st.number_input(
                "Salary min", min_value=-1.0, value=float(salary_data.get("min", 0)), step=1000.0, format="%.0f"
            )
        with sal_col2:
            sal_max = st.number_input(
                "Salary max", min_value=-1.0, value=float(salary_data.get("max", 0)), step=1000.0, format="%.0f"
            )
        with sal_col3:
            currency = st.text_input("Currency", value=salary_data.get("currency", ""))

        # ---- nested: Education -------------------------------------
        st.markdown("##### Education")
        edu_data = _safe_json_loads(row.get("education", {}))

        edu_col1, edu_col2 = st.columns(2)
        with edu_col1:
            degrees = st.text_input("Degrees", value=edu_data.get("degrees", ""))
        with edu_col2:
            fields = st.text_input("Fields of study", value=edu_data.get("fields_of_study", ""))

        # ---- navigation buttons (auto‑save) ----
        nav_prev, nav_next = st.columns(2)

        def _commit_edits():
            # save simple fields
            st.session_state.df.at[idx, "title"] = row["title"]
            st.session_state.df.at[idx, "industry"] = row["industry"]
            st.session_state.df.at[idx, "employment_type"] = row["employment_type"]
            st.session_state.df.at[idx, "employment_contract"] = row["employment_contract"]

            # build updated skills dict then dump to JSON string
            skills_data["hard_skills"] = hard
            skills_data["soft_skills"] = soft
            skills_data["required_languages"] = langs
            skills_data["nice_to_have"] = nice
            st.session_state.df.at[idx, "skills"] = json.dumps(skills_data, ensure_ascii=False)

            # required experience
            req_exp["years"] = {"min": int(years_min), "max": int(years_max)}
            req_exp["level"] = level.strip()
            st.session_state.df.at[idx, "required_experience"] = json.dumps(req_exp, ensure_ascii=False)

            # salary
            salary_data["min"] = sal_min
            salary_data["max"] = sal_max
            salary_data["currency"] = currency.strip()
            st.session_state.df.at[idx, "salary"] = json.dumps(salary_data, ensure_ascii=False)

            # education
            edu_data["degrees"] = degrees.strip()
            edu_data["fields_of_study"] = fields.strip()
            st.session_state.df.at[idx, "education"] = json.dumps(edu_data, ensure_ascii=False)

        if nav_prev.button("Previous"):
            _commit_edits()
            if idx > 0:
                st.session_state["idx"] = idx - 1
            _rerun()

        if nav_next.button("Next"):
            _commit_edits()
            if idx < len(df) - 1:
                st.session_state["idx"] = idx + 1
            _rerun()

    # Download annotated version
    st.download_button(
        "Download Updated CSV", data=st.session_state.df.to_csv(index=False), file_name="corrected_jd.csv"
    )
