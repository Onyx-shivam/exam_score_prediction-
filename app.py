import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

st.set_page_config(
    page_title="KNN Prediction Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "knn.pkl"

# ---------------------------------------------------------------
# Styling
# Colors used:  text #17252a | teal #1f6f6b | dark teal #144e4b
#               page #f3f6f8 | border #dde5e9 | muted #5b6b73
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700;800&family=Figtree:wght@400;500;600;700&display=swap');

/* Page */
.stApp {
    background: #f3f6f8;
    font-family: 'Figtree', sans-serif;
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
}

/* Headings and text in the main area (always dark, so they stay readable) */
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: 'Bricolage Grotesque', sans-serif;
    color: #17252a;
}
[data-testid="stWidgetLabel"] p {
    color: #17252a !important;
    font-weight: 600;
    font-size: 0.92rem;
}
[data-testid="stCaptionContainer"] {
    color: #5b6b73 !important;
}
[data-testid="stMain"] p,
section.main p,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #17252a;
}
[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #dde5e9;
    border-radius: 12px;
}

/* Hero banner */
.hero {
    background: #144e4b;
    padding: 34px 38px;
    border-radius: 20px;
    margin-bottom: 26px;
}
.hero h1 {
    color: #ffffff !important;
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0 0 8px 0;
    padding: 0;
}
.hero p {
    color: #cfe5e3 !important;
    font-size: 1.05rem;
    margin: 0;
    max-width: 620px;
}

/* The form is the card */
div[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #dde5e9;
    border-radius: 18px;
    padding: 28px 30px 24px 30px;
}

/* Group titles inside the form */
.group-title {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #144e4b;
    padding-bottom: 8px;
    margin-bottom: 14px;
    border-bottom: 2px solid #e4efee;
}

/* Input boxes: white with dark text */
div[data-baseweb="input"],
div[data-baseweb="select"] > div {
    background: #f8fafb !important;
    border-radius: 10px !important;
    border: 1px solid #dde5e9 !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="select"] div {
    color: #17252a !important;
}
div[data-baseweb="input"] button {
    background: transparent !important;
    color: #17252a !important;
}

/* Submit button */
div[data-testid="stFormSubmitButton"] > button {
    background: #1f6f6b;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: 700;
}
div[data-testid="stFormSubmitButton"] > button p {
    color: #ffffff !important;
    font-size: 1rem;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    background: #144e4b;
}

/* Metric card */
div[data-testid="stMetric"] {
    background: #f1f8f7;
    border: 1px solid #cfe5e3;
    border-radius: 14px;
    padding: 18px 20px;
}
div[data-testid="stMetricValue"] {
    color: #144e4b !important;
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 800;
}
div[data-testid="stMetricLabel"] p {
    color: #5b6b73 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f2f2e;
}
section[data-testid="stSidebar"] * {
    color: #e6f1f0 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #2a5553;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    # Load only pickle files you trust.
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


def neighbor_check(model, input_df, pass_mark):
    # Looks at the K nearest students the model used and counts how many
    # of them passed. Returns "" if the model type does not allow this.
    try:
        if hasattr(model, "steps"):
            estimator = model.steps[-1][1]
            data = model[:-1].transform(input_df)
        else:
            estimator = model
            data = input_df

        if "Regressor" not in type(estimator).__name__:
            return ""

        distances, indexes = estimator.kneighbors(data)
        neighbor_scores = np.asarray(estimator._y)[indexes[0]]
        total = len(neighbor_scores)
        passed = int((neighbor_scores >= pass_mark).sum())
        return str(passed) + " of " + str(total) + " similar students in the training data passed."
    except Exception:
        return ""


# ---------------------------------------------------------------
# One input widget for each feature name
# ---------------------------------------------------------------
def add_input(feature):
    name = str(feature).lower()
    key = "input_" + str(feature)

    if name == "age":
        return st.number_input("Age", min_value=10, max_value=100,
                               value=21, step=1, key=key)

    if name == "gender":
        return st.selectbox("Gender (encoded)", [0, 1], key=key)

    if name == "course":
        return st.number_input("Course (encoded category)", min_value=0,
                               max_value=100, value=0, step=1, key=key)

    if name == "study_hours":
        return st.slider("Study hours per day", 0.0, 24.0,
                         3.0, 0.5, key=key)

    if name == "class_attendance":
        return st.slider("Class attendance (%)", 0, 100,
                         80, 5, key=key)

    if name == "study_method":
        return st.number_input("Study method (encoded category)",
                               min_value=0, max_value=100,
                               value=0, step=1, key=key)

    if name == "exam_difficulty":
        return st.selectbox("Exam difficulty (encoded)",
                            [0, 1, 2, 3, 4], key=key)

    if name == "sleep_hours":
        return st.slider("Sleep hours per night", 0.0, 16.0,
                         7.0, 0.5, key=key)

    if name == "sleep_quality":
        return st.selectbox("Sleep quality (encoded)",
                            [0, 1, 2, 3, 4], key=key)

    if name == "facility_rating":
        return st.selectbox("Facility rating (encoded)",
                            [0, 1, 2, 3, 4], key=key)

    if name in ("internet_access", "internet_accesss"):
        return st.selectbox(str(feature) + " (encoded)", [0, 1], key=key)

    # Any other feature gets a plain number box
    return st.number_input(str(feature) + " (numeric)", value=0.0, key=key)


def main():
    # ---------- Header ----------
    st.markdown("""
    <div class="hero">
        <h1>KNN Prediction Studio</h1>
        <p>Fill in the student details below and get a prediction
        from your trained K-Nearest Neighbors model.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- Load model ----------
    if not MODEL_PATH.exists():
        st.error("knn.pkl not found. Place it in the same folder as app.py.")
        st.stop()

    try:
        model = load_model()
    except Exception as exc:
        st.error("Unable to load the model. Check that the library versions match the ones used for training.")
        st.code(str(exc))
        st.stop()

    # ---------- Feature names ----------
    default_features = [
        "age", "gender", "course", "study_hours",
        "class_attendance", "internet_access",
        "sleep_hours", "sleep_quality", "study_method",
        "facility_rating", "exam_difficulty", "internet_accesss",
    ]
    features = list(getattr(model, "feature_names_in_", default_features))
    expected_count = getattr(model, "n_features_in_", len(features))

    if len(features) != expected_count:
        st.error("Feature count mismatch. Check your model's training schema.")
        st.stop()

    # ---------- Sidebar ----------
    with st.sidebar:
        st.markdown("## 🎯 KNN Studio")
        st.write("Prediction dashboard")
        st.divider()
        st.markdown("### Model")
        st.write("**Algorithm:** " + type(model).__name__)
        st.write("**Input features:** " + str(expected_count))
        st.divider()
        st.markdown("### Result rule")
        pass_mark = st.number_input("Pass mark (score needed to pass)",
                                    min_value=0.0, max_value=100.0,
                                    value=40.0, step=1.0)
        margin = st.number_input("Borderline margin (+/- marks)",
                                 min_value=0.0, max_value=20.0,
                                 value=3.0, step=0.5)
        st.divider()
        st.caption("Use the same feature order, encodings and scales "
                   "as your training data.")

    # ---------- Group the features ----------
    group_plan = [
        ("Personal details", ["age", "gender", "course"]),
        ("Study habits", ["study_hours", "class_attendance",
                          "study_method", "exam_difficulty"]),
        ("Lifestyle and environment", ["sleep_hours", "sleep_quality",
                                       "facility_rating", "internet_access",
                                       "internet_accesss"]),
    ]

    groups = []
    used = []
    for title, names in group_plan:
        members = []
        for f in features:
            if str(f).lower() in names:
                members.append(f)
                used.append(f)
        if len(members) > 0:
            groups.append((title, members))

    # features that did not match any group
    others = []
    for f in features:
        if f not in used:
            others.append(f)
    if len(others) > 0:
        groups.append(("Other features", others))

    # ---------- Input form ----------
    values = {}

    with st.form("knn_prediction_form"):
        columns = st.columns(len(groups), gap="large")

        for i in range(len(groups)):
            title, members = groups[i]
            with columns[i]:
                st.markdown('<div class="group-title">' + title + '</div>',
                            unsafe_allow_html=True)
                for feature in members:
                    values[feature] = add_input(feature)

        submitted = st.form_submit_button("Generate prediction",
                                          use_container_width=True)

    # ---------- Result ----------
    # Input check: a day only has 24 hours
    hours_ok = True
    if submitted:
        if "study_hours" in values and "sleep_hours" in values:
            if values["study_hours"] + values["sleep_hours"] > 24:
                hours_ok = False
                st.error("Study hours plus sleep hours cannot be more than "
                         "24 in a day. Please correct the values.")

    if submitted and hours_ok:
        try:
            input_df = pd.DataFrame(
                [[values[f] for f in features]],
                columns=features,
            )

            prediction = np.asarray(model.predict(input_df)).reshape(-1)

            if prediction.size == 0:
                raise ValueError("The model returned no prediction.")

            result = prediction[0]

            # Decide PASS or FAIL
            neighbor_text = ""
            try:
                score = float(result)
                score_text = f"{score:,.2f}"
                if score >= pass_mark + margin:
                    status = "PASS"
                    badge_bg = "#dcfce7"
                    badge_text = "#166534"
                    message = "Predicted score is clearly above the pass mark."
                elif score >= pass_mark - margin:
                    status = "BORDERLINE"
                    badge_bg = "#fef3c7"
                    badge_text = "#92400e"
                    message = ("Predicted score is within " + str(margin) +
                               " marks of the pass mark. Treat this result with caution.")
                else:
                    status = "FAIL"
                    badge_bg = "#fee2e2"
                    badge_text = "#991b1b"
                    message = "Predicted score is clearly below the pass mark."
                neighbor_text = neighbor_check(model, input_df, pass_mark)
            except Exception:
                # The model gave text (for example "Pass" or "Fail")
                score_text = str(result)
                status = str(result).upper()
                if status == "PASS":
                    badge_bg = "#dcfce7"
                    badge_text = "#166534"
                else:
                    badge_bg = "#fee2e2"
                    badge_text = "#991b1b"
                message = "Result given directly by the model."

            st.write("")
            st.subheader("Result")

            extra = ""
            if neighbor_text != "":
                extra = ('<div style="color:#5b6b73;font-size:0.9rem;'
                         'margin-top:8px;">' + neighbor_text + '</div>')

            card = ('<div style="background:#ffffff;border:1px solid #dde5e9;'
                    'border-radius:18px;padding:26px 30px;display:flex;'
                    'align-items:center;gap:36px;flex-wrap:wrap;">'
                    '<div>'
                    '<div style="color:#5b6b73;font-size:0.9rem;'
                    'font-weight:600;">Predicted score</div>'
                    '<div style="color:#144e4b;font-size:2.8rem;'
                    'font-weight:800;line-height:1.1;">' + score_text + '</div>'
                    '<div style="color:#5b6b73;font-size:0.85rem;">'
                    'Pass mark: ' + str(pass_mark) + '</div>'
                    '</div>'
                    '<div style="background:' + badge_bg + ';color:' + badge_text + ';'
                    'font-size:1.8rem;font-weight:800;padding:14px 34px;'
                    'border-radius:14px;">' + status + '</div>'
                    '<div style="color:#17252a;font-size:1rem;flex:1;'
                    'min-width:200px;">' + message + extra + '</div>'
                    '</div>')
            st.markdown(card, unsafe_allow_html=True)

            st.write("")
            with st.expander("View submitted input values"):
                st.dataframe(input_df, use_container_width=True,
                             hide_index=True)

        except Exception as exc:
            st.error("Prediction failed. Check your input schema and encodings.")
            st.code(str(exc))
            st.info("If you used a scaler or encoder during training, "
                    "include it in the saved model (for example in a Pipeline).")

    st.caption("Built with Streamlit and scikit-learn.")


if __name__ == "__main__":
    main()
