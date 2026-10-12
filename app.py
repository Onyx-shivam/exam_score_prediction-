
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

st.set_page_config(
    page_title="KNN Prediction Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).parent / "knn.pkl"

# Custom UI styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at top left, #e8eaff, transparent 40%),
        #f6f8fc;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #111827, #312e81, #2563eb);
    padding: 35px;
    border-radius: 24px;
    color: black;
    margin-bottom: 28px;
    box-shadow: 0 18px 40px rgba(49, 46, 129, 0.25);
}

.hero h1 {
    font-family: 'Manrope', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #dbeafe;
    font-size: 1rem;
}

.card {
    background: white;
    border: 1px solid #e7eaf3;
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.07);
    margin-bottom: 20px;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e7eaf3;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}

.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(100deg, #4f46e5, #2563eb);
    color: black;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: 700;
    box-shadow: 0 8px 18px rgba(79, 70, 229, 0.22);
    transition: 0.2s ease;
}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 12px 24px rgba(79, 70, 229, 0.3);
}

div[data-testid="stSidebar"] {
    background: #111827;
}

div[data-testid="stSidebar"] * {
    color: #f8fafc;
}

div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #cbd5e1;
}

h2, h3 {
    font-family: 'Manrope', sans-serif;
}

.footer {
    color: #64748b;
    text-align: center;
    padding: 20px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    # Load only pickle files you trust.
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


def main():
    st.markdown("""
    <div class="hero">
        <p>🎯 MACHINE LEARNING · PREDICTION SYSTEM</p>
        <h1>KNN Prediction Studio</h1>
        <p>Enter your feature values and generate a prediction
        using your trained K-Nearest Neighbors model.</p>
    </div>
    """, unsafe_allow_html=True)

    if not MODEL_PATH.exists():
        st.error("knn.pkl not found. Place it beside app.py.")
        st.stop()

    try:
        model = load_model()
    except Exception as exc:
        st.error("Unable to load the model. Check its compatibility.")
        st.code(str(exc))
        st.stop()

    # Use original feature names when available.
    default_features = [
        "age", "gender", "course", "study_hours",
        "class_attendance", "internet_access",
        "sleep_hours", "sleep_quality", "study_method",
        "facility_rating", "exam_difficulty", "internet_accesss"
    ]

    features = list(
        getattr(model, "feature_names_in_", default_features)
    )

    expected_count = getattr(model, "n_features_in_", len(features))

    if len(features) != expected_count:
        st.error("Feature count mismatch. Check your model's training schema.")
        st.stop()

    with st.sidebar:
        st.markdown("## 🎯 KNN Studio")
        st.write("Your machine learning dashboard")
        st.divider()
        st.markdown("### Model information")
        st.write(f"**Algorithm:** {type(model).__name__}")
        st.write(f"**Input features:** {expected_count}")
        st.divider()
        st.caption(
            "Use the same feature order, categorical encodings "
            "and numerical scales as your training data."
        )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧑‍🎓 Student Profile")
    st.caption("Complete the following fields to generate a prediction.")

    values = {}

    def add_input(feature, key):
        name = str(feature).lower()

        if name == "age":
            return st.number_input(
                "Age", min_value=10, max_value=100,
                value=21, step=1, key=key
            )

        if name == "gender":
            return st.selectbox(
                "Gender (encoded)", [0, 1], key=key
            )

        if name == "course":
            return st.number_input(
                "Course (encoded category)", min_value=0,
                max_value=100, value=0, step=1, key=key
            )

        if name == "study_hours":
            return st.slider(
                "Study hours per day", 0.0, 24.0,
                3.0, 0.5, key=key
            )

        if name == "class_attendance":
            return st.slider(
                "Class attendance (%)", 0, 100,
                80, 5, key=key
            )

        if name in ("internet_access", "internet_accesss"):
            return st.selectbox(
                f"{feature} (encoded)", [0, 1], key=key
            )

        if name == "sleep_hours":
            return st.slider(
                "Sleep hours per night", 0.0, 16.0,
                7.0, 0.5, key=key
            )

        if name == "sleep_quality":
            return st.selectbox(
                "Sleep quality (encoded)", [0, 1, 2, 3, 4],
                key=key
            )

        if name == "study_method":
            return st.number_input(
                "Study method (encoded category)",
                min_value=0, max_value=100,
                value=0, step=1, key=key
            )

        if name == "facility_rating":
            return st.selectbox(
                "Facility rating (encoded)",
                [0, 1, 2, 3, 4], key=key
            )

        if name == "exam_difficulty":
            return st.selectbox(
                "Exam difficulty (encoded)",
                [0, 1, 2, 3, 4], key=key
            )

        return st.number_input(
            f"{feature} (numeric)",
            value=0.0, key=key
        )

    midpoint = (len(features) + 1) // 2

    with st.form("knn_prediction_form"):
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("#### 📋 Personal & Learning Details")
            for feature in features[:midpoint]:
                values[feature] = add_input(
                    feature, f"left_{feature}"
                )

        with right:
            st.markdown("#### 📚 Additional Details")
            for feature in features[midpoint:]:
                values[feature] = add_input(
                    feature, f"right_{feature}"
                )

        submitted = st.form_submit_button(
            "✨ Generate Prediction",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        try:
            input_df = pd.DataFrame(
                [[values[f] for f in features]],
                columns=features
            )

            prediction = np.asarray(
                model.predict(input_df)
            ).reshape(-1)

            if prediction.size == 0:
                raise ValueError("The model returned no prediction.")

            result = prediction[0]

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Prediction Results")

            col1, col2 = st.columns([1, 2])

            with col1:
                if np.issubdtype(
                    np.asarray(result).dtype, np.number
                ):
                    st.metric(
                        "Predicted Value",
                        f"{float(result):,.3f}"
                    )
                else:
                    st.metric("Predicted Value", str(result))

            with col2:
                st.success("Prediction generated successfully!")
                st.write(
                    "Interpret this output according to the target "
                    "variable used when training your model."
                )

            with st.expander("View submitted input values"):
                st.dataframe(
                    input_df,
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as exc:
            st.error("Prediction failed. Check your input schema and encodings.")
            st.code(str(exc))
            st.info(
                "If preprocessing was used during training, make sure "
                "the required scaler or encoder is included in the model."
            )

    st.markdown("""
    <div class="footer">
        Built with Streamlit · Powered by K-Nearest Neighbors
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
