from __future__ import annotations

import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from train_model import METRICS_PATH, MODEL_PATH, load_dataset, train_and_save


st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="mail",
    layout="wide",
    initial_sidebar_state="collapsed",
)


SAMPLE_MESSAGES = {
    "Prize scam": "Congratulations! You have won a free iPhone. Claim now by clicking http://tinyurl.com/prize and send your bank details.",
    "Urgent account alert": "Your account has been suspended. Verify your password immediately to avoid permanent closure.",
    "Normal work email": "Hi Riya, can we move tomorrow's project review to 3 PM? I have attached the updated notes.",
    "Friendly update": "Hey, I reached home safely. Call me when you are free.",
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #14202a;
            --muted: #65717c;
            --line: #d9e1e8;
            --paper: #f7f9fb;
            --accent: #e85d3f;
            --green: #16855b;
            --gold: #c58b1a;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(232, 93, 63, .10), transparent 28rem),
                linear-gradient(135deg, #f9fbfd 0%, #eef4f7 100%);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        .hero {
            margin-bottom: 1.2rem;
            padding: 1.85rem 2rem 2rem;
            min-height: 245px;
            background: #17212b;
            color: white;
            border: 1px solid rgba(20,32,42,.09);
            box-shadow: 0 22px 48px rgba(29, 44, 58, .10);
            border-radius: 8px;
            overflow: hidden;
            position: relative;
        }

        .hero:after {
            content: "";
            position: absolute;
            inset: auto -4rem -5rem auto;
            width: 13rem;
            height: 13rem;
            border-radius: 999px;
            background: rgba(232,93,63,.34);
        }

        .hero-main {
            position: relative;
            z-index: 1;
        }

        .eyebrow {
            color: var(--accent);
            font-size: .8rem;
            letter-spacing: .11rem;
            text-transform: uppercase;
            font-weight: 800;
        }

        h1 {
            font-size: clamp(2.2rem, 5vw, 4.5rem);
            line-height: .94;
            letter-spacing: 0;
            margin: 0 0 1rem;
            color: white;
        }

        .subtitle {
            color: rgba(255,255,255,.76);
            font-size: 1.05rem;
            max-width: 720px;
            line-height: 1.65;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .8rem;
            margin: 1rem 0;
        }

        .metric-card {
            border: 1px solid var(--line);
            background: white;
            border-radius: 8px;
            padding: 1rem;
        }

        .metric-label {
            color: var(--muted);
            font-size: .78rem;
            text-transform: uppercase;
            font-weight: 800;
            letter-spacing: .07rem;
        }

        .metric-value {
            font-size: 1.75rem;
            font-weight: 900;
            color: var(--ink);
            margin-top: .2rem;
        }

        .result {
            border-radius: 8px;
            padding: 1.1rem;
            border: 1px solid var(--line);
            background: white;
        }

        .result.spam {
            border-color: rgba(232,93,63,.35);
            background: #fff5f1;
        }

        .result.ham {
            border-color: rgba(22,133,91,.28);
            background: #f1fbf6;
        }

        .result.review {
            border-color: rgba(197,139,26,.36);
            background: #fff8e8;
        }

        .verdict {
            font-size: 1.75rem;
            font-weight: 900;
            margin: .1rem 0 .35rem;
        }

        .confidence-wrap {
            margin: 1rem 0 1.15rem;
        }

        .confidence-top {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            color: var(--muted);
            font-weight: 800;
            font-size: .82rem;
            text-transform: uppercase;
            letter-spacing: .06rem;
            margin-bottom: .45rem;
        }

        .confidence-track {
            height: .7rem;
            border-radius: 999px;
            background: #dde6ed;
            overflow: hidden;
        }

        .confidence-fill {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #e85d3f, #f0aa3d);
        }

        .risk-pill {
            display: inline-flex;
            align-items: center;
            min-height: 2rem;
            border: 1px solid var(--line);
            background: #fff;
            border-radius: 999px;
            padding: .35rem .75rem;
            margin: .2rem .25rem .2rem 0;
            color: var(--ink);
            font-weight: 700;
            font-size: .86rem;
        }

        .stTextArea textarea {
            border-radius: 8px;
            border-color: var(--line);
            min-height: 190px;
            font-size: 1rem;
        }

        .stButton > button {
            border-radius: 8px;
            border: 0;
            background: var(--ink);
            color: #fff;
            font-weight: 800;
            min-height: 2.8rem;
        }

        .stButton > button:hover {
            background: var(--accent);
            color: #fff;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 8px;
            border-color: rgba(20,32,42,.10);
            background: rgba(255,255,255,.70);
            box-shadow: 0 18px 42px rgba(29, 44, 58, .08);
        }

        @media (max-width: 820px) {
            .metric-row {
                grid-template-columns: 1fr;
            }
            .hero {
                padding: 1.25rem;
                min-height: auto;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_model():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        train_and_save()
    return joblib.load(MODEL_PATH), joblib.load(METRICS_PATH)


def detect_risk_signals(message: str) -> list[str]:
    checks = {
        "money or prize language": r"\b(win|won|prize|cash|free|reward|bonus|lottery)\b",
        "urgent pressure": r"\b(urgent|immediately|limited|expire|act now|final notice)\b",
        "link present": r"(https?://|www\.|bit\.ly|tinyurl)",
        "phone or short code": r"\b\d{5,}\b",
        "credential request": r"\b(password|verify|account|bank|login|pin)\b",
        "all-caps emphasis": r"\b[A-Z]{4,}\b",
    }
    return [
        label
        for label, pattern in checks.items()
        if re.search(pattern, message, flags=re.IGNORECASE)
    ]


def probability_for(model, message: str) -> tuple[str, float, float]:
    classes = list(model.classes_)
    probabilities = model.predict_proba([message])[0]
    spam_probability = float(probabilities[classes.index("spam")])
    ham_probability = float(probabilities[classes.index("ham")])
    prediction = "spam" if spam_probability >= ham_probability else "ham"
    return prediction, spam_probability, ham_probability


def render_metrics(metrics: dict) -> None:
    accuracy = metrics["accuracy"] * 100
    spam_rate = metrics["spam_count"] / metrics["dataset_size"] * 100
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Model accuracy</div>
                <div class="metric-value">{accuracy:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Training messages</div>
                <div class="metric-value">{metrics["dataset_size"]:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Spam examples</div>
                <div class="metric-value">{metrics["spam_count"]:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Spam share</div>
                <div class="metric-value">{spam_rate:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(model, message: str) -> None:
    if not message.strip():
        st.info("Choose or type a message, then click Scan Message to classify it.")
        return

    prediction, spam_probability, ham_probability = probability_for(model, message)
    confidence = max(spam_probability, ham_probability)
    signals = detect_risk_signals(message)
    needs_review = confidence < 0.62 and len(signals) >= 2

    if needs_review:
        label = "Needs manual review"
        detail = "The model is close to a split decision, and the message contains suspicious signals."
        css_class = "review"
    elif prediction == "spam":
        label = "Spam detected"
        detail = "This message has patterns commonly found in spam or phishing."
        css_class = "spam"
    else:
        label = "Looks legitimate"
        detail = "The model sees this as a normal message, but always review links and payment requests."
        css_class = "ham"

    st.markdown(
        f"""
        <div class="result {css_class}">
            <div class="metric-label">Prediction</div>
            <div class="verdict">{label}</div>
            <div>{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="confidence-wrap">
            <div class="confidence-top">
                <span>Confidence</span>
                <span>{confidence:.1%}</span>
            </div>
            <div class="confidence-track">
                <div class="confidence-fill" style="width: {confidence * 100:.1f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c1.metric("Spam probability", f"{spam_probability:.1%}")
    c2.metric("Ham probability", f"{ham_probability:.1%}")

    if signals:
        chips = "".join(f'<span class="risk-pill">{signal}</span>' for signal in signals)
        st.markdown("**Risk signals found**")
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.markdown("**Risk signals found**")
        st.caption("No obvious rule-based warning signs were found.")


def main() -> None:
    inject_styles()
    model, metrics = get_model()

    st.markdown(
        """
        <section class="hero">
            <div class="hero-main">
                <h1>Email Spam Detection</h1>
                <p class="subtitle">
                    A trained spam classifier that uses TF-IDF text features and a Naive Bayes model
                    to separate unwanted promotional, scam, and phishing-style messages from genuine communication.
                </p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    render_metrics(metrics)

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("Message Scanner")
            sample_name = st.selectbox("Try a sample message", ["Custom message"] + list(SAMPLE_MESSAGES))
            default_message = "" if sample_name == "Custom message" else SAMPLE_MESSAGES[sample_name]
            message = st.text_area(
                "Email or SMS content",
                value=default_message,
                placeholder="Paste an email, SMS, or suspicious message here...",
                height=210,
            )
            scan = st.button("Scan Message", use_container_width=True)

    with right:
        with st.container(border=True):
            st.subheader("Classification Result")
            render_result(model, message if scan else "")

    with st.expander("Model evaluation details", expanded=False):
        report = metrics["report"]
        summary = pd.DataFrame(
            [
                {
                    "Class": label.title(),
                    "Precision": report[label]["precision"],
                    "Recall": report[label]["recall"],
                    "F1-score": report[label]["f1-score"],
                }
                for label in ["ham", "spam"]
            ]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

        matrix = pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Actual ham", "Actual spam"],
            columns=["Predicted ham", "Predicted spam"],
        )
        st.dataframe(matrix, use_container_width=True)
        st.caption(f"Dataset source: {Path(metrics['source']).name}")


if __name__ == "__main__":
    main()
