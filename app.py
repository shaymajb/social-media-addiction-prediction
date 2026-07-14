import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# CONFIGURATION  PAGE 
st.set_page_config(
    page_title = "Digital Wellbeing Risk Assessment",
    layout     = "wide"
)

#  @st.cache_resource means Streamlit loads these files ONCE
#  and keeps them in memory. Without this, they reload on
#  every slider move = very slow.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model         = joblib.load(os.path.join(BASE_DIR, "models", "best_model.pkl"))
    scaler        = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "models", "feature_names.pkl"))
    explainer     = shap.TreeExplainer(model)
    return model, scaler, feature_names, explainer

model, scaler, feature_names, explainer = load_artifacts()

#  RISK LEVEL VISUAL CONFIG
RISK = {
    0: {"label": "LOW",      "emoji": "🟢", "color": "#1e7e34", "bg": "#d4edda", "text": "#155724"},
    1: {"label": "MODERATE", "emoji": "🟡", "color": "#d39e00", "bg": "#fff3cd", "text": "#856404"},
    2: {"label": "HIGH",     "emoji": "🟠", "color": "#c0620a", "bg": "#ffe8d1", "text": "#7d3c05"},
    3: {"label": "SEVERE",   "emoji": "🔴", "color": "#bd2130", "bg": "#f8d7da", "text": "#721c24"},
}
LABEL_NAMES = ["Low", "Moderate", "High", "Severe"]

# build model input from user selections

def build_input(age, gender, platform, daily_usage, posts_per_day,
                likes_received, comments_received, messages_sent,
                scroll_rate, emotional_state, productivity_loss,
                mental_health, fomo_score):

    # Start with ALL features = 0
    row = {f: 0 for f in feature_names}

    # Set numeric features directly
    row["Age"]                        = age
    row["Daily_Usage_Time_min"]       = daily_usage
    row["Posts_Per_Day"]              = posts_per_day
    row["Likes_Received_Daily"]       = likes_received
    row["Comments_Received_Daily"]    = comments_received
    row["Messages_Sent_Daily"]        = messages_sent
    row["Scroll_Rate_ppm"]            = scroll_rate
    row["Productivity_Loss_Score"]    = productivity_loss
    row["Mental_Health_Index"]        = mental_health
    row["FOMO_Score"]                 = fomo_score

    # Set categorical features:
    # Gender:   baseline = FEMALE  / columns: Gender_MALE, Gender_NON-BINARY
    # Platform: baseline = FACEBOOK / columns: Platform_INSTAGRAM, ...
    # Emotion:  baseline = ANXIOUS  / columns: Emotional_State_Post_Usage_BORED, ...

    gender_col = f"Gender_{gender}" 
    if gender_col in row:
        row[gender_col] = 1
    # if gender == "FEMALE" : no column to set : all Gender cols stay 0 

    platform_col = f"Platform_{platform}"
    if platform_col in row:
        row[platform_col] = 1
    # if platform == "FACEBOOK" : baseline : stays 0 

    emotion_col = f"Emotional_State_Post_Usage_{emotional_state}"
    if emotion_col in row:
        row[emotion_col] = 1
    # if emotion == "ANXIOUS" : baseline : stays 0 

    return pd.DataFrame([row])[feature_names]  # keep exact column order


#  PAGE HEADER
st.title(" Digital Wellbeing Risk Assessment")
st.markdown(
    "Enter your social media habits in the **sidebar** and click **Predict** "
    "to get your addiction risk level and a detailed explanation of what's driving it."
)
st.divider()

#  SIDEBAR — USER INPUTS
with st.sidebar:
    st.header("⚙️ Your Usage Habits")
    st.caption("Move the sliders to match your real habits.")

    st.subheader("👤 Profile")
    age    = st.slider("Age", 13, 60, 22)
    gender = st.selectbox("Gender", ["FEMALE", "MALE", "NON-BINARY"])

    st.subheader("📲 Social Media Usage")
    platform     = st.selectbox("Main Platform Used", [
        "FACEBOOK", "INSTAGRAM", "LINKEDIN", "PINTEREST",
        "SNAPCHAT", "TIKTOK", "TWITTER", "YOUTUBE"
    ])
    daily_usage  = st.slider("Daily Usage Time (minutes)", 5, 720, 120,
                             help="Total minutes per day on all social media")
    posts_per_day = st.slider("Posts Per Day", 0, 30, 2)
    scroll_rate  = st.slider("Scroll Rate (pages/min)", 5, 120, 40)

    st.subheader("💬 Engagement")
    likes_received    = st.slider("Likes Received Daily",    0, 500, 50)
    comments_received = st.slider("Comments Received Daily", 0, 100, 10)
    messages_sent     = st.slider("Messages Sent Daily",     0, 150, 20)

    st.subheader("🧠 Psychological Impact")
    emotional_state = st.selectbox("How do you feel AFTER using social media?", [
        "ANXIOUS", "BORED", "CALM", "DEPRESSED", "ENVIOUS",
        "HAPPY", "INSPIRED", "MOTIVATED", "NEUTRAL", "STRESSED"
    ])
    fomo_score        = st.slider("FOMO Score (1 = none, 10 = extreme)", 1, 10, 5,
                                  help="Fear of Missing Out when not checking social media")
    mental_health     = st.slider("Mental Health Index (20 = poor, 100 = excellent)", 20, 100, 65)
    productivity_loss = st.slider("Productivity Loss Score (1 = none, 10 = severe)", 1, 10, 4)

    st.divider()
    predict_btn = st.button("🔍 Predict My Risk Level", use_container_width=True, type="primary")


#  DEFAULT STATE — shown before the user clicks Predict
if not predict_btn:
    st.markdown("###  Fill in your habits in the sidebar and click **Predict**")
    st.markdown("The model will classify your risk into one of these 4 levels:")

    cols = st.columns(4)
    for col, (risk_class, cfg) in zip(cols, RISK.items()):
        with col:
            st.markdown(f"""
            <div style="
                background:{cfg['bg']};
                border:2px solid {cfg['color']};
                border-radius:12px;
                padding:20px;
                text-align:center;
            ">
                <p style="font-size:36px; margin:0;">{cfg['emoji']}</p>
                <p style="font-size:16px; font-weight:700;
                   color:{cfg['color']}; margin:6px 0 0;">{cfg['label']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    **How it works:**
    1. You fill in your habits using the sliders on the left
    2. The **Decision Tree** model (99.49% accuracy, 5-fold cross-validated) predicts your risk level
    3. **SHAP** explains exactly which habits are driving the prediction — no black box
    """)


#  PREDICTION STATE — shown after clicking Predict
else:
    # 1 — Build and scale input
    input_df     = build_input(age, gender, platform, daily_usage, posts_per_day,
                               likes_received, comments_received, messages_sent,
                               scroll_rate, emotional_state, productivity_loss,
                               mental_health, fomo_score)
    input_scaled = scaler.transform(input_df)

    # 2 — Predict class and probability
    prediction    = model.predict(input_scaled)[0]          # integer 0-3
    probabilities = model.predict_proba(input_scaled)[0]    # array of 4 probs
    confidence    = probabilities[prediction] * 100
    risk          = RISK[prediction]

    #  Result card + probability breakdown 
    col_result, col_shap = st.columns([1, 2])

    with col_result:
        # Main risk card
        st.markdown(f"""
        <div style="
            background:{risk['bg']};
            border:2px solid {risk['color']};
            border-radius:14px;
            padding:28px 20px;
            text-align:center;
            margin-bottom:16px;
        ">
            <p style="font-size:52px; margin:0;">{risk['emoji']}</p>
            <p style="font-size:12px; color:{risk['text']};
               text-transform:uppercase; letter-spacing:.1em;
               font-weight:600; margin:6px 0 2px;">Your Risk Level</p>
            <p style="font-size:34px; font-weight:800;
               color:{risk['color']}; margin:0;">{risk['label']}</p>
            <p style="font-size:13px; color:{risk['text']};
               margin-top:10px;">Model confidence: <strong>{confidence:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Probability for each class
        st.markdown("**Probability per class:**")
        for i, (lname, prob) in enumerate(zip(LABEL_NAMES, probabilities)):
            cfg = RISK[i]
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; "
                f"margin-bottom:4px;'>"
                f"<span>{cfg['emoji']} {lname}</span>"
                f"<span style='font-weight:700; color:{cfg['color']};'>"
                f"{prob*100:.1f}%</span></div>",
                unsafe_allow_html=True
            )
            st.progress(float(prob))

    # SHAP explanation chart 
    with col_shap:
        st.markdown("####  Why this prediction? (SHAP Explanation)")
        st.markdown(
            "Each bar shows how much one feature **pushed** the prediction toward "
            f"**{risk['label']}** *(red = increases risk)* or "
            "**away** from it *(blue = reduces risk)*."
        )

        # Compute SHAP values for this single input
        shap_values = explainer.shap_values(input_scaled)

        # Handle both SHAP formats:
        #   Old format : list of arrays : shap_values[class][sample]
        #   New format : 3D array      : shap_values[sample, feature, class]
        if isinstance(shap_values, list):
            shap_single = shap_values[prediction][0]
        else:
            shap_single = shap_values[0, :, prediction]

        # Build a sorted DataFrame of top 10 features by |SHAP value|
        shap_df = (
            pd.DataFrame({"feature": feature_names, "shap": shap_single})
            .assign(abs_shap=lambda d: d["shap"].abs())
            .sort_values("abs_shap", ascending=True)
            .tail(10)
        )

        bar_colors = [
            risk["color"] if v > 0 else "#185FA5"
            for v in shap_df["shap"]
        ]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(shap_df["feature"], shap_df["shap"],
                       color=bar_colors, height=0.6)
        ax.axvline(0, color="#333", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP value  ← reduces risk  |  increases risk →", fontsize=11)
        ax.set_title(f"Top 10 factors for: {risk['emoji']} {risk['label']}",
                     fontsize=13, fontweight="bold")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info(
            "💡 **How to read this:** Long red bar = this habit significantly "
            "increases your risk. Long blue bar = this habit is protecting you. "
            "Short bar = this feature had little impact on this prediction."
        )

    #  Input summary 
    st.divider()
    with st.expander(" View your input summary"):
        summary = {
            "Age": age, "Gender": gender, "Platform": platform,
            "Daily Usage (min)": daily_usage, "Posts/Day": posts_per_day,
            "Likes/Day": likes_received, "Comments/Day": comments_received,
            "Messages/Day": messages_sent, "Scroll Rate": scroll_rate,
            "Emotional State": emotional_state,
            "Productivity Loss": productivity_loss,
            "Mental Health Index": mental_health, "FOMO Score": fomo_score
        }
        st.dataframe(pd.DataFrame([summary]).T.rename(columns={0: "Your value"}),
                     use_container_width=True)


#  FOOTER
st.divider()
st.markdown(
    "<p style='text-align:center; font-size:12px; color:gray;'>"
    "Model: Decision Tree · 980 users · Accuracy: 99.49% · "
    "Cross-Val: 99.24% ± 0.74% · Built with Streamlit + SHAP + Scikit-learn"
    "</p>",
    unsafe_allow_html=True
)
