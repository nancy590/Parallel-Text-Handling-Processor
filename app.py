import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from textblob import TextBlob
from collections import Counter
from transformers import pipeline
import re
import nltk
import time
from nltk.corpus import stopwords
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ===============================
# Setup
# ===============================
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white"
})

# ===============================
# Streamlit Config
# ===============================
st.set_page_config(page_title="💬 Fake Review Dashboard", layout="wide", page_icon="💻")

# Global CSS Styling (Responsive + Dark Mode)
st.markdown("""
<style>
    .block-container {
        max-width: 1100px;
        margin: auto;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stApp {
        background-color: #1e1e2f;
        color: #f5f5f5;
        zoom: 0.9;
    }
    @media (max-width: 1200px) {
        .stApp { zoom: 0.85; }
    }
    @media (max-width: 992px) {
        .stApp { zoom: 0.8; }
    }
    @media (max-width: 768px) {
        .stApp { zoom: 0.75; }
    }
    h1, h2, h3 {
        color: #ffb703;
        font-weight: bold;
    }
    .stButton>button {
        background-color:#ffb703;
        color:black;
        font-weight:bold;
        border-radius:8px;
        border:none;
    }
    .stMetric label {
        color:#ffb703 !important;
        font-size:16px !important;
    }
    .stMetric span {
        color:white !important;
        font-size:18px !important;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# Login System
# ===============================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

def show_login():
    st.markdown("""
        <style>
            .login-box { background-color: #0f1724; padding: 24px; border-radius: 12px; color: #f5f5f5; }
            .login-title { color:#ffb703; font-weight:700; font-size:22px; }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("<div class='login-title'>🔐 Sign in to Fake Review Dashboard</div>", unsafe_allow_html=True)
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
        if submitted:
            if email and password:
                st.session_state.authenticated = True
                st.session_state.user_email = email
                st.success(f"Welcome, {email}!")
                st.rerun()
            else:
                st.error("Please enter both email and password.")
    st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.authenticated:
    show_login()
    st.stop()

# ===============================
# Header
# ===============================
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.markdown("### 💬 Fake Review Sentiment Analysis Dashboard")
with col2:
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user_email = ""
        st.rerun()

st.markdown("**Interactive dashboard to visualize review sentiments and common words**")

# ===============================
# Sidebar Navigation
# ===============================
st.sidebar.title("📂 Navigation")
options = st.sidebar.radio("Go to:", [
    "Upload & Preview",
    "TextBlob Sentiment",
    "LLM Sentiment",
    "Top Words",
    "Comparison",
    "Database & Download",
    "Send Email Summary"
])

# ===============================
# Upload Section
# ===============================
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.success("✅ File uploaded successfully!")
    st.write("Columns in CSV:", df.columns.tolist())

    possible_cols = [col for col in df.columns if 'text' in col.lower() or 'review' in col.lower()]
    if not possible_cols:
        st.error("❌ No text/review column found! Please check your CSV.")
        st.stop()
    else:
        text_col = possible_cols[0]
        st.info(f"Using column '{text_col}' as the text column for analysis.")
        df["text_"] = df[text_col].astype(str)

    SAMPLE_LIMIT = st.sidebar.number_input("Preview sample size (rows)", min_value=100, max_value=50000, value=1000, step=100)
    df_sample = df.head(int(SAMPLE_LIMIT)).copy()

    @st.cache_data
    def fast_clean_text_series(series):
        cleaned = []
        for text in series:
            t = str(text).lower()
            t = re.sub(r'[^a-z\s]', ' ', t)
            tokens = [w for w in t.split() if w and w not in stop_words]
            cleaned.append(" ".join(tokens))
        return cleaned

    with st.spinner("Cleaning text for preview... ⚙️"):
        df_sample["cleaned_text"] = fast_clean_text_series(df_sample["text_"])
    st.success("✅ Text cleaned for preview")

    # ===============================
    # Upload & Preview
    # ===============================
    if options == "Upload & Preview":
        st.subheader("📋 Data Preview")
        st.markdown(f"""
        **✅ File loaded successfully!**
        - Total rows: **{len(df)}**
        - Columns: **{len(df.columns)}**
        - Sample used for analysis: **{len(df_sample)}**
        """)
        preview_rows = st.slider("Select number of rows to preview:", 5, min(100, len(df_sample)), 10)
        st.dataframe(df_sample.head(preview_rows), use_container_width=True, height=500)
        st.caption("📌 Only showing preview — full dataset loaded in memory.")

    # ===============================
    # TextBlob Sentiment
    # ===============================
    if options == "TextBlob Sentiment":
        st.subheader("📊 TextBlob Sentiment Analysis")
        start = time.time()

        @st.cache_data
        def get_textblob_sentiments(texts):
            scores = [TextBlob(t).sentiment.polarity for t in texts]
            labels = ["Positive" if s > 0 else ("Negative" if s < 0 else "Neutral") for s in scores]
            return scores, labels

        df_sample["sentiment_score"], df_sample["sentiment_label"] = get_textblob_sentiments(df_sample["cleaned_text"])
        st.session_state["textblob_time"] = time.time() - start

        pos = (df_sample["sentiment_label"] == "Positive").sum()
        neg = (df_sample["sentiment_label"] == "Negative").sum()
        neu = (df_sample["sentiment_label"] == "Neutral").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Positive Reviews", pos, f"{pos/len(df_sample)*100:.1f}%")
        col2.metric("Negative Reviews", neg, f"{neg/len(df_sample)*100:.1f}%")
        col3.metric("Neutral Reviews", neu, f"{neu/len(df_sample)*100:.1f}%")

        sentiment_counts = df_sample["sentiment_label"].value_counts()
        fig, ax = plt.subplots(figsize=(4,3), facecolor="#1e1e2f")
        sentiment_counts.plot(kind='bar', ax=ax, color=['#4CAF50', '#F44336', '#FFC107'])
        ax.set_facecolor("#1e1e2f")
        plt.title("Sentiment Distribution (TextBlob)", color='white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(3.5,3.5), facecolor="#1e1e2f")
        ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=['#4CAF50', '#F44336', '#FFC107'], startangle=90, textprops={'color':'white'})
        plt.title("Sentiment Proportion (TextBlob)", color='white')
        st.pyplot(fig2, use_container_width=True)
        st.info(f"⏱️ Execution Time: {st.session_state['textblob_time']:.2f} seconds")

    # ===============================
    # LLM Sentiment
    # ===============================
    if options == "LLM Sentiment":
        st.subheader("🤖 LLM Sentiment Analysis (DistilBERT)")
        max_allowed = min(300, len(df_sample))
        sample_size = st.slider("Reviews to analyze:", 10, max_allowed, min(100, max_allowed))
        if st.button("Run LLM Sentiment"):
            start = time.time()
            with st.spinner("Running LLM Sentiment..."):
                sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
                df_subset = df_sample.head(sample_size).copy()
                results = []
                for i, text in enumerate(df_subset["cleaned_text"]):
                    label = sentiment_pipeline(text[:512])[0]["label"] if text.strip() else "NEUTRAL"
                    results.append(label)
                    st.progress((i+1)/sample_size)
                df_subset["llm_sentiment"] = results
            st.session_state["llm_time"] = time.time() - start
            st.success(f"✅ Completed in {st.session_state['llm_time']:.2f} seconds")
            st.dataframe(df_subset[["cleaned_text", "llm_sentiment"]].head(), use_container_width=True)

    # ===============================
    # Top Words
    # ===============================
    if options == "Top Words":
        st.subheader("🔠 Top 20 Common Words")
        all_tokens = " ".join(df_sample["cleaned_text"]).split()
        top_df = pd.DataFrame(Counter(all_tokens).most_common(20), columns=["Word", "Frequency"])
        st.dataframe(top_df, use_container_width=True)
        fig, ax = plt.subplots(figsize=(5,3), facecolor="#1e1e2f")
        ax.barh(top_df["Word"], top_df["Frequency"], color="#ffb703")
        ax.set_facecolor("#1e1e2f")
        plt.title("Top 20 Words", color='white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # ===============================
    # Comparison
    # ===============================
    if options == "Comparison":
        st.header("⚖️ TextBlob vs LLM Sentiment Comparison")
        if "textblob_time" not in st.session_state or "llm_time" not in st.session_state:
            st.warning("⚠️ Please run both TextBlob and LLM Sentiment first.")
        else:
            tb_time = st.session_state["textblob_time"]
            llm_time = st.session_state["llm_time"]
            col1, col2 = st.columns(2)
            col1.metric("TextBlob Execution Time", f"{tb_time:.2f} s")
            col2.metric("LLM Execution Time", f"{llm_time:.2f} s")
            faster = "TextBlob" if tb_time < llm_time else "LLM"
            diff = abs(tb_time - llm_time)
            st.info(f"✅ {faster} is faster by {diff:.2f} seconds.")
            fig, ax = plt.subplots(figsize=(4,3), facecolor="#1e1e2f")
            ax.bar(["TextBlob", "LLM"], [tb_time, llm_time], color=["#ffb703", "#5bc0de"])
            ax.set_ylabel("Time (seconds)")
            plt.title("Execution Time Comparison", color='white')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    # ===============================
    # Database & Download
    # ===============================
    if options == "Database & Download":
        st.subheader("💾 Save & Download")
        if st.button("Save Sampled Results to SQLite DB"):
            conn = sqlite3.connect("reviews_ui.db")
            df_sample.to_sql("reviews_sentiment_ui", conn, if_exists="replace", index=False)
            conn.close()
            st.success("💾 Data saved to SQLite (reviews_ui.db)")

        csv = df_sample.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv, file_name="reviews_sentiment_sample_results.csv", mime="text/csv")

    # ===============================
    # Send Email Summary
    # ===============================
    if options == "Send Email Summary":
        st.subheader("📧 Send Summary Email")
        st.info("Send a dataset summary (sentiment results and performance) to your registered email.")
        sender_email = st.text_input("Sender Email (Gmail only)", value="kandikatlanancygrace@gmail.com")
        sender_password = st.text_input("App Password (from Google App Passwords)", type="password")
        if st.button("📨 Send Email Summary"):
            try:
                total_reviews = len(df)
                sample_reviews = len(df_sample)
                pos = (df_sample["sentiment_label"] == "Positive").sum() if "sentiment_label" in df_sample else 0
                neg = (df_sample["sentiment_label"] == "Negative").sum() if "sentiment_label" in df_sample else 0
                neu = (df_sample["sentiment_label"] == "Neutral").sum() if "sentiment_label" in df_sample else 0
                tb_time = st.session_state.get("textblob_time", 0)
                llm_time = st.session_state.get("llm_time", 0)
                body = f"""
                Hello {st.session_state.user_email},

                Here is your Fake Review Analysis Summary:

                ✅ Total Reviews: {total_reviews}
                ✅ Sample Used: {sample_reviews}

                📊 Sentiment Summary (TextBlob):
                Positive: {pos}
                Negative: {neg}
                Neutral: {neu}

                ⚡ Execution Times:
                TextBlob: {tb_time:.2f} seconds
                LLM Model: {llm_time:.2f} seconds

                💾 Database saved as: reviews_ui.db
                📂 CSV file available in dashboard download

                Thank you for using the Fake Review Dashboard!
                """
                msg = MIMEMultipart()
                msg["From"] = sender_email
                msg["To"] = st.session_state.user_email
                msg["Subject"] = "Fake Review Analysis Summary"
                msg.attach(MIMEText(body, "plain"))
                with smtplib.SMTP("smtp.gmail.com", 587) as server:
                    server.starttls()
                    server.login(sender_email, sender_password)
                    server.send_message(msg)
                st.success(f"✅ Email summary sent successfully to {st.session_state.user_email}!")
            except Exception as e:
                st.error(f"❌ Failed to send email: {e}")
else:
    st.info("👆 Please upload a CSV file to begin.")




# dzexfggsgiwvdxev