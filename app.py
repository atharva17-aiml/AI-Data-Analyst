import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
from dotenv import load_dotenv
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from database import init_db, login_user, register_user
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import smtplib
from email.message import EmailMessage
import base64

# ---------------- INIT ----------------
init_db()

if "user" not in st.session_state:
    st.session_state["user"] = None

def logout():
    st.session_state["user"] = None

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- PREMIUM UI ----------------
st.set_page_config(page_title="AI Data Analyst", layout="wide")

st.markdown("""
<style>
.stApp {background-color: #0E1117; color: white;}
.card {
    background: #161B22;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.stButton>button {
    background-color: #22c55e;
    color: white;
    border-radius: 8px;
    height: 40px;
    width: 100%;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
if st.session_state["user"] is None:

    st.title("🔐 Login")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(u, p):
                st.session_state["user"] = u
                st.success("Logged In ✅")
                st.rerun()
            else:
                st.error("Invalid Credentials")

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")

        if st.button("Register"):
            if register_user(u, p):
                st.success("Registered ✅")
            else:
                st.error("User exists")

    st.stop()

# ---------------- HEADER ----------------
col1, col2 = st.columns([8,1])

with col1:
    st.markdown("## 🚀 AI Data Analyst Dashboard")

with col2:
    if st.button("Logout"):
        logout()
        st.rerun()

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

# ---------------- FUNCTIONS ----------------
def ask_ai(prompt):
    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

def create_pdf(text):
    path = "AI_Report.pdf"
    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    story = []

    for p in text.split("\n"):
        story.append(Paragraph(p, styles["Normal"]))
        story.append(Spacer(1, 0.2 * cm))

    doc.build(story)
    return path

def send_email(to_email, pdf_path):
    msg = EmailMessage()
    msg["Subject"] = "AI Report"
    msg["From"] = os.getenv("EMAIL_USER")
    msg["To"] = to_email
    msg.set_content("Report Attached")

    with open(pdf_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename="report.pdf")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
        smtp.send_message(msg)

# ================= MAIN =================
if file is not None:

    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns

    # ---------------- PREVIEW ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- INSIGHTS ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚡ Auto Insights")

    if st.button("Generate Insights"):
        stats = df.describe().to_string()
        prompt = f"Analyze dataset {df.columns.tolist()} {stats}"
        st.write(ask_ai(prompt))

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- ML SUGGEST ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🤖 ML Model Suggestion")

    if st.button("Suggest ML Model"):
        st.write(ask_ai(f"Suggest ML model for {df.columns.tolist()}"))

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- TRAIN MODEL ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🚀 Train ML Model")

    target = st.selectbox("Target Column", df.columns)

    if st.button("Train Model"):
        try:
            X = pd.get_dummies(df.drop(columns=[target]))
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            score = r2_score(y_test, model.predict(X_test))

            st.success("Model Trained ✅")
            st.metric("R2 Score", round(score,3))

        except Exception as e:
            st.error(e)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- CORRELATION ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Correlation")

    if len(numeric_cols)>1:
        corr = df[numeric_cols].corr()
        st.dataframe(corr)

        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.matshow(corr)
        plt.colorbar(im)
        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- VISUALIZATION ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 Visualization")

    col = st.selectbox("Select Column", df.columns)

    fig, ax = plt.subplots()

    if df[col].dtype in ["int64","float64"]:
        ax.hist(df[col])
    else:
        df[col].value_counts().head(10).plot(kind="bar", ax=ax)

    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- ASK AI ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Ask AI")

    q = st.text_input("Ask anything")

    if st.button("Ask AI"):
        st.write(ask_ai(q))

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- REPORT ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📑 Report")

    if st.button("Generate Report"):
        report = ask_ai(f"Analyze dataset {df.columns.tolist()}")
        pdf_path = create_pdf(report)

        st.write(report)

        with open(pdf_path,"rb") as f:
            pdf_bytes = f.read()

        st.download_button("Download PDF", pdf_bytes, "report.pdf")

        base64_pdf = base64.b64encode(pdf_bytes).decode()
        st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400"></iframe>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- EMAIL ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📧 Send Email")

    email = st.text_input("Enter Email")

    if st.button("Send Email"):
        try:
            send_email(email, "AI_Report.pdf")
            st.success("Sent ✅")
        except Exception as e:
            st.error(e)

    st.markdown('</div>', unsafe_allow_html=True)
