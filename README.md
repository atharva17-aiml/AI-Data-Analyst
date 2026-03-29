# 🚀 AI Data Analyst Dashboard

An end-to-end **AI-powered data analysis web application** built using **Streamlit**, enabling users to upload datasets, generate insights, train machine learning models, and create downloadable reports — all in one place.

---

## 📌 Features

### 🔐 Authentication

* User Registration & Login system
* Secure session handling

### 📊 Data Analysis

* Upload CSV or Excel files
* Dataset preview with interactive table
* Automatic statistical summary

### ⚡ AI-Powered Insights

* Generate insights using LLM (Groq API)
* Ask custom questions about your dataset

### 🤖 Machine Learning

* Auto preprocessing:

  * Missing value handling
  * Datetime feature extraction (year, month, day)
  * Categorical encoding
* Train ML model (Random Forest)
* Evaluate using **R² Score**

### 📈 Visualization

* Histogram for numerical columns
* Bar charts for categorical data
* Correlation heatmap

### 📑 Report Generation

* Generate AI-based report
* Export as PDF
* Preview report inside app

### 📧 Email Integration

* Send generated report via email

---

## 🛠️ Tech Stack

* **Frontend & App**: Streamlit
* **Backend**: Python
* **Machine Learning**: Scikit-learn
* **Data Processing**: Pandas, NumPy
* **Visualization**: Matplotlib
* **AI Integration**: Groq (LLM API)
* **PDF Generation**: ReportLab
* **Authentication DB**: SQLite (via custom module)

---

## 📂 Project Structure

```
├── app.py
├── database.py
├── .env
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/ai-data-analyst-dashboard.git
cd ai-data-analyst-dashboard
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Setup Environment Variables

Create a `.env` file and add:

```
GROQ_API_KEY=your_groq_api_key
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_app_password
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. User uploads dataset (CSV/Excel)
2. App preprocesses data:

   * Converts datetime columns
   * Handles missing values
   * Encodes categorical features
3. AI generates insights using LLM
4. ML model is trained and evaluated
5. Report is generated and optionally emailed

---

## 📸 Screenshots (Optional)

*Add screenshots of your app UI here*

---

## 🚀 Future Improvements

* Auto detect **classification vs regression**
* Try multiple ML models and select best
* Interactive charts using Plotly
* Model download (.pkl)
* Deploy on Streamlit Cloud

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repo and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Atharva Unde**

* GitHub: https://github.com/your-username
* LinkedIn: https://linkedin.com/in/your-profile

---

⭐ If you like this project, give it a star!
