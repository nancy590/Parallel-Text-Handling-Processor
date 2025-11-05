# Parallel-Text-Handling-Processor
Fake Review Sentiment Analysis Dashboard

An interactive Streamlit app for detecting fake or biased reviews using dual sentiment analysis techniques — TextBlob (traditional) and DistilBERT LLM (modern) — with secure authentication, data visualization, and automated email summaries.

🌟 Features

✅ Dual Sentiment Models

TextBlob: Fast polarity-based sentiment scoring

DistilBERT (LLM): Transformer-based contextual sentiment understanding

✅ User Authentication

Secure login system with email and password

Session-based authentication using Streamlit state

✅ Dashboard Modules

📁 Upload & Preview: Load CSV and inspect data

🧹 Clean Text: Automated text preprocessing

📊 TextBlob Sentiment: Traditional NLP sentiment scores

🤖 LLM Sentiment: Deep learning analysis with Hugging Face pipeline

🔠 Top Words: Frequency visualization of most common terms

⚖️ Comparison: Performance comparison between TextBlob and LLM

💾 Database & Download: Save results to SQLite or download CSV

📧 Email Summary: Send sentiment report directly to user email

✅ Visual Analytics

Bar and Pie charts for sentiment distribution

Metrics comparison with execution time analysis

✅ Reporting

Download cleaned CSV or send an automatic email summary

🚀 Installation
Prerequisites

Python 3.8+

pip package manager

(Optional) CUDA-enabled GPU for faster LLM inference

Step 1: Clone the Repository
git clone <your-repo-url>
cd fake-review-dashboard

Step 2: Install Dependencies
pip install streamlit pandas numpy matplotlib textblob transformers torch nltk

Step 3: Download NLTK Stopwords
python -c "import nltk; nltk.download('stopwords')"

📂 Project Structure
fake-review-dashboard/
│
├── app.py                     # Main Streamlit application
├── reviews_ui.db              # SQLite database (auto-created)
├── requirements.txt            # Dependency file
├── README.md                   # Documentation (this file)
└── assets/                     # (Optional) icons, logos, or supporting files

🎯 Usage
Run the Application
streamlit run app.py


The dashboard will open automatically in your browser at
👉 http://localhost:8501

🧭 Workflow
1️⃣ Authentication

Enter your email and password

Login securely to access the dashboard

2️⃣ Upload & Preview

Upload a CSV file containing text or review columns

System auto-detects the text column

Optional sample size preview (default: 1000 rows)

3️⃣ TextBlob Sentiment

Calculates polarity and classifies reviews as Positive, Negative, or Neutral

Displays bar and pie charts

4️⃣ LLM Sentiment (DistilBERT)

Uses Hugging Face DistilBERT sentiment-analysis pipeline

GPU acceleration if available

Progress bar for real-time feedback

5️⃣ Top Words

Displays 20 most common words with horizontal bar chart visualization

6️⃣ Comparison

Compare execution time of TextBlob vs LLM

Visualize time difference and performance metrics

7️⃣ Database & Download

Save processed data into SQLite (reviews_ui.db)

Download analyzed CSV

8️⃣ Email Summary

Sends summary email to logged-in user

Includes counts, execution times, and performance details

📊 Key Metrics
Metric	Description
Polarity	Range from -1 (negative) to +1 (positive)
Label Distribution	Count of positive, negative, and neutral sentiments
Execution Time	Total analysis duration for each model
Top Words	Most frequent words after preprocessing
⚙️ Configuration
Email Settings

Edit inside app.py:

sender_email = "your-email@gmail.com"
sender_password = "your-app-password"


Gmail Setup:

Enable 2FA (Two-Factor Authentication)

Generate an App Password under Google Account → Security → App Passwords

Replace credentials in the code

🧠 Technologies Used
Category	Tools
Frontend	Streamlit
NLP Models	TextBlob, DistilBERT (Hugging Face)
Database	SQLite
Libraries	Pandas, NLTK, Matplotlib, Transformers
Email Service	smtplib (Gmail SMTP)
🧩 Troubleshooting
Issue	Fix
TextBlob ImportError	pip install textblob & python -m textblob.download_corpora
CUDA not available	Use CPU or install CUDA-enabled PyTorch
Model download fails	Check internet connection or Hugging Face cache
Database locked	Close all Streamlit sessions or delete .db and restart
Email fails	Ensure valid Gmail App Password and “Allow less secure apps” setting if needed
