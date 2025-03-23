# CareerFlow - Job Scraping & JD Extraction Pipeline

CareerFlow is a Python-based tool that scrapes job offers from LinkedIn and extracts structured information from job descriptions using LangChain and LLMs (e.g., Gemini). It detects the job description language, translates if needed, and outputs structured fields like title, skills, responsibilities, etc.

---

## 🚀 Features

- 🔍 Scrapes job offers from LinkedIn
- 🤖 Extracts structured job info using LangChain prompt templates
- 💾 Caches results using TinyDB (resilient to crashes or restarts)
- 📦 Outputs results to a CSV file

## ⚙️ Installation

1. Clone the repo:
```bash
git clone https://github.com/your-username/careerflow.git
cd careerflow
```
2. Create and activate a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
3. Set up environment variables:

Create a .env file in the root folder:
```
GOOGLE_API_KEY=your_gemini_or_google_key
```
## 🧪 Usage

Run the pipeline from the command line:
```bash
python main.py \
  --title "Machine Learning Engineer" \
  --location "Paris" \
  --max-pages 2 \
  --prompt-dir extractor/prompts \
  --output scraped_jobs.csv
  --cache-path cache/job_cache.json
```
