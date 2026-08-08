# AI Skill Gap Recommendation Dashboard — Version 3

## What Version 3 fixes

Version 3 is the professor-ready logic revision.

- Fully matched skills no longer receive unrelated training-course recommendations.
- Training recommendations are shown prominently only for Partially Matched or Missing / Low Alignment skills.
- Recommendation course names are limited to courses that actually exist in Dataset 01.
- Alignment now considers both:
  - Sentence-BERT semantic similarity (75%)
  - Curriculum skill-level adequacy (25%)
- Role-aware filtering remains active to reduce unrealistic combinations in the synthetic job dataset.
- Required Skills still originate from Dataset 02.
- Curriculum evidence and Recommended Training remain grounded in Dataset 01.
- The UI is closer to the professor's requested top-to-bottom flow.

## macOS Instructions

### Step 1
Download and extract:

`AI_Skill_Gap_Recommendation_Dashboard_V3.zip`

### Step 2
Move the extracted folder to Desktop.

The folder name should be:

`AI_Skill_Gap_Recommendation_Dashboard_V3`

### Step 3
Open Terminal and run:

```bash
cd ~/Desktop/AI_Skill_Gap_Recommendation_Dashboard_V3
```

### Step 4
Install requirements:

```bash
python3 -m pip install -r requirements.txt
```

If packages are already installed, Terminal will mostly show `Requirement already satisfied`.

### Step 5
Start the dashboard:

```bash
python3 -m streamlit run app.py
```

### Step 6
If Streamlit asks for an email:
leave it blank and press Enter.

The dashboard normally opens automatically in the browser.

If not, open:

`http://localhost:8501`

## Important
Do not edit either CSV file or `app.py`.
