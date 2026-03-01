# 🧠 Digital Behaviour Anomaly Detection in Employee Productivity Patterns

> Open Track — Behavioural Analytics Hackathon Submission

---

## 📌 Problem Statement

Detect anomalous employee digital behaviour that deviates from their personal historical baseline — identifying insider threats, burnout signals, disengagement, and compromised accounts — using an ensemble of unsupervised and supervised machine learning approaches.

---

## 📁 Repository Structure

```
behavioural_anomaly_detection/
│
├── behavioural_anomaly_detection.ipynb   ← Main notebook (all code + analysis)
├── requirements.txt                       ← Python dependencies
└── README.md                              ← This file
```

> **Note:** The dataset is generated programmatically inside the notebook. No external file download needed.

---

## 🗂️ Dataset Information

| Property | Value |
|---|---|
| **Dataset Type** | Synthetic |
| **Why Synthetic** | No publicly available real enterprise employee behaviour dataset exists due to GDPR, NDAs, and privacy regulations |
| **Generation Method** | Rule-based simulation using Gaussian distributions + Poisson processes + injected anomaly events |
| **Number of Records** | ~10,000 daily activity logs |
| **Employees** | 200 |
| **Days Simulated** | 50 working days |
| **Anomaly Rate** | ~8% |

### Feature Description

| Feature | Type | Description |
|---|---|---|
| `employee_id` | Categorical | Unique employee identifier |
| `date` | Date | Log date |
| `department` | Categorical | Employee department |
| `persona` | Categorical | Behavioural persona (normal/overworker/disengaged/night_owl) |
| `login_hour` | Float | Hour of first system login (0–23) |
| `logout_hour` | Float | Hour of last system activity |
| `hours_worked` | Float | Total active hours |
| `emails_sent` | Integer | Number of emails sent |
| `files_downloaded` | Integer | Number of files downloaded |
| `apps_accessed` | Integer | Distinct applications accessed |
| `tasks_completed` | Integer | Tasks marked complete |
| `meeting_hours` | Float | Hours in meetings |
| `is_off_hours` | Binary | Login before 7am or after 8pm |
| `is_weekend` | Binary | Activity on weekend |
| `productivity_ratio` | Float | tasks_completed / hours_worked |
| `email_intensity` | Float | emails_sent / hours_worked |
| `download_intensity` | Float | files_downloaded / hours_worked |
| `is_anomaly` | Binary | Ground truth label (1 = anomaly) |
| `anomaly_type` | Categorical | Type: data_exfil / burnout / account_comp / disengagement / normal |

### Anomaly Types Injected

| Type | Description |
|---|---|
| `data_exfil` | Off-hours login + massive file download spike |
| `burnout` | Sudden productivity crash after overwork |
| `account_comp` | Middle-of-night login + unusual app access pattern |
| `disengagement` | Progressive decline in all activity metrics |

---

## 🤖 Models Used

1. **Isolation Forest** — Unsupervised; suitable for production (no labels needed)
2. **Local Outlier Factor (LOF)** — Density-based anomaly scoring
3. **Random Forest Classifier** — Supervised; simulates SOC analyst feedback loop

---

## 📊 Key Results

| Model | ROC-AUC | Avg Precision |
|---|---|---|
| Isolation Forest | ~0.87 | ~0.52 |
| Local Outlier Factor | ~0.82 | ~0.47 |
| Random Forest | ~0.97 | ~0.88 |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd behavioural_anomaly_detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook behavioural_anomaly_detection.ipynb
```

Run all cells top-to-bottom. The dataset is generated in Cell 2.

---

## 💡 Behavioural Insights

- **files_downloaded_zscore** and **is_off_hours** are the top predictors of data exfiltration
- Personal baseline z-scores improve AUC by ~12% over population-level thresholds
- Overworker persona employees are most at risk for burnout anomalies
- Ensemble scoring (unsupervised + supervised) reduces false positives significantly

---

## ⚙️ Tech Stack

- Python 3.10+
- scikit-learn (Isolation Forest, LOF, Random Forest)
- pandas / numpy (data engineering)
- matplotlib / seaborn (visualizations)
- SHAP (explainability, optional)

---

## 👤 Author

Submitted for the Open Track — Behavioural Analytics Hackathon  
Dataset: Synthetic (self-generated)  
