## Sentiment Analysis
# SentiScope — Sentiment Analysis with Social Media Data
## Complete ML Project | Beginner → Intermediate Level

---

## 📁 Project Structure

```
sentiment-analyzer/
├── backend/
│   ├── app.py                      ← Flask REST API
│   ├── requirements.txt            ← Python dependencies
│   ├── render.yaml                 ← Render.com deployment config
│   ├── model/
│   │   ├── train.py                ← Train the ML model
│   │   ├── predict.py              ← Prediction logic
│   │   └── sentiment_model.pkl     ← Saved model (auto-generated)
│   └── data/
│       ├── generate_data.py        ← Create synthetic dataset
│       └── social_media_data.csv   ← Generated dataset (auto-generated)
└── frontend/
    └── index.html                  ← Complete single-page UI
```

---

## 🧠 ML Pipeline Explained

```
Raw Text Input
    │
    ▼
[1] TEXT CLEANING (regex)
    • Lowercase
    • Remove URLs, @mentions, hashtags
    • Strip special characters
    │
    ▼
[2] TF-IDF VECTORIZATION
    • Converts words → numbers
    • TF  = how often a word appears in THIS post
    • IDF = how rare the word is across ALL posts
    • High TF-IDF = distinctive word
    • ngram_range=(1,2) captures "not good" as one feature
    │
    ▼
[3] LOGISTIC REGRESSION
    • Learns weights for each TF-IDF feature
    • Softmax converts raw scores → probabilities
    • Probabilities sum to 1.0 across all 3 classes
    │
    ▼
Prediction: positive / negative / neutral
Confidence: probability of the winning class
```

---

## 🚀 LOCAL SETUP (Step by Step)

### Prerequisites
- Python 3.9 or higher
- pip (comes with Python)

### Step 1 — Clone / Download the project
```bash
# Option A: Clone from GitHub
git clone https://github.com/your-username/sentiment-analyzer.git
cd sentiment-analyzer

# Option B: Download ZIP and unzip
```

### Step 2 — Set up Python environment
```bash
cd backend

# Create a virtual environment (isolates project dependencies)
python -m venv venv

# Activate it:
#   macOS/Linux:
source venv/bin/activate
#   Windows:
venv\Scripts\activate

# Your terminal prompt should show (venv) when active
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```
This installs: Flask, Flask-CORS, scikit-learn, numpy, gunicorn

### Step 4 — Generate the dataset
```bash
python data/generate_data.py
```
Creates: `data/social_media_data.csv` (600 rows: 200 per class)

### Step 5 — Train the model
```bash
python model/train.py
```
You should see output like:
```
✅ ACCURACY: 0.9417 (94.17%)

📋 CLASSIFICATION REPORT:
              precision    recall  f1-score   support
    negative       0.94      0.95      0.95        40
     neutral       0.93      0.92      0.93        40
    positive       0.96      0.95      0.96        40

✅ Model saved → model/sentiment_model.pkl
```

### Step 6 — Start the API server
```bash
python app.py
```
API is running at: http://localhost:5000

### Step 7 — Open the Frontend
Open `frontend/index.html` directly in your browser.
**Important**: Make sure `API_BASE_URL = "http://localhost:5000"` in the JS.

---

## ☁️ FREE DEPLOYMENT (Production)

### Backend → Render.com (Free Tier)

1. Push your `backend/` folder to a **GitHub repository**

2. Sign up at [render.com](https://render.com) (free, no credit card)

3. Click **New → Web Service**

4. Connect your GitHub repo

5. Render auto-detects `render.yaml` and configures everything

6. Wait ~3 minutes for the build to complete

7. Your API URL will be: `https://your-service-name.onrender.com`

> ⚠️ **Free tier note**: Render's free tier spins down after 15 minutes of inactivity.
> First request after sleep takes ~30 seconds to wake up. This is normal.

---

### Frontend → GitHub Pages (Free, always on)

1. Create a new GitHub repo (e.g., `sentiscope-frontend`)

2. Add `frontend/index.html` to the repo

3. **Edit the API URL** in index.html:
   ```javascript
   // Change this line:
   const API_BASE_URL = "http://localhost:5000";
   // To your Render URL:
   const API_BASE_URL = "https://your-service.onrender.com";
   ```

4. Go to repo **Settings → Pages**

5. Set Source: `Deploy from a branch → main → / (root)`

6. Your site is live at: `https://your-username.github.io/sentiscope-frontend`

---

## 🔌 API Reference

### POST `/predict`
```json
// Request
{ "text": "I absolutely love this product!" }

// Response
{
  "status": "success",
  "data": {
    "sentiment": "positive",
    "confidence": 0.8932,
    "probabilities": {
      "positive": 0.8932,
      "neutral": 0.0712,
      "negative": 0.0356
    },
    "emoji": "😊",
    "color": "#22c55e",
    "inference_time_ms": 2.4,
    "cleaned_text": "i absolutely love this product"
  }
}
```

### POST `/predict-batch`
```json
// Request
{ "texts": ["post 1", "post 2", "post 3"] }

// Response includes "results" array + "summary" counts
```

### GET `/health`
Returns server status and uptime.

### GET `/stats`
Returns model info and request counts.

---

## 📊 Key ML Concepts in This Project

| Concept | Where Used | Why It Matters |
|---|---|---|
| Train/Test Split | train.py | Honestly evaluate model on unseen data |
| Class Balance | generate_data.py | Prevents model bias toward majority class |
| TF-IDF | train.py (Pipeline) | Efficient text-to-number conversion |
| Bigrams | TF-IDF config | Captures "not good" as a feature |
| Regularization (C) | Logistic Regression | Prevents overfitting |
| Softmax Probabilities | predict.py | Get confidence per class |
| Pickle Serialization | train.py / predict.py | Save and reload trained models |
| REST API | app.py | Standard ML model serving pattern |
| CORS | app.py | Allow browser to call the API |
| Confusion Matrix | train.py | See where the model makes mistakes |

---

## 🛠 Extending This Project

Once comfortable, try these improvements:

1. **Better preprocessing**: Add stemming (PorterStemmer) or lemmatization (spaCy)
2. **Better features**: Add emoji sentiment scores as extra features
3. **Better model**: Try `RandomForestClassifier` or `GradientBoostingClassifier`
4. **Real data**: Scrape Reddit posts using the PRAW library (free API)
5. **Deep learning**: Replace TF-IDF + LR with a BERT model (HuggingFace)
6. **Database**: Store predictions in SQLite for history/analytics
7. **Visualization**: Add trend charts showing sentiment over time

---

## 📚 What You Learned

- ✅ End-to-end ML project structure (data → train → serve → UI)
- ✅ Text preprocessing with regex
- ✅ TF-IDF feature extraction
- ✅ Multi-class Logistic Regression
- ✅ sklearn Pipelines (chain preprocessing + model)
- ✅ Model evaluation (accuracy, F1, confusion matrix)
- ✅ Model serialization with pickle
- ✅ REST API design with Flask
- ✅ Frontend ↔ Backend communication with fetch()
- ✅ Free cloud deployment (Render + GitHub Pages)