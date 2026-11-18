# InfoShield: Fake News & Disinformation Detection System
## Comprehensive Project Specification & Requirements

**Course:** INT395 — Supervised Learning  
**Problem Statement #18:** InfoShield — LLM Disinformation Red-Teaming  
**Assigned Datasets:** LIAR Fake News Dataset (`train.tsv`, `valid.tsv`, `test.tsv`) & PolitiFact Fact-Check Data (`politifact_factcheck_data.json`)  
**Dataset Directory:** [`/home/Aruuu/int395/datasets/`](file:///home/Aruuu/int395/datasets/)

---

## 1. Executive Summary & Goals

InfoShield is an end-to-end intelligent fake news detection, explainability, and continual learning system. The application serves two main goals:
1. **Accurate Supervised Classification:** Classify news statements and articles into credibility levels (True, Mostly True, Half True, Barely True, False, Pants-on-Fire, or Binary: Reliable vs. Fake).
2. **Explainability & Human-in-the-Loop Feedback:** Explain to the user *why* a particular piece of news was flagged (highlighting influential keywords and reasoning), accept user feedback/corrections, and continuously learn from new feedback over time without losing past knowledge.
3. **Multi-Platform Deployment:** Accessible via an interactive **Web Application** and as a **Google Chrome Extension** allowing users to highlight text on any webpage and verify its authenticity in real time.

---

## 2. Dataset Overview & Preprocessing Pipeline

### 2.1 Datasets Located in `datasets/`
* **`train.tsv`** (~10,240 records): Primary training set from the LIAR benchmark.
* **`valid.tsv`** (~1,284 records): Validation set for hyperparameter tuning.
* **`test.tsv`** (~1,267 records): Held-out benchmark test set.
* **`politifact_factcheck_data.json`** (~9.8 MB): Enriched PolitiFact records with statements, verdicts, and factual explanation snippets.

### 2.2 Preprocessing Steps
* **Target Label Mapping:**
  * **Mode A (Binary - Recommended for Web/Extension):**
    * *Reliable (0):* `true`, `mostly-true`
    * *Disinformation / Fake (1):* `half-true`, `barely-true`, `false`, `pants-fire`
  * **Mode B (Fine-Grained 6-class):** Preserving all 6 original LIAR labels for in-depth academic evaluation.
* **Text Normalization:**
  * Lowercasing, punctuation stripping, URL extraction/stripping, stop-word removal.
  * Lemmatization / Stemming via `nltk` or `spacy`.
  * Feature extraction using **TF-IDF Vectorization** (n-grams from $(1, 2)$ or $(1, 3)$, max features: 10,000–25,000).
  * Optional metadata fusion: Incorporating speaker party, context, and subject as auxiliary categorical features.

---

## 3. Model Architecture & Selection

To ensure **low RAM usage**, fast execution on standard hardware, and compatibility with continuous learning, the system uses:

| Model | Purpose | Why Selected | RAM Footprint |
| :--- | :--- | :--- | :--- |
| **Passive-Aggressive Classifier / SGDClassifier** | Primary Online / Continual Learning Model | Supports incremental learning (`partial_fit`) so it learns from user feedback instantly without full retraining. Excellent for text classification. | < 100 MB |
| **Multinomial / Complement Naive Bayes** | Fast Baseline Classifier | Probabilistic, fast, highly effective with TF-IDF, supports incremental updates. | < 50 MB |
| **Logistic Regression (L2 / ElasticNet)** | Explainability Benchmark | Clean calibrated probabilities, direct coefficient-level feature weights for explanations. | < 80 MB |
| **LightGBM / Random Forest** (Optional) | Non-linear Tabular/Metadata Ensemble | High classification accuracy when combining text TF-IDF + speaker metadata. | < 250 MB |

---

## 4. Model Persistence: "Remembering Training for Eternity"

To ensure the model never loses its knowledge across server restarts or system shutdowns:

### 4.1 Serialization & Pipeline Packaging
* The entire end-to-end pipeline (Text Preprocessor + TF-IDF Vectorizer + Trained Classifier) is serialized into an atomic model bundle using **`joblib`**:
  ```python
  import joblib
  from sklearn.pipeline import Pipeline
  
  pipeline = Pipeline([
      ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=15000)),
      ('clf', PassiveAggressiveClassifier(max_iter=1000, random_state=42))
  ])
  
  pipeline.fit(X_train, y_train)
  joblib.dump(pipeline, "models/infoshield_v1.joblib")
  ```

### 4.2 Model Versioning & Registry
* **Versioned Checkpoints:** Save models under `models/model_v{N}.joblib`.
* **Metadata Manifest (`models/registry.json`):** Records:
  * Model version & creation timestamp
  * Training sample count (base dataset + user feedback count)
  * Evaluation metrics on test set (Accuracy, Precision, Recall, F1)
  * Hash of the active weights file

---

## 5. Explainability Engine (XAI — "Why It Thinks This")

When the system classifies a statement, it must justify its reasoning to the user.

### 5.1 Keyword Attribution (Local Interpretable Token Weights)
* Compute word attribution scores using either:
  1. **Direct Linear Coefficients:** Multiply input TF-IDF vector by model weight vector $w_i \cdot x_i$. Positive contributions push toward "Fake/Disinformation", negative contributions push toward "Real".
  2. **LIME (Local Interpretable Model-agnostic Explanations):** Perturbs the input sentence and fits an interpretable surrogate model to produce word importance scores.
* **Visual Output:** Words highlighted in red (high disinformation signal, sensationalism, dubious claims) and green (credible / verifiable language).

### 5.2 Structured Plain-English Explanation
* The model generates a structured verdict response:
  * **Verdict:** *Likely Disinformation (84% confidence)*
  * **Key Trigger Words:** `["miracle cure", "conspiracy", "secretly hidden", "shocking truth"]`
  * **Summary Rationale:** *"The statement contains high-frequency sensationalist phrases and linguistic patterns typically correlated with unsubstantiated or false claims in the PolitiFact dataset."*

---

## 6. Feedback Loop & Continual Learning ("Learning from Predictions")

### 6.1 Feedback Collection
* In the Web App and Chrome Extension, users can submit:
  * **Feedback Type:** *Agree* or *Disagree*
  * **User-Assigned Label:** *Real* or *Fake*
  * **Verification Source / Link:** URL or source proving the claim.
* Feedbacks are logged in a local SQLite database (`storage/feedback.db`) with fields: `id`, `text`, `predicted_label`, `confidence`, `user_label`, `evidence_url`, `timestamp`, `applied_to_model`.

### 6.2 Learning Mechanism: Dual-Track Updating
1. **Track 1: Online Incremental Update (Instant)**
   * Using algorithms with `.partial_fit()` (`PassiveAggressiveClassifier` or `SGDClassifier`), the model updates its weights with the newly validated sample immediately.
   * `model.partial_fit([new_text_vector], [user_label])`
   * The updated pipeline is immediately saved back to `models/infoshield_active.joblib`.
2. **Track 2: Periodic Batch Retraining (Robustness & Guardrails)**
   * When $N \ge 50$ feedback submissions are reached:
     * Retrain the base model on `Original LIAR Dataset + Validated Feedback Data`.
     * Evaluate against a protected **Golden Benchmark Test Set** (`test.tsv`).
     * Only promote the new model if its benchmark accuracy does not degrade (prevents adversarial poisoning).

---

## 7. Web Application Architecture

```
[ User Browser / Chrome Extension ]
                │
         HTTP REST API
                ▼
      ┌──────────────────┐
      │   FastAPI Core   │
      └─────────┬────────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
[ Predict ] [ Explain ] [ Feedback ]
    │           │           │
    ▼           ▼           ▼
[ Model Pipeline ]     [ SQLite DB ]
(infoshield.joblib)     (feedback.db)
```

### 7.1 Backend API (FastAPI)
* `POST /api/v1/predict`:
  * **Input:** `{"text": "Breaking news headline or paragraph..."}`
  * **Output:** `{"prediction": "Fake", "confidence": 0.88, "risk_level": "High", "key_factors": [...], "explanation": "..."}`
* `POST /api/v1/feedback`:
  * **Input:** `{"text": "...", "model_prediction": "Fake", "user_label": "Real", "source_url": "..."}`
  * **Output:** `{"status": "recorded", "model_updated": true}`
* `GET /api/v1/metrics`:
  * **Output:** Model accuracy, total feedback count, current model version.

### 7.2 Frontend Web Interface
* **Streamlit / Modern HTML5 + Tailwind CSS Dashboard:**
  * Clean text area to paste news text or URLs.
  * Visual credibility gauge (0% to 100% Reliability score).
  * Interactive highlighted text showing positive/negative keyword impact.
  * One-click "Correct This Result" feedback drawer.

---

## 8. Chrome Extension Integration (Feasibility & Design)

### 8.1 Is it Feasible?
**Yes, 100% feasible.** A lightweight Chrome Extension (Manifest V3) can connect directly to your local FastAPI backend running on your machine.

### 8.2 Chrome Extension Architecture:
* **`manifest.json`:** Configures permissions (`contextMenus`, `activeTab`, `storage`).
* **`content.js`:**
  * Listens for text selection across any webpage.
  * Shows a floating **"Verify with InfoShield"** icon next to highlighted text.
* **`background.js` (Service Worker):**
  * Also registers a Right-Click Context Menu item: *"Check authenticity with InfoShield"*.
  * Sends an asynchronous HTTP request:
    ```javascript
    fetch("http://localhost:8000/api/v1/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: selectedText })
    })
    ```
* **Popup / In-Page Tooltip:**
  * Displays the verification badge: **🔴 Likely Disinformation** or **🟢 Credible**.
  * Shows the top contributing words and the confidence score.
  * Provides quick **Thumbs Up / Thumbs Down** buttons to send instant feedback to the API.

---

## 9. Technology Stack

* **Language:** Python 3.10+
* **ML & NLP:** `scikit-learn`, `joblib`, `nltk`, `lime` / `shap`
* **Backend:** `FastAPI`, `uvicorn`, `pydantic`
* **Database:** `SQLite` (zero configuration, low memory)
* **Web UI:** `Streamlit` (or lightweight HTML/JS + Tailwind)
* **Browser Extension:** Chrome Extension Manifest V3 (JavaScript, HTML, CSS)
* **Hardware Footprint:** **CPU-only**, < 200 MB RAM for ML engine, instant execution.

---

## 10. Implementation Roadmap

1. **Phase 1: Data Pipeline & Baseline Model**
   * Load and clean `datasets/train.tsv`, `datasets/valid.tsv`, and `datasets/test.tsv`.
   * Train baseline models (`PassiveAggressiveClassifier`, `LogisticRegression`, `NaiveBayes`).
   * Evaluate baseline accuracy and confusion matrix.
2. **Phase 2: Serialization & Explainability**
   * Build unified `scikit-learn` Pipeline with `joblib` save/load routines.
   * Implement feature attribution and token highlighter.
3. **Phase 3: Feedback & Incremental Learning**
   * Setup SQLite feedback table.
   * Implement `partial_fit` online update logic and retraining guardrails.
4. **Phase 4: FastAPI REST Service & Web Interface**
   * Build `/predict` and `/feedback` endpoints.
   * Create interactive demo interface.
5. **Phase 5: Chrome Extension Development**
   * Write `manifest.json`, `content.js`, and popup UI.
   * Test end-to-end highlighting in Chrome.
