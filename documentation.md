# AI Applications Project Documentation Template

Use this template to document your project concisely and completely.
Fill in all required fields. Keep answers short and precise.

## Documentation Hint

Important:
When possible, reference the corresponding code location directly in your description.

### Example: Reference to a notebook section

Reference to the header `## Data Preprocessing` in the notebook `analysis.ipynb`:

> See *Data Preprocessing* in
> [`analysis.ipynb`](analysis.ipynb#data-preprocessing)

### Example: Reference to Python code

Reference to a single line in `model.py`, line 42:
> [`model.py`, line 42](model.py#L42)

Reference to multiple lines in `train.py`, lines 15-38:
> [`train.py`, lines 15-38](train.py#L15-L38)

## Project Metadata

- Project title: Used-Car Buyer's Assistant
- Student: Seyid Hussein Husseini
- GitHub repository URL: <https://github.com/husse786/zhaw-car-buyers-assistant>
- Deployment URL: <https://zhaw-car-buyers-assistant.streamlit.app>
- Submission date: 05.06.2026

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [x] Computer Vision

Primary blocks used for core solution (choose 2):

- Primary block 1: ML Numeric Data — predicts a fair used-car price from specifications.
- Primary block 2: NLP — extracts specs from a free-text listing and explains the price in plain, persona-adapted language (RAG-grounded).

If a third block is selected, it is documented and graded separately as extra work.

- Third block :Computer Vision — classifies visible damage severity, which adjusts the price estimate.

Guidance hint: Keep the project idea short and consistent. Focus most details on the selected blocks.
Evidence hint: Show where each selected block contributes to the final system.

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition

- Problem statement: Non-expert buyers struggle to judge whether a used-car listing is fairly priced. They cannot easily assess a car's condition from photos and do not know what specifications such as mileage, fuel type, or transmission mean for value. This information asymmetry disadvantages them against sellers and car dealers.
- Goal: Estimate a guideline fair price from a free-text listing, adjust it for visible damage when a photo is supplied, and explain the result in plain, persona-adapted language.
- Success criteria:
  - Price model beats baseline (R² 0.946, MAE €4,651, MAPE 11.5%)
  - Damage classifier distinguishes severity (accuracy 0.685, macro-F1 0.682)
  - Clear, knowledge-grounded, persona-adapted explanations
  - Working public deployment

### 1.2 Integration Logic

- How the selected blocks interact: The blocks form one pipeline. NLP extracts structured specs from the listing; these feed the ML model, which predicts a base price. CV classifies damage severity from an optional photo, driving a rule-based price adjustment (decision logic). NLP then produces a final RAG-grounded explanation combining specs, price, and damage. The full pipeline is orchestrated in [`app/app.py`](app/app.py) (function `analyze`).
- Data and output flow between blocks:

```markdown
listing text ──> NLP extract ──> specs ──┐
                                         ├──> ML model ──> base price ──┐
photo (optional) ──> CV severity ────────┘                             │
                          └──> decision-logic adjustment ──> adjusted price
   specs + base price + severity + adjusted price + persona ──> NLP (RAG) ──> explanation
```

Guidance hint: This section should be short. The detailed work belongs in block sections.
Evidence hint: Include one clear pipeline overview.

---

## 2. Block Documentation

Complete only selected blocks. Mark non-selected block sections as N/A.

### 2A. ML Numeric Data (If selected)

#### 2A.1 Data Source(s)

List every usage of a data source as a separate entry. If the same source is used twice for different roles, add it twice.

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | AutoScout24 used-car listings (Kaggle, 2025 snapshot) <https://www.kaggle.com/datasets/clkmuhammed/autoscout24-car-listings-dataset> | Tabular (CSV) | ~118k rows × 75 cols (raw); 109,396 after cleaning | Training data for the price regression model |

#### 2A.2 Preprocessing and Features

- Cleaning steps:
Restricted to a single currency (EUR) and a plausible price range (€500–€150,000); removed rows with impossible or missing production years, non-positive engine power, and clipped extreme mileage; dropped leakage columns (e.g. net price, VAT rate) and non-predictive free-text/administrative columns; removed duplicate rows. The row count was tracked after each step (118,382 → 109,396 rows). See *Data Cleaning* in [`01_eda_preprocessing.ipynb`](notebooks/01_eda_preprocessing.ipynb#5-data-cleaning).

- Preprocessing steps: Applied a `log1p` transform to the target price (heavy right skew); standardised numeric features with `StandardScaler`; encoded low-cardinality categoricals with `OneHotEncoder(handle_unknown='ignore')`; applied frequency encoding to high-cardinality columns (`make`, `model`, `original_market`). All encoders and statistics were fit on the training split only to prevent data leakage. See *Preprocessing Pipeline* in [`02_ml_price_model.ipynb`](notebooks/02_ml_price_model.ipynb#2-preprocessing-pipeline-fit-on-train-only).

- Feature engineering and selection: Derived `car_age` (current year − production year) and `mileage_per_year` (mileage normalised by age). Excluded high-cardinality free-text fields and leakage-prone columns. The strongest predictors of price were `power_hp`, `car_age`, and `mileage_km`. See *Feature Engineering* in [`01_eda_preprocessing.ipynb`](notebooks/01_eda_preprocessing.ipynb#6-feature-engineering).

#### 2A.3 Model Selection

- Models tested: Ridge Regression (regularised linear baseline), Random Forest Regressor, and HistGradientBoostingRegressor.
- Why these models were chosen: The Ridge baseline establishes a reference point with a simple linear model and shows how much non-linear methods improve on it. Random Forest and HistGradientBoosting are tree-based ensembles chosen because used-car price depends on non-linear interactions between features (e.g. the effect of mileage differs by car age and engine power), which a linear model cannot capture. Gradient boosting was included as it typically reaches lower bias than bagging by fitting residuals sequentially. Hyperparameters were tuned on the validation split, and the final model was selected by validation performance. See *Model Training & Comparison* in [`02_ml_price_model.ipynb`](notebooks/02_ml_price_model.ipynb#3-model-training--comparison--3-models).

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Establish a baseline | Linear model on encoded features; `log1p` target | Ridge | Test R² −0.44; RMSE €38,858; MAE €10,333 | — (reference) |
| 2 | Capture non-linear feature interactions | Switched to a bagging tree ensemble | Random Forest | Test R² 0.932; RMSE €8,418; MAE €4,863 | Large gain over baseline (R² −0.44 -> 0.932) |
| 3 | Reduce bias further | Switched to gradient boosting (sequential residual fitting) | HistGradientBoosting | Test R² 0.946; RMSE €7,604; MAE €4,651 | Best result; selected model (RMSE €8,418 -> €7,604) |

#### 2A.5 Evaluation and Error Analysis

- Metrics used: RMSE, MAE, MAPE, and R², all computed in Euros (the `log1p` target transform is inverted with `expm1` before scoring). Data was split 80/10/10 into train/validation/test (seed 42). Models were compared on the validation split and the selected model was evaluated once on the held-out test set.
- Final results (HistGradientBoosting, test set): RMSE €7,604, MAE €4,651, MAPE 11.5%, R² 0.946.
- Error patterns and likely causes:
  - Highest percentage error on cheap cars (<€5k): approx. 41% MAPE, because small absolute errors are large relative to a low price, and budget listings have noisier pricing.
  - Largest absolute errors on high-end cars (>€60k): fewer training examples at the top of the range and luxury premiums that are hard to learn from specifications alone.
  - Error rises with car age: from approx.8.9% MAPE for 0–3 year-old cars to approx. 32% for cars over 20 years, where rarity and condition (not captured in the data) drive price.
  - Electric/hybrid and rare fuel types (e.g. LPG) are less accurate, reflecting volatile and feature-dependent pricing not fully represented in the specifications.
  
  See Error Analysis in [`02_ml_price_model.ipynb`](notebooks/02_ml_price_model.ipynb#5-error-analysis).

#### 2A.6 Integration with Other Block(s)

- Inputs received from other block(s): Structured car specifications (make, model, production year, mileage, power, fuel category, transmission, body type) extracted by the NLP block from the user's free-text listing. Specifications not present in the listing are filled with the model's training-time defaults before prediction.
- Outputs provided to other block(s): A base fair-price estimate in EUR. This estimate is then (a) adjusted by a rule-based factor driven by the Computer Vision damage-severity output (decision logic), and (b) passed to the NLP block, which explains the price and the specifications in plain language. The full pipeline is orchestrated in the `analyze` function in [`app.py`, Main orchestration](app/app.py).

Guidance hint: Keep entries practical and evidence-based.
Evidence hint: Add values, not only claims.

### 2B. NLP

#### 2B.1 Data Source(s)

List every usage of a data source as a separate entry. If the same source is used twice for different roles, add it twice.

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | `carknowledge.md` — curated used-car buying knowledge base, compiled from public car-buying websites (e.g. Carwow and similar guides) | Text (Markdown) | ~23,750 characters; 11 concept sections | Knowledge base for retrieval-augmented generation (RAG): grounds the explanation in verified car-buying facts |
| 2 | User free-text car listing (runtime input) | Text (user input) | One short text per request | Source text from which the LLM extracts structured specifications |

#### 2B.2 Preprocessing and Prompt Design

- Text preprocessing: The knowledge base is split into concept-level chunks by section heading, so each chunk is one self-contained topic (e.g. mileage, vehicle history report, frame damage), producing 11 chunks. Each chunk is embedded once with the OpenAI `text-embedding-3-small` model (1536-dimensional vectors) and stored in memory at startup. At query time, a query is embedded and the most relevant chunks are retrieved by cosine similarity (top-k = 3). See *Stage C — RAG* in [`04_nlp_rag.ipynb`](notebooks/04_nlp_rag.ipynb#stage-c-rag).

- Prompt design or retrieval setup: Two LLM prompts are used.
  - Extraction prompt (`temperature = 0` for consistency): a system instruction asks for strict JSON with a fixed set of keys, requires numeric values for numeric fields, constrains categorical fields (fuel, transmission, body type) to the exact vocabulary the price model was trained on, and returns `null` for values not stated rather than guessing. One worked example is included to anchor the format.
  - Explanation prompt (`temperature = 0.3`): a system instruction adapts the answer to a chosen buyer persona (first-time / budget-conscious / non-native speaker), explains the specifications and the predicted price in plain language, weaves in the retrieved knowledge-base context for grounding, and requires exactly one uncertainty note. It is instructed not to recompute the price.
  See *Stage A: Extract* and *Stage B: Explain* in [`04_nlp_rag.ipynb`](notebooks/04_nlp_rag.ipynb#stage-a-extract).

#### 2B.3 Approach Selection

- Approach used (classical NLP, transformer, RAG, prompt engineering): A combination of prompt engineering and retrieval-augmented generation (RAG). Prompt engineering drives two LLM calls: structured spec extraction (strict JSON, constrained vocabulary) and a persona-adapted explanation. RAG augments the explanation by retrieving relevant chunks from the curated knowledge base and injecting them into the prompt, so advice is grounded in verified car-buying facts rather than the model's internal knowledge alone.
- Alternatives considered: A prompt-only explanation (no retrieval) was implemented first and kept as the comparison baseline; RAG was then layered on top to reduce vague or unsupported claims. A fully local stack (open-source LLM with a vector database such as FAISS) was considered but not chosen: OpenAI embeddings with in-memory cosine similarity were sufficient for a small knowledge base (11 chunks) and kept the deployment lightweight. See *Stage C — RAG* and the prompt-only vs. RAG comparison in [`04_nlp_rag.ipynb`](notebooks/04_nlp_rag.ipynb#stage-c-evaluation).

#### 2B.4 Comparison and Iterations

| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Reliable spec extraction | Extraction prompt returning strict JSON | LLM, `temperature = 0`, one worked example | Valid JSON returned; but categorical values echoed the listing's wording (e.g. "petrol", "automatic"), risking mismatch with the price model's vocabulary | — (initial) |
| 2 | Prevent encoding mismatch | Constrained categorical outputs to the price model's exact training vocabulary; `null` for unstated values | Same call with an updated system prompt | Categorical values now standardised (e.g. "petrol" → "Gasoline"); unstated fields correctly returned as `null` | Eliminated silent feature-mismatch at the ML interface |
| 3 | Ground the explanation in verified facts | Added RAG: retrieve top-3 knowledge chunks and inject into the explanation prompt | Explanation prompt + retrieval (cosine similarity) | Qualitative: RAG version referenced verified concepts (pre-purchase inspection, history check, mileage wear) absent from the prompt-only version | More grounded, source-backed advice vs. prompt-only |

See the prompt-only vs. RAG comparison in *Stage C — Evaluation* in [`04_nlp_rag.ipynb`](notebooks/04_nlp_rag.ipynb#stage-c-evaluation).

#### 2B.5 Evaluation and Error Analysis

- Evaluation strategy: The NLP block was evaluated qualitatively, because there is no labelled ground truth for "correct" explanations. Extraction was checked by running varied listings (German/Swiss and English, complete and sparse) and verifying the JSON was valid, the values were standardised to the price model's vocabulary, and unstated fields were returned as `null` rather than guessed. The explanation was assessed by comparing the prompt-only and RAG-grounded versions on the same car and by checking persona adaptation across the three buyer profiles.
- Results: Extraction produced valid, correctly standardised JSON across test listings (e.g. Swiss "120'000 km" parsed to 120000; "petrol"/"automatic" mapped to "Gasoline"/"Automatic"; missing power/transmission returned as `null`). The RAG-grounded explanation referenced verified knowledge-base concepts (e.g. pre-purchase inspection, vehicle-history check, mileage-related wear) that the prompt-only version did not. Persona adaptation produced visibly different focus and reading level (e.g. the budget persona emphasised running costs; the non-native persona used shorter, simpler sentences).
- Error patterns and likely causes:
  - Ambiguous categorical values are handled conservatively rather than guessed: e.g. a "hybrid" listing was returned as fuel `Unknown` instead of choosing a specific hybrid subtype, since the type was not explicit.
  - Retrieval precision is moderate: for some queries the top-3 chunks include a loosely related section (e.g. lemon laws), because the knowledge base is small and several concepts overlap. The most relevant chunk is still retrieved, so the grounded answer remains accurate, but retrieval is not always tightly focused.
  - Persona differentiation is real but moderate: explanations share a similar opening and uncertainty-note structure across personas, with the main variation in focus and sentence complexity.
  
  See *Stage A — Extract*, *Stage B — Explain*, and *Stage C — Evaluation* in [`04_nlp_rag.ipynb`](notebooks/04_nlp_rag.ipynb#stage-c-evaluation).

#### 2B.6 Integration with Other Block(s)

- Inputs received from other block(s): The base price estimate (EUR) from the ML block, and the damage severity label (minor / moderate / severe) from the Computer Vision block when a photo is provided. The block also receives the user's free-text listing and chosen persona directly.
- Outputs provided to other block(s): Structured specifications (as JSON) that feed the ML price model, and the final natural-language explanation shown to the user — combining the specs, the predicted price, the damage adjustment, and retrieved knowledge.
- Representative output (Audi A4 Avant, 2019, 78,000 km, petrol, automatic; minor damage photo; first-time-buyer persona):
  - Extracted specs → `{"make": "Audi", "model": "A4 Avant", "production_year": 2019, "mileage_km": 78000, "power_hp": 190, "fuel_category": "Gasoline", "transmission": "Automatic", "body_type": "Station wagon"}`
  - Base price €25,282 → minor-damage adjustment −7% → adjusted €23,513
  - Explanation (excerpt): states the adjusted price, explains that 78,000 km mileage indicates moderate wear that reduces value, notes the minor-damage adjustment, and closes with an uncertainty note about service history and hidden issues.
  
  The full pipeline is orchestrated in the `analyze` function in [`app.py`, Main orchestration](app/app.py).

Guidance hint: Show concrete prompt or retrieval decisions.
Evidence hint: Include representative outputs or failure cases.

### 2C. Computer Vision

#### 2C.1 Data Source(s)

List every usage of a data source as a separate entry. If the same source is used twice for different roles, add it twice.

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |

#### 2C.2 Preprocessing and Augmentation
- Image preprocessing:
- Augmentation strategy:

#### 2C.3 Model Selection
- Vision model(s) used:
- Why these model(s) were chosen:

#### 2C.4 Model Comparison and Iterations
| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |

#### 2C.5 Evaluation and Error Analysis
- Metrics and/or visual checks:
- Final results:
- Error patterns and limitations:

#### 2C.6 Integration with Other Block(s)
- Inputs received from other block(s):
- Outputs provided to other block(s):

Guidance hint: Use concise examples from real predictions.
Evidence hint: Include sample outputs and observed failure cases.

---

## 3. Deployment

- Deployment URL:
- Main user flow:
- Screenshot or short demo:

Guidance hint: Deployment must be usable.
Evidence hint: Add screenshots or short demo references.

---

## 4. Execution Instructions

- Environment setup:
- Data setup:
- Training command(s):
- Inference/run command(s):
- Reproducibility notes:

Guidance hint: Another person should be able to run your project from this section.
Evidence hint: Include exact commands and versions.

---

## 5. Optional Bonus Evidence

Use this section for exceptional work beyond the core requirements.

- [ ] Third selected block implemented with strong quality
- [ ] More than two data sources used with clear added value
- [ ] A core section is done exceptionally well
- [ ] Extended evaluation
- [ ] Ethics, bias, or fairness analysis
- [ ] Creative or exceptional use case

Evidence for selected bonus items:
