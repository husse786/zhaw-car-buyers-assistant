"""
Used-Car Buyer's Assistant — Gradio app.
Integrates three blocks:
  - ML Numeric: predicts a fair price from car specifications.
  - NLP: extracts specs from free text and explains the price in plain language (RAG-grounded).
  - Computer Vision: classifies visible damage severity, which adjusts the price.
Inference only — all models are pre-trained and loaded at startup.
"""

# ── Standard library ──
import os
import re
import json

# ── Third-party ──
import numpy as np
import joblib
import gradio as gr
from openai import OpenAI
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load .env for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Configuration ──
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL   = os.getenv("LLM_MODEL", "gpt-5.4-mini")
EMBED_MODEL = "text-embedding-3-small"

CV_MODEL_NAME = "hussesey/zhaw-aiapp-vit-car-damage-severity"

# Paths work both locally (repo layout) and on the Space (files next to app.py).
def _find(filename, local_path):
    """Return the local repo path if it exists, else the filename (Space layout)."""
    return local_path if os.path.exists(local_path) else filename

MODEL_PATH     = _find("price_model.pkl", "../models/price_model.pkl")
KNOWLEDGE_PATH = _find("carknowledge.md", "../data/carknowledge.md")

# Damage severity -> price adjustment (illustrative business logic)
SEVERITY_ADJUSTMENT = {
    "01-minor":    -0.07,
    "02-moderate": -0.20,
    "03-severe":   -0.40,
}

print("Config loaded. LLM model:", LLM_MODEL)
print("Price model path:", MODEL_PATH)
print("Knowledge path:", KNOWLEDGE_PATH)

# ── OpenAI client ──
client = OpenAI(api_key=LLM_API_KEY) if LLM_API_KEY else None
if client is None:
    print("WARNING: LLM_API_KEY not set.")

# ── Load the price model (ML block output) ──
price_bundle = joblib.load(MODEL_PATH)
print("Price model loaded:", price_bundle.get("model_name", "unknown"))

# ── Load the damage-severity model (CV block) from Hugging Face ──
cv_processor = AutoImageProcessor.from_pretrained(CV_MODEL_NAME)
cv_model = AutoModelForImageClassification.from_pretrained(CV_MODEL_NAME)
cv_model.eval()
print("CV model loaded:", CV_MODEL_NAME)
print("CV labels:", cv_model.config.id2label)

# ── Load and embed the knowledge base (NLP/RAG) ──
with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    knowledge_text = f.read()

def chunk_by_section(text):
    raw_parts = re.split(r'\n(?=#{1,3}\s|\*\*[A-Z])', text)
    return [p.strip() for p in raw_parts if len(p.strip()) > 50]

def embed_texts(texts):
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([item.embedding for item in response.data])

chunks = chunk_by_section(knowledge_text)
chunk_embeddings = embed_texts(chunks)
print(f"Knowledge base ready: {len(chunks)} chunks, dim {chunk_embeddings.shape[1]}")

# ── NLP: extract specs from free text ──
EXTRACT_FIELDS = ["make", "model", "production_year", "mileage_km",
                  "power_hp", "fuel_category", "transmission", "body_type"]
FUEL_CATEGORIES = ['CNG', 'Diesel', 'Electric', 'Electric/Diesel', 'Electric/Gasoline',
                   'Ethanol', 'Gasoline', 'LPG', 'Others', 'Unknown']
TRANSMISSIONS   = ['Automatic', 'Manual', 'Semi-automatic', 'Unknown']
BODY_TYPES      = ['Box', 'Breakdown truck', 'Car transport', 'Compact', 'Convertible',
                   'Coupe', 'Flatbed van', 'Flatbed+tarpaulin', 'Hydraulic work platform',
                   'Off-Road/Pick-up', 'Other', 'Panel van', 'Sedan', 'Station wagon',
                   'Station wagon/van', 'Transporter', 'Van', 'Van-high roof']

def extract_specs(listing_text):
    system_prompt = (
        "You are a precise assistant that extracts used-car specifications from a "
        "free-text listing. Return ONLY valid JSON, no extra text. "
        f"Use exactly these keys: {', '.join(EXTRACT_FIELDS)}.\n"
        "Rules:\n"
        "- Use numbers for production_year, mileage_km, and power_hp.\n"
        f"- 'petrol'/'benzin' maps to 'Gasoline'. Map fuel to ONE of: {FUEL_CATEGORIES}.\n"
        f"- Map transmission to ONE of: {TRANSMISSIONS}.\n"
        f"- Map body_type to the closest ONE of: {BODY_TYPES}.\n"
        "- If a categorical value is stated but unclear, use 'Unknown' (fuel/transmission) "
        "or 'Other' (body_type).\n"
        "- If a value is NOT stated at all, set it to null. Do not guess or invent values."
    )
    example_user = "BMW 320d, 2017, 95'000 km, 190 PS, Diesel, Automat, Limousine."
    example_json = json.dumps({"make": "BMW", "model": "320d", "production_year": 2017,
                               "mileage_km": 95000, "power_hp": 190, "fuel_category": "Diesel",
                               "transmission": "Automatic", "body_type": "Sedan"})
    response = client.chat.completions.create(
        model=LLM_MODEL, temperature=0,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": example_user},
                  {"role": "assistant", "content": example_json},
                  {"role": "user", "content": listing_text}])
    return json.loads(response.choices[0].message.content)

# ── ML: predict fair price from specs ──
CURRENT_YEAR = 2026
NUMERIC_DEFAULTS = {"ratings_average": np.nan, "ratings_count": np.nan,
    "ratings_recommend_percentage": np.nan, "nr_seats": 5, "nr_doors": 4, "gears": np.nan,
    "cylinders": np.nan, "cylinders_volume_cc": np.nan, "weight_kg": np.nan,
    "fuel_cons_comb_l100_km": np.nan, "co2_emission_grper_wltp_km": np.nan, "nr_prev_owners": np.nan}
CATEGORICAL_DEFAULTS = {"vehicle_type": "Unknown", "body_color": "Unknown", "paint_type": "Unknown",
    "upholstery": "Unknown", "upholstery_color": "Unknown", "drive_train": "Unknown",
    "primary_fuel": "Unknown", "envir_standard": "Unknown", "country_code": "Unknown",
    "seller_type": "Unknown"}
REMAINDER_DEFAULTS = {"has_particle_filter": False, "is_used": True, "is_new": False,
    "is_preregistered": False, "had_accident": False, "has_full_service_history": False,
    "non_smoking": False, "is_rental": False, "seller_is_dealer": True}
FREQ_DEFAULT_TEXT = {"original_market": "Unknown"}

def predict_price(specs):
    import pandas as pd
    row = dict(specs)
    year = row.get("production_year")
    row["car_age"] = max(CURRENT_YEAR - year, 0) if year else np.nan
    mileage = row.get("mileage_km")
    if mileage is not None and row.get("car_age") and row["car_age"] > 0:
        row["mileage_per_year"] = mileage / row["car_age"]
    else:
        row["mileage_per_year"] = mileage if mileage is not None else np.nan
    for col, default in {**NUMERIC_DEFAULTS, **CATEGORICAL_DEFAULTS,
                         **REMAINDER_DEFAULTS, **FREQ_DEFAULT_TEXT}.items():
        if col not in row or row[col] is None:
            row[col] = default
    ohe_cols = price_bundle["preprocessor"].transformers_[1][2]
    for col in price_bundle["feature_cols"]:
        if col not in row or row[col] is None:
            row[col] = "Unknown" if col in ohe_cols else np.nan
    df_row = pd.DataFrame([row])
    for col, freq in price_bundle["freq_maps"].items():
        df_row[col] = df_row[col].map(freq).fillna(0.0)
    df_row = df_row[price_bundle["feature_cols"]]
    X = price_bundle["preprocessor"].transform(df_row)
    return float(np.expm1(price_bundle["model"].predict(X)[0]))

# ── RAG: retrieve relevant knowledge ──
def cosine_similarity(query_vec, matrix):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return m @ q

def retrieve(query, top_k=3):
    qv = embed_texts([query])[0]
    scores = cosine_similarity(qv, chunk_embeddings)
    return [chunks[i] for i in np.argsort(scores)[::-1][:top_k]]

# ── NLP: RAG-grounded, persona-adapted explanation ──
PERSONAS = {
    "first_time": "a first-time car buyer who does not know technical car terms; explain jargon simply.",
    "budget": "a budget-conscious buyer; focus on running costs, reliability, and value for money.",
    "expat": "a non-native English speaker; use very clear, simple language and short sentences.",
}

def generate_explanation(specs, base_price, severity, adjusted_price, persona="first_time"):
    persona_desc = PERSONAS.get(persona, PERSONAS["first_time"])
    query = (f"used car buying advice: {specs.get('mileage_km')} km, "
             f"{specs.get('production_year')}, {specs.get('fuel_category')} engine, "
             f"damage and what to check before buying")
    context = "\n\n".join(retrieve(query, top_k=3))
    damage_line = ("No photo was provided, so no damage adjustment was applied."
                   if severity is None else
                   f"A photo indicated {severity} damage, adjusting the price to "
                   f"EUR {adjusted_price:.0f}.")
    system_prompt = (
        "You are a helpful used-car buying advisor. "
        f"You are explaining to {persona_desc} "
        "Use ONLY the factual knowledge in the context below to support advice; do not invent "
        "facts. A machine-learning model already estimated the price — do NOT recalculate it.\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Write a short explanation (3-4 sentences) that states the estimated fair price, explains "
        "what the key specs mean for value, mentions the damage adjustment if any, and includes "
        "exactly one uncertainty or limitation. Return ONLY valid JSON with key 'answer'.")
    user_content = (f"Specifications: {json.dumps(specs, ensure_ascii=False)}\n"
                    f"Base estimated price (EUR): {base_price:.0f}\n{damage_line}")
    response = client.chat.completions.create(
        model=LLM_MODEL, temperature=0.3,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_content}])
    return json.loads(response.choices[0].message.content)["answer"]

# ── CV: classify damage severity from an image ──
def classify_damage(image):
    """Classify visible damage severity from a PIL image.
    Returns (severity_label, confidence) or (None, None) if no image."""
    if image is None:
        return None, None
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)   # Gradio may pass a numpy array
    image = image.convert("RGB")
    inputs = cv_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = cv_model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs))
    label = cv_model.config.id2label[idx]
    confidence = float(probs[idx])
    return label, confidence

# ── Decision logic: CV severity adjusts the predicted price ──
def apply_damage_adjustment(base_price, severity):
    """Adjust the base price based on visible damage severity.
    Illustrative business logic. Returns (adjusted_price, pct_applied)."""
    if severity is None:
        return base_price, 0.0
    pct = SEVERITY_ADJUSTMENT.get(severity, 0.0)
    adjusted = base_price * (1 + pct)
    return adjusted, pct
# ── Orchestration: the full pipeline the UI calls ──
def analyze(listing_text, persona, image):
    """Run the full pipeline: extract specs -> predict price -> classify damage ->
    adjust price -> explain. Returns outputs for the UI."""
    if not listing_text or not listing_text.strip():
        return {}, "Please paste a car listing.", "—", "—", ""

    # 1. NLP: extract structured specs from the listing text
    specs = extract_specs(listing_text)

    # 2. ML: predict the base fair price from specs
    base_price = predict_price(specs)

    # 3. CV: classify damage severity (only if a photo was provided)
    severity, confidence = classify_damage(image)

    # 4. Decision logic: adjust price for damage
    adjusted_price, pct = apply_damage_adjustment(base_price, severity)

    # 5. NLP: plain, persona-adapted, RAG-grounded explanation
    explanation = generate_explanation(specs, base_price, severity, adjusted_price, persona)

    # Format outputs for display
    base_str = f"EUR {base_price:,.0f}"
    if severity is None:
        severity_str = "No photo provided"
        adjusted_str = f"EUR {base_price:,.0f} (no adjustment)"
    else:
        severity_str = f"{severity}  (confidence {confidence:.0%})"
        adjusted_str = f"EUR {adjusted_price:,.0f}  ({pct:+.0%} for damage)"

    return specs, base_str, severity_str, adjusted_str, explanation

# ── Gradio UI ──
EXAMPLES = [
    ["VW Golf 1.6 TDI, Baujahr 2016, 120'000 km, 110 PS, Diesel, manuell, Kombi.", "first_time", "example_inputs/example_minor.JPEG"],
    ["Audi A4 Avant, 2019, 78000 km, 190 hp, petrol, automatic.", "budget", "example_inputs/example_moderate.JPEG"],
    ["Toyota Yaris hybrid 2020, 45'000 km, one owner, great condition.", "expat", "example_inputs/example_severe.JPEG"],
]

with gr.Blocks(title="Used-Car Buyer's Assistant") as demo:
    gr.Markdown(
        "# Used-Car Buyer's Assistant\n"
        "Paste a car listing to get an estimated fair price and a plain-language explanation. "
        "Optionally upload a photo of visible damage to adjust the estimate.\n\n"
        "*This tool gives a guideline estimate only - not a binding valuation. But it can help you to negotiate the price with car dealer better and have a good estimation*"
    )

    with gr.Row():
        with gr.Column(scale=1):
            listing_in = gr.Textbox(
                label="Car listing (free text)",
                placeholder="e.g. Audi A4 Avant, 2019, 78000 km, 190 hp, petrol, automatic.",
                lines=4,
            )
            persona_in = gr.Dropdown(
                choices=["first_time", "budget", "expat"],
                value="first_time",
                label="Explain for",
            )
            image_in = gr.Image(
                type="pil",
                label="Photo of visible damage (optional)",
            )
            gr.Markdown(
                "*Damage severity reflects the closest of three categories "
                "(minor / moderate / severe); the model is trained on damaged cars only.*"
            )
            run_btn = gr.Button("Analyze", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=[listing_in, persona_in, image_in])

        with gr.Column(scale=2):
            specs_out    = gr.JSON(label="Extracted specifications")
            base_out     = gr.Textbox(label="Estimated fair price (specs only)")
            severity_out = gr.Textbox(label="Damage severity")
            adjusted_out = gr.Textbox(label="Adjusted price estimate")
            explanation_out = gr.Textbox(label="Explanation", lines=6)

    run_btn.click(
        fn=analyze,
        inputs=[listing_in, persona_in, image_in],
        outputs=[specs_out, base_out, severity_out, adjusted_out, explanation_out],
    )

if __name__ == "__main__":
    demo.launch()