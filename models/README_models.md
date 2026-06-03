# Models

This folder documents the trained models used in the Used-Car Buyer's Assistant.
Large model weights are NOT committed to GitHub — they live on the Hugging Face Hub
(CV model) or are produced by the training notebook (ML model). See pointers below.

## 1. Price Model (ML Numeric block)

- File: `price_model.pkl` (saved scikit-learn pipeline bundle)
- Type: HistGradientBoostingRegressor (best of 3 compared: Ridge, Random Forest, HGB)
- Predicts: fair used-car price (EUR) from specifications
- Test performance: R² 0.946, MAE €4,651, MAPE 11.5%
- Trained in: `notebooks/02_ml_price_model.ipynb`
- Input: raw feature columns matching `data/autoscout_clean.csv` (minus `price`)

## 2. Car Damage Severity Model (Computer Vision block)

- Type: Vision Transformer (ViT), transfer learning
  (backbone frozen, classifier head fine-tuned)
- Classes (3): 01-minor, 02-moderate, 03-severe
- Test performance: accuracy 0.685, macro-F1 0.682
- Trained in: `notebooks/03_cv_damage_model.ipynb`
- Hosted on Hugging Face: <https://huggingface.co/hussesey/zhaw-aiapp-vit-car-damage-severity>
- Weights (config.json, model.safetensors, preprocessor_config.json) are stored on
  Hugging Face, not in this repo (file size).

## Note on training/inference separation

Both models are trained offline (notebooks / Lightning AI GPU for the ViT) and only
loaded for inference in the deployed app. No training happens in the app.
