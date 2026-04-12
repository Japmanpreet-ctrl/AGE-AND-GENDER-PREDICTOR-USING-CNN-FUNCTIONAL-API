from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from age_gender_predictor.config import APP_TITLE, DATASET_DIR, MODEL_INFO_PATH, MODEL_PATH
from age_gender_predictor.inference import predict_image


CUSTOM_CSS = """
:root {
  --page-bg:
    radial-gradient(circle at top left, rgba(30, 64, 175, 0.24), transparent 30%),
    radial-gradient(circle at top right, rgba(14, 165, 233, 0.18), transparent 28%),
    linear-gradient(180deg, #081120 0%, #0f172a 52%, #172554 100%);
  --panel: rgba(15, 23, 42, 0.82);
  --panel-strong: rgba(15, 23, 42, 0.94);
  --panel-border: rgba(148, 163, 184, 0.18);
  --ink: #f8fafc;
  --muted: #cbd5e1;
  --accent: #22c55e;
  --accent-strong: #16a34a;
  --accent-soft: rgba(34, 197, 94, 0.14);
  --shadow: 0 24px 60px rgba(2, 6, 23, 0.45);
}
body, .gradio-container {
  background: var(--page-bg) !important;
  color: var(--ink) !important;
  font-family: "Avenir Next", "Segoe UI", sans-serif !important;
}
.gradio-container {
  max-width: 1180px !important;
}
.hero, .panel, .stat-card {
  background: var(--panel);
  border: 1px solid var(--panel-border);
  box-shadow: var(--shadow);
  border-radius: 24px;
}
.hero {
  padding: 32px;
  margin-bottom: 20px;
  background:
    radial-gradient(circle at top right, rgba(34, 197, 94, 0.16), transparent 32%),
    linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.96));
}
.panel {
  padding: 20px !important;
  background: var(--panel-strong);
}
.eyebrow {
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #86efac;
  font-size: 12px;
  font-weight: 700;
}
.hero h1 {
  margin: 8px 0 12px;
  font-size: 2.6rem;
  line-height: 1.05;
}
.hero p {
  margin: 0;
  max-width: 760px;
  color: var(--muted);
  font-size: 1.02rem;
}
.stats {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin: 18px 0 22px;
}
.stat-card {
  padding: 16px 18px;
  background: rgba(30, 41, 59, 0.92);
}
.stat-label {
  font-size: 0.8rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}
.stat-value {
  margin-top: 6px;
  font-size: 1.05rem;
  font-weight: 700;
}
.hint {
  margin-top: 14px;
  padding: 14px 16px;
  border-radius: 16px;
  background: var(--accent-soft);
  color: var(--ink);
}
.gr-image,
.gr-box,
.gr-form,
.gr-button,
.gr-markdown,
.gradio-container input,
.gradio-container textarea {
  color: var(--ink) !important;
}
.gradio-container label,
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose strong {
  color: var(--ink) !important;
}
.gradio-container .prose code {
  color: #bfdbfe !important;
  background: rgba(15, 23, 42, 0.95) !important;
}
button.primary {
  background: linear-gradient(135deg, var(--accent), var(--accent-strong)) !important;
  border: none !important;
  color: white !important;
}
@media (max-width: 900px) {
  .stats {
    grid-template-columns: 1fr;
  }
  .hero h1 {
    font-size: 2rem;
  }
}
"""


def _dataset_summary(dataset_dir: Path) -> str:
    if not dataset_dir.exists():
        return "Dataset folder missing"
    count = sum(1 for _ in dataset_dir.rglob("*.jpg"))
    return f"{count:,} images found"


def _load_model_summary(model_info_path: Path) -> dict[str, str] | None:
    if not model_info_path.exists():
        return None
    try:
        data = json.loads(model_info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    metrics = data.get("final_metrics", {})
    return {
        "age_mae": f"{metrics.get('val_age_mae', 'n/a')}",
        "gender_accuracy": f"{metrics.get('val_gender_accuracy', 'n/a')}",
        "gender_auc": f"{metrics.get('val_gender_auc', 'n/a')}",
        "dataset_size": f"{data.get('dataset_size', 'n/a'):,}" if isinstance(data.get("dataset_size"), int) else "n/a",
    }


def _render_prediction(image):
    prediction = predict_image(image)
    if "error" in prediction:
        return f"### Prediction\n\n{prediction['error']}"

    return (
        "## Prediction Result\n\n"
        f"**Predicted age:** {prediction['predicted_age']}\n\n"
        f"**Predicted gender:** {prediction['predicted_gender']}\n\n"
        f"**Gender probability:** {prediction['gender_probability']}\n\n"
        f"`{prediction['model_path']}`"
    )


def create_app() -> gr.Blocks:
    model_status = "Ready" if MODEL_PATH.exists() else "Model file needed"
    dataset_status = _dataset_summary(DATASET_DIR)
    model_summary = _load_model_summary(MODEL_INFO_PATH)
    validation_age = model_summary["age_mae"] if model_summary else "n/a"
    validation_gender = model_summary["gender_accuracy"] if model_summary else "n/a"
    validation_auc = model_summary["gender_auc"] if model_summary else "n/a"

    with gr.Blocks(title=APP_TITLE, css=CUSTOM_CSS) as demo:
        gr.HTML(
            f"""
            <section class="hero">
              <div class="eyebrow">Computer Vision Demo</div>
              <h1>{APP_TITLE}</h1>
              <p>Upload one clear face image and get age and gender predictions from the trained multitask ResNet50 pipeline using MTCNN face detection.</p>
              <div class="stats">
                <div class="stat-card">
                  <div class="stat-label">Model</div>
                  <div class="stat-value">{model_status}</div>
                </div>
                <div class="stat-card">
                  <div class="stat-label">Dataset</div>
                  <div class="stat-value">{dataset_status}</div>
                </div>
                <div class="stat-card">
                  <div class="stat-label">Validation Age MAE</div>
                  <div class="stat-value">{validation_age}</div>
                </div>
                <div class="stat-card">
                  <div class="stat-label">Validation Gender Acc</div>
                  <div class="stat-value">{validation_gender}</div>
                </div>
                <div class="stat-card">
                  <div class="stat-label">Validation Gender AUC</div>
                  <div class="stat-value">{validation_auc}</div>
                </div>
                <div class="stat-card">
                  <div class="stat-label">Pipeline</div>
                  <div class="stat-value">Detect, crop, predict</div>
                </div>
              </div>
              <div class="hint"><strong>Best results:</strong> use one clear, front-facing face with good lighting and minimal background clutter.</div>
            </section>
            """
        )

        with gr.Row():
            with gr.Column(scale=5, elem_classes=["panel"]):
                input_image = gr.Image(type="pil", label="Upload face image")
                run_button = gr.Button("Run Prediction", variant="primary")
            with gr.Column(scale=5, elem_classes=["panel"]):
                output_markdown = gr.Markdown("## Prediction Result\n\nUpload an image and click **Run Prediction**.")

        gr.Examples(
            examples=[["Example.jpeg"]],
            inputs=[input_image],
        )

        run_button.click(fn=_render_prediction, inputs=[input_image], outputs=[output_markdown])

    return demo
