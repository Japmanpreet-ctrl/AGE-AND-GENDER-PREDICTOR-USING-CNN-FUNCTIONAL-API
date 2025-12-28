# Age & Gender Prediction System

## Overview
This project is a deep learning–based **Age and Gender Prediction System** that estimates a person’s **age (regression)** and **gender (binary classification)** from a face image.  
It uses a **fine-tuned ResNet50 backbone**, trained on the **UTKFace dataset**, with a clean and consistent preprocessing pipeline to ensure stable real-world predictions.

The system is designed for **academic projects, demos, and learning purposes**, not for biometric or medical use.

---

## Key Features
- Predicts **age** as a continuous value (non-negative)
- Predicts **gender** with probability and uncertainty handling
- Uses **MTCNN** for face detection during inference
- Clean **training–inference consistency** (same preprocessing)
- Interactive **Gradio web interface**
- Robust to moderate pose, lighting, and background variations

---

## Model Architecture
- **Backbone**: ResNet50 (ImageNet pretrained, `include_top=False`)
- **Shared feature extractor**
- **Age head**:
  - Conv2D → BatchNorm → GlobalAveragePooling
  - Dense layers
  - ReLU output (prevents negative ages)
- **Gender head**:
  - Conv2D → BatchNorm → GlobalAveragePooling
  - Dense layers
  - Sigmoid output (probability)

---

## Dataset
- **UTKFace Dataset**
- Each image filename encodes:
  - Age
  - Gender
- Images are already **aligned and cropped**, making them suitable for training

Preprocessing applied:
- Resize to `224 × 224`
- `ResNet50 preprocess_input`
- Data augmentation **only on training set**

---

## Training Strategy
### Stage 1: Frozen Backbone
- ResNet50 weights frozen
- Train only task-specific heads
- Learning rate: `1e-4`

### Stage 2: Fine-tuning
- Unfreeze `conv5` block of ResNet50
- Lower learning rate: `1e-5`
- Early stopping on validation gender AUC

---

## Loss Functions and Metrics
### Loss
- **Age**: Huber Loss (robust to outliers)
- **Gender**: Binary Cross-Entropy

### Loss Weights
- Age: `1.0`
- Gender: `0.7` (prevents gender task from dominating)

### Metrics
- Age: MAE
- Gender: Accuracy, AUC

---

## Inference Pipeline
1. User uploads an image
2. **MTCNN** detects and crops the most confident face
3. Face is resized and preprocessed
4. Model predicts:
   - Age
   - Gender probability
5. Gender decision:
   - `< 0.45` → Male
   - `> 0.55` → Female
   - `0.45–0.55` → Uncertain

---

## Gradio Interface
- Simple web UI for image upload
- Displays:
  - Predicted Age
  - Predicted Gender
  - Gender Probability
- Handles cases where no face is detected

---

## Expected Performance
- **Gender accuracy**: ~85–90% on clear frontal faces
- **Age MAE**:
  - Adults: ±6–8 years
  - Children / elderly: ±8–12 years

Performance may degrade for:
- Side profiles
- Poor lighting
- Heavy occlusion
- Very young children

---

## Limitations
- Not suitable for real-world biometric or legal applications
- Dataset bias (UTKFace is not globally balanced)
- Age estimation is inherently noisy
- Gender prediction can be ambiguous for some faces

---

## Technologies Used
- Python
- TensorFlow / Keras
- ResNet50
- MTCNN
- Gradio
- NumPy, Pandas, Matplotlib

---

## How to Run
1. Train the model using the provided pipeline
2. Load the trained model
3. Launch the Gradio interface
4. Upload a face image and view predictions

---

## Disclaimer
This project is for **educational and demonstration purposes only**.  
Predictions are probabilistic and may be incorrect. Do not use this system for sensitive or decision-critical applications.

---

## Author Notes
This project emphasizes:
- Correct ML pipeline design
- Training–inference consistency
- Honest uncertainty handling
- Clean, explainable architecture

Built to be **defensible in viva, reviews, and demos**.
