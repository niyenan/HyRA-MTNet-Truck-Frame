import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shap
import joblib
import json
import os
import sys


# ==========================================
# 1. Model Definition
# (Must be identical to the architecture in main.py)
# ==========================================
class DeepLifeHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):
        return self.model(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_dim, attn_dim=256, num_heads=4):
        super().__init__()
        self.project_in = nn.Linear(input_dim, attn_dim)
        self.attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True)
        self.project_out = nn.Linear(attn_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        x_in = x.unsqueeze(1)
        x_proj = self.project_in(x_in)
        attn_out, _ = self.attn(x_proj, x_proj, x_proj)
        out = self.project_out(attn_out).squeeze(1)
        return self.norm(out + x)


class FinalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, input_dim)
            )
            for _ in range(num_layers)
        ])
        self.attn = AttentionBlock(input_dim)
        self.mass_head = nn.Linear(input_dim, 1)
        self.modal_head = nn.Linear(input_dim, 3)
        self.stress_head = nn.Linear(input_dim, 4)
        self.stiffness_head = nn.Linear(input_dim, 2)
        self.disp_head = nn.Linear(input_dim, 4)
        self.life_head = DeepLifeHead(input_dim)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        x = self.attn(x)

        mass_out = self.mass_head(x)
        modal_out = self.modal_head(x)
        stress_out = self.stress_head(x)
        stiffness_out = self.stiffness_head(x)
        disp_out = self.disp_head(x)
        life_out = self.life_head(x)

        return torch.cat([mass_out, modal_out, stress_out, stiffness_out, life_out, disp_out], dim=1)


# ==========================================
# 2. SHAP Wrapper
# ==========================================
class PyTorchModelWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.numpy()


# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting Explainability Analysis...")

    # --- Step 1: Check for required files ---
    required_files = [
        'best_model.pth',
        'ohe_encoder.joblib',
        'scaler_X.joblib',
        'ohe_input_cols.json',
        'X_train_processed.npy'  # This file comes from main.py
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Error: The following required files are missing: {missing_files}")
        print("👉 Please run 'main.py' first to generate the model and processed data.")
        sys.exit(1)

    # --- Step 2: Load resources ---
    print("📥 Loading pre-processed data and model artifacts...")

    # Load processed training data (numpy array)
    X_processed = np.load('X_train_processed.npy')

    # Load feature names
    with open('ohe_input_cols.json', 'r') as f:
        feature_names = json.load(f)

    # Initialize model
    print("🧠 Initializing model...")
    input_dim = X_processed.shape[1]
    model = FinalModel(input_dim, hidden_dim=512, num_layers=6)

    # Load trained weights
    try:
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model weights: {e}")
        sys.exit(1)

    # --- Step 3: SHAP Analysis ---
    print("✨ Calculating SHAP values (this may take a few minutes)...")

    # Use a random subset of 200 samples as the background dataset for SHAP to speed up calculation
    # (Using the full dataset would be extremely slow)
    if len(X_processed) > 200:
        background_indices = np.random.choice(X_processed.shape[0], 200, replace=False)
        background_samples = X_processed[background_indices]
    else:
        background_samples = X_processed

    model_wrapper = PyTorchModelWrapper(model)
    explainer = shap.Explainer(model_wrapper, background_samples)

    # Calculate SHAP values
    shap_values = explainer(background_samples)

    # --- Step 4: Global Feature Importance (Figure 11a source) ---
    print("📊 Generating Global Feature Importance...")

    # Average SHAP values across all output tasks
    shap_values_mean = shap_values.values.mean(axis=2)
    shap_values_df = pd.DataFrame(shap_values_mean, columns=feature_names)

    # Calculate mean absolute SHAP value for ranking
    shap_importance = shap_values_df.abs().mean(axis=0).sort_values(ascending=False).head(20)

    # Save to CSV
    shap_importance_df = pd.DataFrame({
        'Feature': shap_importance.index,
        'Mean SHAP Value': shap_importance.values
    })
    shap_importance_df.to_csv("top_20_shap_features.csv", index=False)
    print("✅ Saved: 'top_20_shap_features.csv'")

    # --- Step 5: Fatigue Life Feature Importance (Figure 11b source) ---
    print("📊 Generating Fatigue Life Feature Importance...")

    # 'Life' is the 11th task (index 10) based on target_cols definition in main.py
    life_task_index = 10
    shap_values_life = shap_values.values[:, :, life_task_index]

    shap_life_df = pd.DataFrame(shap_values_life, columns=feature_names)
    shap_life_importance = shap_life_df.abs().mean(axis=0).sort_values(ascending=False).head(20)

    # Save to CSV
    shap_life_importance_df = pd.DataFrame({
        'Feature': shap_life_importance.index,
        'Mean SHAP Value': shap_life_importance.values
    })
    shap_life_importance_df.to_csv("top_20_shap_features_life.csv", index=False)
    print("✅ Saved: 'top_20_shap_features_life.csv'")

    print("\n🎉 Explainability analysis completed successfully!")