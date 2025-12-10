import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import shap
import time
import os
import json
import joblib

# ==========================================
# 1. Data Loading & Preparation
# ==========================================

print("📥 Loading datasets...")
try:
    # Try loading from the 'data/' directory (standard GitHub structure)
    df_sim = pd.read_csv("data/simulation_data_real.csv")
    df_exp = pd.read_csv("data/experiment_data_dummy.csv")
    data = pd.concat([df_sim, df_exp], axis=0).reset_index(drop=True)
    print("✅ Data loaded successfully from 'data/' folder.")
except FileNotFoundError:
    # Fallback for local testing if files are in the root directory
    print("⚠️ 'data/' folder not found. Attempting to load from root directory...")
    try:
        data = pd.read_csv("data_offnoise.csv")
        print("✅ Data loaded from root directory.")
    except FileNotFoundError:
        print("❌ Error: Dataset not found. Please ensure 'data_offnoise.csv' or the 'data/' folder exists.")
        exit()

# Data cleaning
data = data.replace([np.inf, -np.inf], np.nan).dropna()
data = data[data['life'] > 0]

# Define columns
material_cols = [f"m{i}" for i in range(1, 14)]
# Automatically detect numerical columns based on column positions
num_cols = data.columns[data.columns.get_loc('m13') + 1:data.columns.get_loc('d6_h') + 1]
input_cols = list(material_cols) + list(num_cols)
target_cols = [
    'mass', 'modal_frequency_1', 'modal_frequency_2', 'modal_frequency_3',
    'bend_stress', 'tor_stress', 'steer_stress', 'brake_stress',
    'bend_stiffness', 'tor_stiffness', 'life',
    'bend_disp', 'tor_disp', 'steer_disp', 'brake_disp'
]

# Feature groups for engineering
stress_cols = ['bend_stress', 'tor_stress', 'steer_stress', 'brake_stress']
stiffness_cols = ['bend_stiffness', 'tor_stiffness']
modal_cols = [f'modal_frequency_{i}' for i in range(1, 4)]

# ==========================================
# 2. Feature Engineering & Preprocessing
# ==========================================

# One-Hot Encoding for material features
if sklearn.__version__ >= "1.2":
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
else:
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

material_ohe = ohe.fit_transform(data[material_cols])
material_ohe_cols = ohe.get_feature_names_out(material_cols)
material_ohe_df = pd.DataFrame(material_ohe, columns=material_ohe_cols, index=data.index)

# Combine OHE features with numerical features
data = pd.concat([material_ohe_df, data.drop(columns=material_cols)], axis=1)

# --- Advanced Feature Engineering ---
# Generating domain-specific interaction features
new_feats = {}

# Stress & Stiffness Interactions
for stress in stress_cols:
    for stiff in stiffness_cols:
        new_feats[f'{stress}_x_{stiff}'] = data[stress] * data[stiff]
        new_feats[f'{stress}_div_{stiff}'] = data[stress] / (data[stiff] + 1e-6)

# Global Statistical Features
new_feats['stress_sum'] = data[stress_cols].sum(axis=1)
new_feats['stiffness_sum'] = data[stiffness_cols].sum(axis=1)
new_feats['stress_density'] = new_feats['stress_sum'] / (data['mass'] + 1e-6)

# Energy-related Feature (Strain Energy Approximation)
new_feats['energy_feature'] = (
        data['bend_stress'] ** 2 / (data['bend_stiffness'] + 1e-6) +
        data['tor_stress'] ** 2 / (data['tor_stiffness'] + 1e-6) +
        data['steer_stress'] ** 2 / (data['bend_stiffness'] + 1e-6) +
        data['brake_stress'] ** 2 / (data['tor_stiffness'] + 1e-6)
)

# Fatigue Life Proxy Feature
new_feats['life_feature'] = np.log(new_feats['stress_sum'] / (new_feats['stiffness_sum'] + 1e-6) + 1.0)

# Physical Ratios
new_feats['mass_density'] = data['mass'] / (data['bend_stiffness'] + data['tor_stiffness'] + 1e-6)
new_feats['bend_ratio'] = data['bend_stress'] / (data['bend_stiffness'] + 1e-6)
new_feats['tor_ratio'] = data['tor_stress'] / (data['tor_stiffness'] + 1e-6)
new_feats['combined_ratio'] = (new_feats['bend_ratio'] + new_feats['tor_ratio']) / 2
new_feats['stress_var'] = data[stress_cols].var(axis=1)
new_feats['stiffness_var'] = data[stiffness_cols].var(axis=1)
new_feats['stress_max'] = data[stress_cols].max(axis=1)
new_feats['stiffness_min'] = data[stiffness_cols].min(axis=1)
new_feats['tor_to_bend'] = data['tor_stiffness'] / (data['bend_stiffness'] + 1e-6)

# Modal Frequency Interactions
new_feats['mass_x_combined_ratio'] = data['mass'] * new_feats['combined_ratio']
new_feats['mass_x_stiffness_sum'] = data['mass'] * new_feats['stiffness_sum']
new_feats['modal1_x_mass'] = data['modal_frequency_1'] * data['mass']
new_feats['modal1_x_combined'] = data['modal_frequency_1'] * new_feats['combined_ratio']
new_feats['modal2_x_mass'] = data['modal_frequency_2'] * data['mass']
new_feats['modal2_div_stiff'] = data['modal_frequency_2'] / (new_feats['stiffness_sum'] + 1e-6)
new_feats['modal3_x_stiff'] = data['modal_frequency_3'] * new_feats['stiffness_sum']

# Displacement Interactions
new_feats['disp_sum'] = data[['bend_disp', 'tor_disp', 'steer_disp', 'brake_disp']].sum(axis=1)
new_feats['disp_div_mass'] = new_feats['disp_sum'] / (data['mass'] + 1e-6)
new_feats['bend_disp_x_stress'] = data['bend_disp'] * data['bend_stress']
new_feats['tor_disp_x_tor_stiff'] = data['tor_disp'] * data['tor_stiffness']
new_feats['steer_disp_x_energy'] = data['steer_disp'] * new_feats['energy_feature']
new_feats['brake_disp_x_mass'] = data['brake_disp'] * data['mass']
new_feats['steer_disp_x_steer_stress'] = data['steer_disp'] * data['steer_stress']
new_feats['steer_disp_div_steer_stress'] = data['steer_disp'] / (data['steer_stress'] + 1e-6)
new_feats['steer_disp_x_tor_stiffness'] = data['steer_disp'] * data['tor_stiffness']
new_feats['steer_disp_x_bend_stiffness'] = data['steer_disp'] * data['bend_stiffness']
new_feats['steer_disp_div_mass'] = data['steer_disp'] / (data['mass'] + 1e-6)
new_feats['steer_disp_x_stress_density'] = data['steer_disp'] * new_feats['stress_density']

# Modal Coupling Features
new_feats['modal2_x_modal3'] = data['modal_frequency_2'] * data['modal_frequency_3']
new_feats['modal2_div_modal3'] = data['modal_frequency_2'] / (data['modal_frequency_3'] + 1e-6)
new_feats['modal2_x_combined'] = data['modal_frequency_2'] * new_feats['combined_ratio']
new_feats['modal2_x_energy'] = data['modal_frequency_2'] * new_feats['energy_feature']
new_feats['modal2_div_mass_density'] = data['modal_frequency_2'] / (new_feats['mass_density'] + 1e-6)
new_feats['modal3_x_combined'] = data['modal_frequency_3'] * new_feats['combined_ratio']
new_feats['modal3_x_mass_density'] = data['modal_frequency_3'] * new_feats['mass_density']
new_feats['modal3_x_energy'] = data['modal_frequency_3'] * new_feats['energy_feature']
new_feats['modal3_div_stress_sum'] = data['modal_frequency_3'] / (new_feats['stress_sum'] + 1e-6)

# Steer Displacement Interactions
new_feats['steer_disp_x_modal1'] = data['steer_disp'] * data['modal_frequency_1']
new_feats['steer_disp_x_modal2'] = data['steer_disp'] * data['modal_frequency_2']
new_feats['steer_disp_x_modal3'] = data['steer_disp'] * data['modal_frequency_3']
new_feats['steer_disp_x_mass_density'] = data['steer_disp'] * new_feats['mass_density']
new_feats['steer_disp_div_stress_sum'] = data['steer_disp'] / (new_feats['stress_sum'] + 1e-6)
new_feats['steer_disp_div_energy'] = data['steer_disp'] / (new_feats['energy_feature'] + 1e-6)
new_feats['steer_disp_x_combined_ratio'] = data['steer_disp'] * new_feats['combined_ratio']

# High-order Modal Statistics
for i in range(1, 4):
    mf = f'modal_frequency_{i}'
    new_feats[f'{mf}_div_stiffness_var'] = data[mf] / (new_feats['stiffness_var'] + 1e-6)
    new_feats[f'{mf}_x_stress_density'] = data[mf] * new_feats['stress_density']
    new_feats[f'{mf}_div_stiffness_min'] = data[mf] / (new_feats['stiffness_min'] + 1e-6)
    new_feats[f'{mf}_x_stiffness_min'] = data[mf] * new_feats['stiffness_min']
    new_feats[f'{mf}_x_energy_per_stress'] = data[mf] * (new_feats['energy_feature'] / (new_feats['stress_sum'] + 1e-6))
    new_feats[f'{mf}_x_energy_per_mass'] = data[mf] * (new_feats['energy_feature'] / (data['mass'] + 1e-6))
    new_feats[f'{mf}_div_bend_ratio'] = data[mf] / (new_feats['bend_ratio'] + 1e-6)
    new_feats[f'{mf}_div_tor_ratio'] = data[mf] / (new_feats['tor_ratio'] + 1e-6)
    new_feats[f'{mf}_div_energy_feature'] = data[mf] / (new_feats['energy_feature'] + 1e-6)
    new_feats[f'{mf}_x_stress_var'] = data[mf] * new_feats['stress_var']
    new_feats[f'{mf}_x_stiffness_var'] = data[mf] * new_feats['stiffness_var']
    new_feats[f'{mf}_x_tor_to_bend'] = data[mf] * new_feats['tor_to_bend']

# Modal Frequency Differences & Ratios
new_feats['modal2_minus_modal1'] = data['modal_frequency_2'] - data['modal_frequency_1']
new_feats['modal3_minus_modal1'] = data['modal_frequency_3'] - data['modal_frequency_1']
new_feats['modal2_div_modal1'] = data['modal_frequency_2'] / (data['modal_frequency_1'] + 1e-6)
new_feats['modal3_div_modal1'] = data['modal_frequency_3'] / (data['modal_frequency_1'] + 1e-6)
new_feats['modal3_minus_modal2'] = data['modal_frequency_3'] - data['modal_frequency_2']
new_feats['modal3_div_modal2'] = data['modal_frequency_3'] / (data['modal_frequency_2'] + 1e-6)

# Modal 1 Specialized Features
new_feats['modal1_div_mass_density'] = data['modal_frequency_1'] / (new_feats['mass_density'] + 1e-6)
new_feats['modal1_div_combined_ratio'] = data['modal_frequency_1'] / (new_feats['combined_ratio'] + 1e-6)
new_feats['modal1_div_energy'] = data['modal_frequency_1'] / (new_feats['energy_feature'] + 1e-6)

# Bending Displacement Features
new_feats['bend_disp_div_modal1'] = data['bend_disp'] / (data['modal_frequency_1'] + 1e-6)
new_feats['bend_disp_x_stress_density'] = data['bend_disp'] * new_feats['stress_density']
new_feats['bend_disp_div_combined_ratio'] = data['bend_disp'] / (new_feats['combined_ratio'] + 1e-6)

# Braking Displacement Features
new_feats['brake_disp_x_modal1'] = data['brake_disp'] * data['modal_frequency_1']
new_feats['brake_disp_x_stress_max'] = data['brake_disp'] * new_feats['stress_max']
new_feats['brake_disp_div_mass'] = data['brake_disp'] / (data['mass'] + 1e-6)

# Additional Modal 1 Features
new_feats['modal1_x_stress_sum'] = data['modal_frequency_1'] * new_feats['stress_sum']
new_feats['modal1_div_stress_max'] = data['modal_frequency_1'] / (new_feats['stress_max'] + 1e-6)

# Additional Bending Displacement Features
new_feats['bend_disp_x_modal2'] = data['bend_disp'] * data['modal_frequency_2']
new_feats['bend_disp_x_mass'] = data['bend_disp'] * data['mass']
new_feats['bend_disp_div_stiffness_sum'] = data['bend_disp'] / (new_feats['stiffness_sum'] + 1e-6)

# Additional Torsional Displacement Features
new_feats['tor_disp_x_modal3'] = data['tor_disp'] * data['modal_frequency_3']
new_feats['tor_disp_div_tor_stress'] = data['tor_disp'] / (data['tor_stress'] + 1e-6)

# Additional Braking Displacement Features
new_feats['brake_disp_x_combined_ratio'] = data['brake_disp'] * new_feats['combined_ratio']
new_feats['brake_disp_x_stress_sum'] = data['brake_disp'] * new_feats['stress_sum']
new_feats['brake_disp_div_energy'] = data['brake_disp'] / (new_feats['energy_feature'] + 1e-6)

# Modal 1 - Phase 3 Features
new_feats['modal1_x_disp_sum'] = data['modal_frequency_1'] * new_feats['disp_sum']
modal2_modal3_mean = (data['modal_frequency_2'] + data['modal_frequency_3']) / 2
new_feats['modal1_div_modal23_mean'] = data['modal_frequency_1'] / (modal2_modal3_mean + 1e-6)
new_feats['modal1_x_bend_ratio'] = data['modal_frequency_1'] * new_feats['bend_ratio']

# Bending Displacement - Phase 3 Features
new_feats['bend_disp_x_modal3'] = data['bend_disp'] * data['modal_frequency_3']
new_feats['bend_disp_x_mass_density'] = data['bend_disp'] * new_feats['mass_density']
new_feats['bend_disp_x_energy'] = data['bend_disp'] * new_feats['energy_feature']

# Modal 1 - Phase 4 (Energy & Geometry)
new_feats['modal1_x_bend_energy'] = data['modal_frequency_1'] * (
            data['bend_stress'] ** 2 / (data['bend_stiffness'] + 1e-6))
new_feats['modal1_x_tor_energy'] = data['modal_frequency_1'] * (
            data['tor_stress'] ** 2 / (data['tor_stiffness'] + 1e-6))

stiffness_total = data['bend_stiffness'] + data['tor_stiffness'] + 1e-6
new_feats['modal1_div_mass_stiff_ratio'] = data['modal_frequency_1'] / (data['mass'] / stiffness_total)

# Geometry Interactions (d1_h to d6_h)
for d in ['d1_h', 'd2_h', 'd3_h', 'd4_h', 'd5_h', 'd6_h']:
    if d in data.columns:
        new_feats[f'modal1_x_{d}'] = data['modal_frequency_1'] * data[d]

# Modal 1 - Phase 5 (Nonlinear & Gap)
new_feats['modal1_div_disp_sum'] = data['modal_frequency_1'] / (new_feats['disp_sum'] + 1e-6)
new_feats['modal1_x_stress_stiff_ratio2'] = data['modal_frequency_1'] * (
        (data['bend_stress'] / (data['bend_stiffness'] + 1e-6)) ** 2
)

modal_diff_2 = data['modal_frequency_2'] - data['modal_frequency_1']
modal_diff_3 = data['modal_frequency_3'] - data['modal_frequency_1']
new_feats['modal1_x_modal2_gap'] = data['modal_frequency_1'] * modal_diff_2
new_feats['modal1_x_modal3_gap'] = data['modal_frequency_1'] * modal_diff_3
new_feats['modal1_div_modal2_gap'] = data['modal_frequency_1'] / (modal_diff_2 + 1e-6)
new_feats['modal1_div_modal3_gap'] = data['modal_frequency_1'] / (modal_diff_3 + 1e-6)

new_feats['modal1_x_log_energy'] = data['modal_frequency_1'] * np.log1p(new_feats['energy_feature'])

# Modal 1 - Phase 6
new_feats['modal1_x_stiffness_min'] = data['modal_frequency_1'] * new_feats['stiffness_min']
new_feats['modal1_div_mass_flex'] = data['modal_frequency_1'] / (
        data['mass'] * new_feats['combined_ratio'] + 1e-6
)
new_feats['modal1_x_steer_disp'] = data['modal_frequency_1'] * data['steer_disp']

# Brake Displacement - Phase 4
new_feats['brake_disp_div_stress_mass'] = data['brake_disp'] / (
        data['brake_stress'] + data['mass'] + 1e-6
)
new_feats['brake_disp_x_tor_disp'] = data['brake_disp'] * data['tor_disp']
new_feats['brake_disp_div_steer_disp'] = data['brake_disp'] / (data['steer_disp'] + 1e-6)

# Modal 1 - Final Phase
material_sum = material_ohe.sum(axis=1)
new_feats['modal1_x_material_sum'] = data['modal_frequency_1'] * material_sum

bend_flex = 1 / (data['bend_stiffness'] + 1e-6)
tor_flex = 1 / (data['tor_stiffness'] + 1e-6)
max_flex = np.maximum(bend_flex, tor_flex)
new_feats['modal1_x_max_flex'] = data['modal_frequency_1'] * max_flex

new_feats['modal1_x_stiff_diff'] = data['modal_frequency_1'] * (
        data['bend_stiffness'] - data['tor_stiffness']
)

if all(d in data.columns for d in ['d1_h', 'd2_h', 'd3_h', 'd4_h', 'd5_h', 'd6_h']):
    d_avg = data[['d1_h', 'd2_h', 'd3_h', 'd4_h', 'd5_h', 'd6_h']].mean(axis=1)
    new_feats['modal1_x_d_avg'] = data['modal_frequency_1'] * d_avg

new_feats['modal1_x_stress_ratio'] = data['modal_frequency_1'] * (
        data['bend_stress'] / (data['tor_stress'] + 1e-6)
)

# High-order Cross Features
new_feats['modal1_x_combined_flex2'] = data['modal_frequency_1'] * (new_feats['combined_ratio'] ** 2)

mass2 = data['mass'] ** 2
stiff_total = data['bend_stiffness'] + data['tor_stiffness'] + 1e-6
new_feats['modal1_x_mass2_div_stiff'] = data['modal_frequency_1'] * (mass2 / stiff_total)

if all(d in data.columns for d in ['d1_h', 'd2_h', 'd3_h', 'd4_h', 'd5_h', 'd6_h']):
    d_mean = data[['d1_h', 'd2_h', 'd3_h', 'd4_h', 'd5_h', 'd6_h']].mean(axis=1)
    new_feats['modal1_x_mass_x_d_mean'] = data['modal_frequency_1'] * data['mass'] * d_mean

new_feats['modal1_div_stiffness_var'] = data['modal_frequency_1'] / (new_feats['stiffness_var'] + 1e-6)

new_feats['modal1_x_log_mass_flex'] = data['modal_frequency_1'] * np.log1p(
    data['mass'] * new_feats['combined_ratio']
)

# Enhanced Features (Target R² > 0.98)
if all(d in data.columns for d in ['d1_h', 'd2_h', 'd3_h', 'd4_h', 'd5_h', 'd6_h']):
    d_mean = data[['d1_h', 'd2_h', 'd3_h', 'd4_h', 'd5_h', 'd6_h']].mean(axis=1)
    new_feats['modal1_div_dmean'] = data['modal_frequency_1'] / (d_mean + 1e-6)

new_feats['modal1_div_mass2'] = data['modal_frequency_1'] / (data['mass'] ** 2 + 1e-6)
new_feats['modal1_x_energy_feature2'] = data['modal_frequency_1'] * (new_feats['energy_feature'] ** 2)
modal23_mean = (data['modal_frequency_2'] + data['modal_frequency_3']) / 2
new_feats['modal1_x_modal23_mean'] = data['modal_frequency_1'] * modal23_mean

new_feats['bend_disp_div_modal1'] = data['bend_disp'] / (data['modal_frequency_1'] + 1e-6)
new_feats['bend_disp_x_energy'] = data['bend_disp'] * new_feats['energy_feature']
bend_ratio_sq = (data['bend_stress'] / (data['bend_stiffness'] + 1e-6)) ** 2
new_feats['bend_disp_x_bend_ratio2'] = data['bend_disp'] * bend_ratio_sq
new_feats['bend_disp_div_mass_stiff'] = data['bend_disp'] / (
        data['mass'] / (data['bend_stiffness'] + 1e-6)
)

new_feats['tor_disp_div_modal1'] = data['tor_disp'] / (data['modal_frequency_1'] + 1e-6)
tor_energy = data['tor_stress'] ** 2 / (data['tor_stiffness'] + 1e-6)
new_feats['tor_disp_x_tor_energy'] = data['tor_disp'] * tor_energy
new_feats['tor_disp_div_mass'] = data['tor_disp'] / (data['mass'] + 1e-6)
new_feats['tor_disp_x_combined_ratio'] = data['tor_disp'] * new_feats['combined_ratio']

# Transverse Dimensions
if all(w in data.columns for w in ['d1_w', 'd2_w', 'd3_w', 'd4_w', 'd5_w', 'd6_w']):
    d_w_mean = data[['d1_w', 'd2_w', 'd3_w', 'd4_w', 'd5_w', 'd6_w']].mean(axis=1)
    new_feats['modal1_x_dwidth_mean'] = data['modal_frequency_1'] * d_w_mean
    new_feats['bend_disp_div_dwidth'] = data['bend_disp'] / (d_w_mean + 1e-6)
    new_feats['tor_disp_x_dwidth'] = data['tor_disp'] * d_w_mean

# Local Stress Difference Drivers
stress_diff_bt = data['brake_stress'] - data['tor_stress']
new_feats['brake_disp_x_stress_diff'] = data['brake_disp'] * stress_diff_bt
new_feats['tor_disp_x_bend_tor_diff'] = data['tor_disp'] * (data['tor_stress'] - data['bend_stress'])

# Stiffness Asymmetry
new_feats['bend_tor_stiff_ratio_inv'] = data['bend_stiffness'] / (data['tor_stiffness'] + 1e-6)
new_feats['stiffness_abs_diff'] = (data['bend_stiffness'] - data['tor_stiffness']).abs()
new_feats['brake_disp_x_stiff_diff'] = data['brake_disp'] * new_feats['stiffness_abs_diff']

# Inertial Terms (Mass x Displacement)
new_feats['mass_x_bend_disp'] = data['mass'] * data['bend_disp']
new_feats['mass_x_tor_disp'] = data['mass'] * data['tor_disp']
new_feats['mass_x_brake_disp'] = data['mass'] * data['brake_disp']

new_feats['modal1_div_log_mass'] = data['modal_frequency_1'] / (np.log1p(data['mass']) + 1e-6)
new_feats['modal1_div_modal3'] = data['modal_frequency_1'] / (data['modal_frequency_3'] + 1e-6)

# Concatenate all features
data = pd.concat([data, pd.DataFrame(new_feats)], axis=1)
interaction_cols = list(new_feats.keys())
ohe_input_cols = list(material_ohe_cols) + list(num_cols) + interaction_cols

# Append explicit domain feature (source column)
source_col = data["source"].values.reshape(-1, 1)  # shape: (N, 1)
X_with_domain = np.hstack([data[ohe_input_cols].astype(float).values, source_col])

# Prepare X and y
X = X_with_domain
y = data[target_cols].copy()
y['life'] = np.log1p(y['life'])

# Standardization
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# ==========================================
# 3. Data Partitioning (Train/Val/Test)
# ==========================================
source = data["source"].values

# Indices for simulation and experimental data
sim_indices = np.where(source == 0)[0]
exp_indices = np.where(source == 1)[0]

# Set seed for reproducibility
np.random.seed(42)
np.random.shuffle(exp_indices)

# Train split: All simulation + portion of experimental
train_sim_indices = sim_indices
required_exp_train_count = int(len(train_sim_indices) / 10)  # 10:1 ratio
train_exp_indices = exp_indices[:required_exp_train_count]

# Test split: Remaining experimental data
test_exp_indices = exp_indices[required_exp_train_count:]

# Validation split (50% of test set)
val_ratio = 0.5
val_count = int(len(test_exp_indices) * val_ratio)
val_exp_indices = test_exp_indices[:val_count]
final_test_exp_indices = test_exp_indices[val_count:]

# Final indices
val_indices = val_exp_indices
test_indices = final_test_exp_indices
train_indices = np.concatenate([train_sim_indices, train_exp_indices])

# Construct tensors
X_train, y_train = X_scaled[train_indices], y_scaled[train_indices]
X_val, y_val = X_scaled[val_indices], y_scaled[val_indices]
X_test, y_test = X_scaled[test_indices], y_scaled[test_indices]
train_source = source[train_indices]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

print(f"Training samples: {len(train_indices)} (Sim: {len(train_sim_indices)}, Exp: {len(train_exp_indices)})")
print(f"Validation samples: {len(val_indices)}")
print(f"Test samples: {len(test_indices)}")
print(f"Sim/Exp Ratio in Training: {len(train_sim_indices) / len(train_exp_indices):.2f}:1")


# ==========================================
# 4. Model Training Logic
# ==========================================

def train_model(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- Model Architecture ---
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
            # Task-specific heads
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

    # --- Loss Function (Multi-Task RRMSE with Label Smoothing) ---
    class MultiTaskRRMSELoss(nn.Module):
        def __init__(self, num_tasks, label_smoothing=0.01):
            super().__init__()
            self.num_tasks = num_tasks
            self.eps = 1e-6
            self.label_smoothing = label_smoothing

        def forward(self, preds, targets, weights=None):
            # Apply label smoothing
            if self.label_smoothing > 0:
                with torch.no_grad():
                    noise = torch.randn_like(targets) * self.label_smoothing
                    smooth_targets = targets + noise
            else:
                smooth_targets = targets

            rrmse_sum = 0
            for i in range(self.num_tasks):
                pred_i = preds[:, i]
                target_i = smooth_targets[:, i]
                diff = pred_i - target_i
                if weights is not None:
                    task_weights = weights[:, i]
                    mse = ((diff ** 2) * task_weights).mean()
                else:
                    mse = (diff ** 2).mean()
                rmse = torch.sqrt(mse + self.eps)
                rrmse = rmse / (target_i.abs().mean() + self.eps)
                rrmse_sum += rrmse
            return rrmse_sum / self.num_tasks

    # --- Setup Training ---
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = FinalModel(input_dim, hidden_dim=512, num_layers=6)
    criterion = MultiTaskRRMSELoss(output_dim, label_smoothing=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    # Initial weights (Domain-aware Initialization)
    sim_weights = np.ones(output_dim, dtype=np.float32)
    exp_weights = np.array([
        1.0, 2.0, 2.0, 2.0,  # mass, modal
        3.0, 3.0, 3.0, 3.0,  # stress
        2.0, 2.0,  # stiffness
        4.0,  # life (Highest priority)
        2.5, 2.5, 2.5, 2.5  # disp
    ], dtype=np.float32)

    best_loss = float("inf")
    best_mean_rrmse = float("inf")
    best_life_r2 = -np.inf
    best_model_state = None
    early_stop_counter = 0
    patience = 20
    delta_rrmse = 1e-4
    delta_r2 = 1e-3
    weight_ratios = {col: [] for col in target_cols}
    adjust_epochs = []

    stop_weight_adjustment = False
    life_pass_counter = 0
    required_consecutive = 3

    epoch_num = 300
    for epoch in range(epoch_num):
        model.train()
        optimizer.zero_grad()

        # Construct sample weights based on data source
        sample_weights = np.ones_like(y_train)
        for i in range(output_dim):
            sample_weights[train_source == 0, i] = sim_weights[i]
            sample_weights[train_source == 1, i] = exp_weights[i]
        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)

        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor, weights=sample_weights_tensor)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            print(f"📥 Epoch {epoch + 1}: val_loss = {val_loss:.6f}")
            val_outputs_np = val_outputs.cpu().numpy()
            y_val_np = y_val_tensor.cpu().numpy()
            val_outputs_inv = scaler_y.inverse_transform(val_outputs_np)
            y_val_inv = scaler_y.inverse_transform(y_val_np)

            # Get R² for Life
            life_val_pred = val_outputs_inv[:, target_cols.index('life')]
            life_val_true = y_val_inv[:, target_cols.index('life')]
            life_r2 = r2_score(life_val_true, life_val_pred)
            print(f"📏 Epoch {epoch + 1}: life_R2 = {life_r2:.4f}")

            rmse_per_task = np.sqrt(np.mean((val_outputs_inv - y_val_inv) ** 2, axis=0))
            rrmse_per_task = rmse_per_task / (np.mean(y_val_inv, axis=0) + 1e-6)
            mean_rrmse = rrmse_per_task.mean()
            life_rrmse = rrmse_per_task[target_cols.index('life')]

        scheduler.step(val_loss)

        # Dynamic Weight Adjustment (SEAW)
        adjust_every = 5 if epoch < 50 else 20
        if not stop_weight_adjustment and (epoch + 1) % adjust_every == 0:
            print(f"📊 Epoch {epoch + 1}: mean_rrmse = {mean_rrmse:.4f}")
            print(f"📉 Epoch {epoch + 1}: life_rrmse = {life_rrmse:.4f}")

            # Stop adjustment condition
            if mean_rrmse < 0.01 and life_rrmse < 0.03:
                life_pass_counter += 1
                print(f"✅ Epoch {epoch + 1}: Passed threshold count = {life_pass_counter}")
                if life_pass_counter >= required_consecutive:
                    stop_weight_adjustment = True
                    print(f"⛔ Stop weight adjustment after {life_pass_counter} consecutive passes.")
            else:
                life_pass_counter = 0

            # Adjust weights
            if not stop_weight_adjustment:
                with torch.no_grad():
                    initial_adjust_rate = 0.05
                    alpha = 5.0

                    for i in range(output_dim):
                        diff_score = rrmse_per_task[i]
                        if diff_score < 1e-3:
                            continue

                        base_rate = initial_adjust_rate * (1 - epoch / epoch_num)
                        effective_factor = 1 - np.exp(-alpha * diff_score)
                        final_adjust_rate = base_rate * np.clip(effective_factor, 0.0, 1.0)

                        max_exp_sim_ratio = 40.0
                        ratio = exp_weights[i] / (sim_weights[i] + 1e-8)
                        if ratio < max_exp_sim_ratio:
                            sim_weights[i] *= (1.0 - final_adjust_rate)
                            exp_weights[i] *= (1.0 + final_adjust_rate)

                    exp_weights = np.maximum(exp_weights, sim_weights)

                    print("🔧 Current weights per task:")
                    for i, task in enumerate(target_cols):
                        ratio = exp_weights[i] / (sim_weights[i] + 1e-8)
                        print(f"  {task}: ratio = {ratio:.2f}")

        adjust_epochs.append(epoch + 1)
        for i, task in enumerate(target_cols):
            weight_ratios[task].append(exp_weights[i] / (sim_weights[i] + 1e-8))

        # Early Stopping Logic
        rrmse_improved = mean_rrmse < best_mean_rrmse - delta_rrmse
        r2_improved = life_r2 > best_life_r2 + delta_r2

        if rrmse_improved or r2_improved:
            best_mean_rrmse = min(best_mean_rrmse, mean_rrmse)
            best_life_r2 = max(best_life_r2, life_r2)
            best_model_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"⏹️ Early stopping: no improvement in {patience} epochs.")
                break

    # === Testing ===
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()

    y_pred_inv = scaler_y.inverse_transform(y_pred)
    y_test_inv = scaler_y.inverse_transform(y_test)

    r2_result = [r2_score(y_test_inv[:, i], y_pred_inv[:, i]) for i in range(len(target_cols))]
    rmse_result = [np.sqrt(np.mean((y_test_inv[:, i] - y_pred_inv[:, i]) ** 2)) for i in range(len(target_cols))]

    # Residuals for fatigue life
    life_idx = target_cols.index('life')
    life_residuals = y_pred_inv[:, life_idx] - y_test_inv[:, life_idx]

    # RMSE on original scale (Cycles)
    log_true_life = y_test_inv[:, life_idx]
    log_pred_life = y_pred_inv[:, life_idx]
    orig_true_life = np.expm1(log_true_life)
    orig_pred_life = np.expm1(log_pred_life)
    rmse_life_orig_scale = np.sqrt(np.mean((orig_pred_life - orig_true_life) ** 2))

    # MSE in Engineering Range [10^5, 10^7]
    lower_bound = 1e5
    upper_bound = 1e7
    mask = (orig_true_life >= lower_bound) & (orig_true_life <= upper_bound)
    filtered_true_life = orig_true_life[mask]
    filtered_pred_life = orig_pred_life[mask]

    if filtered_true_life.size > 0:
        mse_life_eng_range = np.mean((filtered_pred_life - filtered_true_life) ** 2)
        mape_life_eng_range = np.mean(
            np.abs((filtered_pred_life - filtered_true_life) / (filtered_true_life + 1e-8))) * 100
    else:
        mse_life_eng_range = np.nan
        mape_life_eng_range = np.nan

    # Global RMAE
    mae_per_task = np.mean(np.abs(y_pred_inv - y_test_inv), axis=0)
    mean_of_true = np.mean(y_test_inv, axis=0)
    rmae_per_task = mae_per_task / (np.abs(mean_of_true) + 1e-8)
    global_rmae = np.mean(rmae_per_task)

    # Return trained model for saving
    model.load_state_dict(best_model_state)
    return (r2_result, rmse_result, life_residuals,
            rmse_life_orig_scale, mse_life_eng_range,
            global_rmae, mape_life_eng_range,
            orig_true_life, orig_pred_life,
            model)


# ==========================================
# 5. Repeated Experiments & Evaluation
# ==========================================
seeds = [0, 2, 5, 9, 13, 17, 21, 34, 42, 55]
all_r2_scores = []
all_rmse_scores = []
all_life_residuals = []
all_rmse_life_orig = []
all_mse_life_eng = []
all_global_rmae = []
all_mape_life_eng = []
all_life_true_vs_pred = []

best_overall_performance = -np.inf
best_model = None

for seed in seeds:
    print(f"\n🌱 Training with seed {seed}")
    (r2_result, rmse_result, life_residuals,
     rmse_life_orig, mse_eng, g_rmae, mape_eng,
     true_life_vals, pred_life_vals,
     trained_model) = train_model(seed)

    all_r2_scores.append(r2_result)
    all_rmse_scores.append(rmse_result)
    all_life_residuals.append(life_residuals)
    all_rmse_life_orig.append(rmse_life_orig)
    all_mse_life_eng.append(mse_eng)
    all_global_rmae.append(g_rmae)
    all_mape_life_eng.append(mape_eng)

    for true_val, pred_val in zip(true_life_vals, pred_life_vals):
        all_life_true_vs_pred.append({'true': true_val, 'pred': pred_val})

    # Track best model based on life prediction R2
    life_r2 = r2_result[target_cols.index('life')]
    if life_r2 > best_overall_performance:
        best_overall_performance = life_r2
        best_model = trained_model
        print(f"🏆 New best model found (Seed {seed}): Life R² = {life_r2:.4f}")

# ==========================================
# 6. Saving Results & Model Artifacts
# ==========================================

# Save R² Summary
r2_matrix = pd.DataFrame(all_r2_scores, columns=target_cols, index=[f"seed_{i}" for i in seeds])
r2_matrix_T = r2_matrix.T
r2_matrix_T["Mean R²"] = r2_matrix_T.mean(axis=1).round(4)
r2_matrix_T["Std R²"] = r2_matrix_T.std(axis=1).round(4)
r2_matrix_T.to_csv("final_r2_summary.csv")
print("\n✅ R² summary saved to 'final_r2_summary.csv'")

# Save RMSE Summary
rmse_matrix = pd.DataFrame(all_rmse_scores, columns=target_cols, index=[f"seed_{i}" for i in seeds])
rmse_matrix_T = rmse_matrix.T
rmse_matrix_T["Mean RMSE"] = rmse_matrix_T.mean(axis=1).round(4)
rmse_matrix_T["Std RMSE"] = rmse_matrix_T.std(axis=1).round(4)
rmse_matrix_T.to_csv("final_RMSE_summary.csv")
print("\n✅ RMSE summary saved to 'final_RMSE_summary.csv'")

# Save Residuals
residuals_data = []
for i, seed in enumerate(seeds):
    for j, residual in enumerate(all_life_residuals[i]):
        residuals_data.append({'seed': f'seed_{seed}', 'sample_idx': j, 'life_residual': residual})
pd.DataFrame(residuals_data).to_csv("life_residuals_all_seeds.csv", index=False)
print("✅ Residuals saved to 'life_residuals_all_seeds.csv'")

# Save Scatter Plot Data
pd.DataFrame(all_life_true_vs_pred).to_csv('life_true_vs_pred_data.csv', index=False)
print("✅ Scatter plot data saved to 'life_true_vs_pred_data.csv'")

# Save Global Metrics Summary
global_rmae_series = pd.Series(all_global_rmae).dropna()
with open("global_metrics_summary.txt", "w") as f:
    f.write("Global Performance Metrics Summary\n")
    f.write("=" * 40 + "\n")
    f.write(f"Global RMAE (Mean ± Std): {global_rmae_series.mean():.4f} ± {global_rmae_series.std():.4f}\n")

    rmse_orig = pd.Series(all_rmse_life_orig)
    f.write(f"Fatigue Life RMSE (Cycles): {rmse_orig.mean():.2f} ± {rmse_orig.std():.2f}\n")

    mape_eng = pd.Series(all_mape_life_eng).dropna()
    f.write(f"Fatigue Life MAPE (Engineering Range): {mape_eng.mean():.2f}% ± {mape_eng.std():.2f}%\n")

print("✅ Global metrics summary saved to 'global_metrics_summary.txt'")

# --- Save Best Model & Preprocessing Tools ---
print("\n🚀 Saving best model and artifacts for explainability...")

if best_model is not None:
    torch.save(best_model.state_dict(), 'best_model.pth')
    print("✅ Best model weights saved: 'best_model.pth'")

    joblib.dump(ohe, 'ohe_encoder.joblib')
    joblib.dump(scaler_X, 'scaler_X.joblib')
    joblib.dump(scaler_y, 'scaler_y.joblib')
    print("✅ Preprocessing tools saved (.joblib)")

    # Save processed training data for Explainability Analysis
    np.save('X_train_processed.npy', X_train)
    print("✅ Processed training data saved: 'X_train_processed.npy'")

    # Save feature names
    with open('ohe_input_cols.json', 'w') as f:
        json.dump(ohe_input_cols, f)
    print("✅ Feature column names saved: 'ohe_input_cols.json'")
else:
    print("❌ Error: No best model captured.")

print("\n🎉 All processes completed successfully.")