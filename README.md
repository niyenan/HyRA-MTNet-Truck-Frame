# Truck Frame Performance Prediction Code

This repository contains the source code for the paper "**Truck Frame Performance Prediction Using Adaptive Fusion of Multi-Source Data and Residual Attention-based Multi-Task Learning**".

Please run main.py first to train the model, then run explainability.py for SHAP analysis.

## ⚠️ Note on Data Privacy and Reproducibility

*   **Simulation Data**: The full simulation dataset is provided in the `data/` folder.
*   **Physical Test Data**: The real physical test data used for validation is proprietary to **China National Heavy Duty Truck Group Co., Ltd. (CNHTC)** and cannot be released due to commercial trade secrets.
*   **For Verification**: We provide a **dummy dataset** (`data/experiment_dummy.csv`) with the same structure as the confidential data. You can use this to run the code and verify the algorithm logic. 
*   **Note**: Since the experimental data is synthetic, the results generated from this repository will **not** match the accuracy metrics reported in the manuscript.
