📌 Repository contents
1. normalize_redox.py

Purpose:
Normalizes raw redox measurements (superoxide, nitrites, H₂O₂, GSH, GSSG) on a 0–1 scale per viable cell.
No machine learning — deterministic preprocessing.

Output: redox_data_normalized.csv

2. learn_weights_ai_v5.py

Purpose:
Learns feature weights using ElasticNetCV, PolynomialFeatures and GroupKFold validation.
This is the core AI model behind RDI_v5.

Output files:

ai_learned_weights_v5.csv

RDI_AI_v5_vs_Viability.png

feature_importance_v5_rf.png

ai_weight_distribution_v5.png

v5_cv_scores.txt

3. compute_rdi_v5.py

Purpose:
Generates the final RDI_v5 and RDI_v5_score_0_100 using:

normalized CSV

AI-derived weights

Features:

single + interaction terms

automatic orientation (higher RDI = lower viability)

robust scaling (percentiles 5–95)

quick-start mode

Output: redox_data_with_RDI_v5.csv

4. rdi_visual_v5.py

Purpose:
Visual verification of RDI_v5 results.

Generates:

histogram

RDI vs viability scatter (R² + Pearson)

time-course or concentration plots

5. scaler_v5.json

Stored means and stds used during development.

6. ai_learned_weights_v5.csv

Final machine-learned weights used for RDI_v5 computation.

📁 Suggested structure
RedoxAI-RDI/
│
├─ scripts/
│   (all python scripts + model files)
│
├─ data/
│   example_input.csv (optional)
│
├─ README.md
└─ requirements.txt

🐍 Python requirements

Python: 3.10+ recommended

Libraries:

numpy  
pandas  
scikit-learn  
matplotlib

▶️ Basic usage

Step 1 — normalize

python normalize_redox.py


Step 2 — learn weights (optional)

python learn_weights_ai_v5.py


Step 3 — compute RDI_v5

python compute_rdi_v5.py --input-csv redox_data_normalized.csv --weights-csv ai_learned_weights_v5.csv


Step 4 — visualize

python rdi_visual_v5.py --csv redox_data_with_RDI_v5.csv

📘 Citation

Please cite this software package as:

Živanović M. RedoxAI-RDI: Python toolkit for Redox Disturbance Index (RDI_v5). Zenodo (2025).
DOI: to be added after the first GitHub release

📄 License

MIT License

✉️ Contact

Dr. Marko Živanović

Redefining Biology for the Next Generation
