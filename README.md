# Responsible & Explainable AI: Toxicity Detection & Fairness Mitigation

A comprehensive implementation of fair, robust, and explainable AI for toxicity detection in text. This project combines adversarial robustness testing, fairness audits, bias mitigation techniques, and a three-layer moderation pipeline.

## Project Overview

This assignment demonstrates best practices in responsible AI:
1. **Baseline Model**: DistilBERT-based toxicity classifier
2. **Bias Audit**: Cohort-level fairness analysis
3. **Adversarial Testing**: Evasion and poisoning attack evaluation
4. **Bias Mitigation**: Three post-hoc techniques with Pareto frontier analysis
5. **Production Pipeline**: Three-layer guardrail system with regex filtering, calibrated scoring, and human review queue

---

## Project Structure

### Part 1: Baseline Toxicity Classifier (`part1.ipynb`)
- **Goal**: Train a baseline DistilBERT sequence classifier on the Jigsaw Unintended Bias dataset
- **Key Components**:
  - Data loading and stratified train/eval split (100K train, 20K eval)
  - DistilBERT fine-tuning with HuggingFace Trainer
  - Evaluation metrics: accuracy, F1, confusion matrix
- **Output**: `part1_baseline/` model checkpoint
- **Dataset**: [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification)

### Part 2: Bias Audit (`part2.ipynb`)
- **Goal**: Audit baseline fairness across demographic cohorts
- **Key Components**:
  - Cohort construction: high-black (black ≥ 0.5), reference (black < 0.1 ∧ white ≥ 0.5)
  - Fairness metrics: Statistical Parity Difference (SPD), Equal Opportunity Difference (EOD)
  - Performance gaps: FPR, TPR, False Negative Rate (FNR) analysis
- **Findings**: Baseline exhibits significant disparities (SPD ~0.124, EOD ~0.192)

### Part 3: Adversarial Robustness (`part3.ipynb`)
- **Goal**: Evaluate vulnerability to adversarial text perturbations and data poisoning
- **Evasion Attack**:
  - Techniques: zero-width space insertion, unicode homoglyph substitution, character duplication
  - Attack Success Rate: 96% on 500 high-confidence toxic comments
  - Confidence drop: 0.898 → 0.040
- **Poisoning Attack**:
  - Label flip: 5% of training data flipped
  - Effect: Modest accuracy/F1 drop, increased false negatives
  - Key insight: Evasion attacks are more operationally dangerous than poisoning
- **Takeaway**: Model needs stronger robustness against obfuscation

### Part 4: Bias Mitigation & Pareto Frontier (`part4.ipynb`)
- **Goal**: Apply three mitigation techniques and analyze accuracy-fairness trade-offs
- **Techniques**:
  1. **Reweighing** (AIF360): Adjust training sample weights to balance selection rates
  2. **Threshold Optimization** (fairlearn): Post-process predictions with per-group thresholds
  3. **Oversampling**: Oversample high-black toxic examples 3x
- **Results**:
  - Baseline: F1=0.784, SPD=0.124, EOD=0.192
  - Best (Threshold Opt): F1=0.789, SPD=0.031, EOD=0.007
- **Pareto Frontier**: Dense 91×91 threshold grid showing trade-off between fairness and accuracy
- **Key Finding**: Cannot simultaneously achieve demographic parity AND equalized odds due to unequal base rates (0.334 vs 0.280)

### Part 5: Production Moderation Pipeline (`part5.ipynb` + `pipeline.py`)
- **Goal**: Deploy a practical three-layer guardrail system
- **Architecture**:
  - **Layer 1 - Input Filter**: Fast regex-based pre-filter (20+ patterns across 5 categories)
  - **Layer 2 - Calibrated Model**: Probabilistic toxicity scoring with isotonic calibration
  - **Layer 3 - Review Queue**: Human review for uncertain cases (confidence 0.4–0.6)
- **Categories Filtered**:
  - Direct threats of violence (5 patterns)
  - Calls for self-harm/suicide (4 patterns)
  - Doxxing & stalking threats (4 patterns)
  - Severe dehumanization (4 patterns)
  - Coordinated harassment signals (3+ patterns)
- **Evaluation** (1,000 demo examples):
  - Layer distribution: 97.5% auto-action, 2.5% review queue
  - Auto-action F1: 0.5312, Precision: 0.7727, Recall: 0.4048
  - Review-queue composition: 56% toxic, 44% non-toxic
  - Threshold-band sensitivity analysis: (0.4–0.6), (0.45–0.55), (0.3–0.7)

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- GPU (recommended for model training)
- 10GB+ free disk space

### Install Dependencies
```bash
pip install -r requirements.txt
```

Key packages:
- `transformers==4.44.2` — HuggingFace model loading/training
- `torch==2.4.1` — PyTorch backend
- `scikit-learn==1.5.1` — Metrics, calibration, utilities
- `fairlearn==0.10.0` — Threshold optimization
- `aif360==0.6.1` — Reweighing, fairness metrics
- `pandas==2.2.2`, `numpy==1.26.4` — Data handling
- `matplotlib==3.9.2`, `seaborn==0.13.2` — Visualization

---

## Running the Project

### Option 1: Google Colab (Recommended)
1. Upload all `.ipynb` files and data CSVs to Colab
2. Run notebooks **sequentially**: Part 1 → 2 → 3 → 4 → 5
3. Models and artifacts auto-save to `/content/saved_model/`

### Option 2: Local / Jupyter Lab
```bash
# Configure Python environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Start Jupyter
jupyter lab

# Open and run notebooks in order
```

### Order of Execution
1. **part1.ipynb** — Trains baseline model (takes ~15 min on GPU)
2. **part2.ipynb** — Audits baseline fairness
3. **part3.ipynb** — Tests adversarial robustness
4. **part4.ipynb** — Applies mitigation techniques & exports model artifacts
5. **part5.ipynb** — Deploys moderation pipeline

---

## Key Files & Outputs

### Input Data
- `jigsaw-unintended-bias-train.csv` — Main training dataset (~1.8M rows)
- `validation.csv` — Optional validation split

### Code Files
- `part1.ipynb` — Baseline training
- `part2.ipynb` — Fairness audit
- `part3.ipynb` — Adversarial robustness
- `part4.ipynb` — Mitigation & Pareto frontier
- `part5.ipynb` — Production pipeline demo
- `pipeline.py` — Reusable ModerationPipeline class

### Generated Artifacts (in `saved_model/` or `outputs/`)
```
saved_model/
├── part1_baseline/           # DistilBERT baseline checkpoint
├── part4_mitigations/
│   ├── best_mitigated/       # Best mitigated model + artifacts
│   ├── reweighing_model/     # Reweighed DistilBERT
│   ├── oversampling_model/   # Oversampled DistilBERT
│   └── threshold_optimizer.joblib
└── part1_distilbert/         # Alternative checkpoint

outputs/
└── part{N}_{technique}/      # Training logs & checkpoints
```

---

## Fairness & Accountability

### Fairness Definitions
- **Statistical Parity (DP)**: Equal selection rates across groups
- **Equal Opportunity (EO)**: Equal true positive rates across groups
- **Why Both Impossible**: With unequal base rates, satisfying both exactly requires a trivial classifier

### Base Rates (This Dataset)
| Cohort | P(Y=1) |
|--------|--------|
| High-black (black ≥ 0.5) | 0.3338 |
| Reference (black < 0.1 ∧ white ≥ 0.5) | 0.2798 |
| **Difference** | **0.0540** |

This difference makes simultaneous perfect fairness impossible for any useful model.

### Mitigation Trade-offs
| Technique | F1 | SPD | EOD |
|-----------|-----|-----|-----|
| Baseline | 0.784 | 0.124 | 0.192 |
| Reweighing | 0.783 | 0.035 | 0.059 |
| Threshold Opt | 0.789 | 0.031 | **0.007** |
| Oversampling | 0.783 | 0.034 | 0.059 |

**Best Choice**: Threshold optimization balances fairness gains with minimal accuracy loss.

---

## Results Summary

### Part 3: Adversarial Robustness
- **Evasion (Perturbation)**: 96% success rate; confidence drops from 0.898 to 0.040
- **Poisoning**: 5% label flip causes ~0.3% accuracy loss, +0.8% false-negative rate increase
- **Conclusion**: Evasion attacks are the primary threat for live systems

### Part 4: Bias Mitigation
- **Pareto Frontier**: Dense grid search (91×91 thresholds) reveals smooth trade-off curve
- **Best Point**: Threshold optimization at EOD≈0.007 with F1≈0.789
- **Impossibility Result**: Cannot achieve SPD=0 and EOD=0 simultaneously due to base-rate difference

### Part 5: Production Pipeline
- **3-Layer Efficiency**: Regex filter blocks ~0% of test cases; model handles 97.5% automatically
- **Review Queue**: 2.5% of decisions routed to humans for manual review
- **Quality Metrics**:
  - Auto-action F1: 0.5312 (precision 0.7727 favors blocking suspicious content)
  - Review-queue toxic rate: 56% (indicates well-calibrated uncertainty zones)

---

## Architecture Decisions

### Why 3-Layer Pipeline?
1. **Layer 1 (Regex)**: Catches obvious toxic patterns with zero model latency
2. **Layer 2 (Calibrated Model)**: Probabilistic scoring with isotonic calibration ensures confidence ≈ accuracy
3. **Layer 3 (Review Queue)**: Humans handle ambiguous cases; reduces both false positives and negatives

### Isotonic Calibration
- Applied via `CalibratedClassifierCV(method='isotonic')` on 5,000 holdout examples
- Ensures model confidence scores correspond to true accuracy rates
- Essential for reliable decision thresholds (0.4–0.6 confidence band)

### Threshold Optimization Strategy
- Tested three uncertainty bands: (0.4–0.6), (0.45–0.55), (0.3–0.7)
- Default 0.4–0.6 balances review volume (~2.5%) with model confidence
- Tighter bands (0.45–0.55) increase auto-action but reduce review quality
- Looser bands (0.3–0.7) protect against model errors but burden reviewers

---

## How to Use the Pipeline in Production

### Quick Start
```python
from pipeline import ModerationPipeline

# Initialize with your best mitigated model
pipe = ModerationPipeline(
    model_dir='saved_model/part1_baseline',  # or part4_mitigations/best_mitigated
    lower_threshold=0.4,
    upper_threshold=0.6,
    device='cuda'  # or 'cpu'
)

# Fit calibrator on known-good examples
pipe.fit_calibrator(
    calibration_texts=[...],
    calibration_labels=[...],
    batch_size=64
)

# Make predictions
result = pipe.predict("This is a comment to moderate")
# {
#   'decision': 'block' | 'allow' | 'review',
#   'layer': 'input_filter' | 'model',
#   'confidence': 0.75,
#   'category': 'direct_threat' (if Layer 1) or 'none'
# }
```

### Adjusting Thresholds
```python
# Stricter: more to review
pipe.set_uncertainty_band(lower=0.45, upper=0.55)

# More permissive: more auto-blocking
pipe.set_uncertainty_band(lower=0.3, upper=0.7)
```

---

## Notable Findings

1. **Fairness Impossibility**: With unequal base rates, demographic parity and equalized odds are mathematically incompatible for non-trivial classifiers.

2. **Evasion >> Poisoning**: Text obfuscation attacks are far more operationally dangerous than training-time poisoning.

3. **Calibration Matters**: Raw model confidence scores do not map to true accuracy; isotonic calibration is essential for reliable decision thresholds.

4. **Human-in-the-Loop Works**: 2.5% of decisions routed to humans with 56% toxic rate in review queue suggests good uncertainty calibration.
