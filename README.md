# Reproducing Experiments: Top-k and Ranking Stability

This repository contains code to reproduce the experiments from our paper on stable methods for top-$k$ selection and full ranking.

---

## 📄 Paper

- **Title:** "Assumption-free stability for ranking problems"
- **Authors:** Ruiting Liang, Jake A. Soloff, Rina Foygel Barber, Rebecca Willett
[comment]: <> (- **Link:** [Insert arXiv Link])

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📁 Directory Overview

```
topk-exp.py                  # Netflix top-k experiments + plotting
combine-raw-data.py          # Converts Netflix raw data to fullData.csv
full-ranking-experiment.py   # Synthetic full-ranking experiments
configs/                     # Optional: config files for experiments
results/                     # Optional: stores outputs/plots
```

---

## 🔁 Reproducing Experiments

### 1. Top-$k$ Experiments (Netflix Prize Data)

- **Script:** `topk-exp.py`  
- **Description:** Runs experiments on top-$k$ selection using the Netflix Prize dataset and generates corresponding appendix plots.

#### Dataset Setup

1. Download the [Netflix Prize Data from Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data).
2. Convert the raw files into a single `fullData.csv` file using:
   ```bash
   python combine-raw-data.py
   ```

---

### 2. Full Ranking Simulations (Synthetic Data)

- **Script:** `full-ranking-experiment.py`  
- **Description:** Runs synthetic experiments for full-list ranking stability.

---

## 📊 Results

- Running the scripts will generate outputs and plots corresponding to results in the paper.

---

[comment]: <> (## 📚 Citation

If you use this code, please cite the paper using:

```bibtex
@article{your_citation,
  title={...},
  author={...},
  journal={...},
  year={...}
}
```

---

## 📝 License

[Insert license name here, e.g., MIT License]
)