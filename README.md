# Releaf: Unsupervised Urban Tree Biodiversity Mapping

This repository implements the pipeline described in  
**"Unsupervised Urban Tree Biodiversity Mapping from Street-Level Imagery Using Spatially Aware Visual Clustering"**  
by Diaa Abuhani *et al.*

The framework enables large-scale, label-free biodiversity assessment directly from street-level imagery, combining **visual embeddings** from BioCLIP with **spatial embeddings** from TaxaBind. The approach clusters trees into taxonomically coherent groups and computes key biodiversity metrics.

---

## Overview

Urban tree biodiversity is crucial for climate resilience, ecological stability, and livability. Traditional mapping requires expert-led field surveys and taxonomic labels, limiting scalability.  
This framework addresses those limitations by:

- **Operating without ground-truth labels**
- **Fusing visual similarity and spatial priors**
- **Applying spatially aware clustering** (HDBSCAN + post-processing)
- **Computing ecological indicators** like Shannon entropy, Simpson index, and species richness

---

![WhatsApp Image 2025-06-30 at 3 10 52 PM](https://github.com/user-attachments/assets/4c68c337-17ae-4e7f-bc47-f7409aae6910)


## Features

- **Embedding Extraction**  
  - BioCLIP for visual tree features  
  - TaxaBind for spatial features (latitude/longitude)

- **Spatially Aware Clustering**  
  - Density-based clustering (HDBSCAN)  
  - Outlier elimination, cluster merging, and reassignment steps

- **Metric Computation**  
  - **Repro mode:** Shannon, Simpson, Richness, V-score (when ground truth available)  
  - **Deploy mode:** Shannon, Simpson, Richness (for new cities without GT)

- **Hyperparameter Optimization**  
  - Separate **Latin Hypercube Sampling (LHS)** search procedure  
  - Weighted scoring combining diversity, coverage, and accuracy metrics

---

## Repository Structure

configs/ # YAML configuration files for embeddings, evaluation, LHS
src/releaf_tuning/ # Main pipeline code
embeddings.py # TFRecord → NPZ embedding extraction
pipeline.py # Clustering and post-processing
metrics.py # Shannon, Simpson, Richness, V-score
postprocess.py # Outlier elimination & merging
evaluate.py # Repro & Deploy evaluation modes
scripts/ # Optional helper scripts


---
## Installation

```bash
# Clone the repository
git clone https://github.com/Diaa340/Releaf.git
cd releaf

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install package
pip install --upgrade pip
pip install -e .

# Install TaxaBind (if not on PyPI)
pip install git+https://github.com/MVRL/taxabind.git

# Install OpenCLIP for BioCLIP embeddings
pip install git+https://github.com/mlfoundations/open_clip.git

# (Optional) Install Jupyter for notebooks
pip install jupyter
```
## Quick Start

### 1. Extract Embeddings
Edit `configs/embeddings.yaml` with your base directory and city folders, then run:

```bash
releaf-embed --config configs/embeddings.yaml
```
<img width="1049" height="590" alt="download - 2025-08-11T025515 999" src="https://github.com/user-attachments/assets/495be337-a254-4a43-add5-61a41aed31e4" />

This will generate .npz files containing image and location embeddings for each city.


### 2.  Run Evaluation
With Ground Truth (Reproducibility mode):
```bash
releaf-eval --config configs/eval.yaml --mode repro
```
Without Ground Truth (Deploy mode for new cities):
```bash
releaf-eval --config configs/eval.yaml --mode deploy
```
Both modes output per-area and aggregated biodiversity metrics.

### 3. Hyperparameter Search (Optional)
Use Latin Hypercube Sampling (LHS) to explore the parameter space.
```bash
python -m releaf_tuning.lhs_search_procedure --config configs/default.yaml --samples 100
```
## Citation

If you use this repository, please cite our paper:

```bibtex
@misc{abuhani2025unsupervisedurbantreebiodiversity,
      title={Unsupervised Urban Tree Biodiversity Mapping from Street-Level Imagery Using Spatially-Aware Visual Clustering}, 
      author={Diaa Addeen Abuhani and Marco Seccaroni and Martina Mazzarello and Imran Zualkernan and Fabio Duarte and Carlo Ratti},
      year={2025},
      eprint={2508.13814},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.13814}, 
}
```








