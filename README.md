# Context-Aware Recommender (Kuaishou)

A context-aware neural recommender system for short-video ranking, built on the KuaiRec dataset from the Kuaishou platform. It starts from hybrid models combining TimeSVD++ with k-nearest neighbours to leverage behavioural and metadata features, then extends both components to neural variants using trainable embeddings and an MLP to capture non-linear user–video relationships. Models are evaluated with Precision@K, NDCG and novelty to measure both ranking quality and recommendation diversity.

Dataset was too large to upload to github as there are images in the dataset.

### Complete Dataset To Run The Code
You can find the Dataset of this code and all things needed to run it at this google Drive link:
https://drive.google.com/drive/folders/1_xzdlUP93N51X65UMA_fw7zXX3RT6cny?usp=share_link
https://drive.google.com/drive/folders/1_xzdlUP93N51X65UMA_fw7zXX3RT6cny?usp=sharing

What you will need is really the `data-5` and `KuaiRec 2-2.0` file from this Google drive link.
Or you can just download the whole google drive and run it as it is from there.

### Runtime Scripts (3 files)
- `run_RS1.py` - Interactive CLI for RS1 (TimeSVD++ + KNN)
- `run_RS2.py` - Interactive CLI for RS2 (SASRec + KNN)
- `run_eval.py` - Comprehensive evaluation comparing both systems

### Model Implementations (2 files)
- `timesvdpp.py` - TimeSVD++ algorithm
- `sasrec_model.py` - SASRec transformer model

### Trained Models (2 files in `models/`)
- `hybrid_system_timesvdpp.pkl` - Pre-trained RS1 model
- `hybrid_sasrec_knn.pkl` - Pre-trained RS2 model

### Preprocessed Data (`KuaiRec 2-2.0/data/`)
**Only final preprocessed outputs - no raw data:**
- `collab_df.csv` - Collaborative filtering features (100K interactions)
- `content_df.csv` - Content-based features (3,327 videos)
- `content_df_cli.csv` - CLI-enhanced content data
- `content_features.npz` - TF-IDF sparse feature matrix
- `content_metadata.pkl` - Additional metadata
- `sasrec_preprocessed/` - Preprocessed sequences for SASRec

### Thumbnails (optional)
- `data-5/` - Video thumbnail images for CLI display

### Documentation (4 files)
- `README.md` - Comprehensive project documentation
- `HOW_TO_RUN.md` - Quick terminal commands
- `FILE_DEPENDENCIES.md` - File dependency mapping
- `requirements.txt` - Python dependencies (minimal)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Interactive Recommender
```bash
# RS1 (TimeSVD++ + KNN)
python run_RS1.py

# RS2 (SASRec + KNN)
python run_RS2.py
```

### Run Evaluation
```bash
python run_eval.py
```

### Size Breakdown:
- Preprocessed data: ~80 MB
- Trained models: ~10 MB
- Thumbnails: ~20 MB (optional)
- Code + docs: <1 MB

## Verification Checklist

All systems have been tested and **confirmed working**:
- `run_RS1.py` - Loads model, displays recommendations
- `run_RS2.py` - Loads model, displays recommendations  
- `run_eval.py` - Evaluates both models, generates metrics

## Complete File Structure

```
SUBMISSION/
├── run_RS1.py
├── run_RS2.py
├── run_eval.py
├── timesvdpp.py
├── sasrec_model.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── hybrid_system_timesvdpp.pkl
│   └── hybrid_sasrec_knn.pkl
│
├── KuaiRec 2-2.0/data/
│   ├── collab_df.csv
│   ├── content_df.csv
│   ├── content_df_cli.csv
│   ├── content_features.npz
│   ├── content_metadata.pkl
│   └── sasrec_preprocessed/
│       ├── mappings.pkl
│       ├── sequences.pkl
│       ├── train_examples.pkl
│       ├── val_examples.pkl
│       └── ... (other preprocessed files)
│
└── data-5/ (thumbnails)
```
