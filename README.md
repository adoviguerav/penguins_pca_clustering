# 🐧 Penguins: PCA + Clustering

Unsupervised analysis of the classic **Palmer Penguins** dataset: dimensionality reduction with **PCA** plus **hierarchical clustering** and **K-means** — all in a single notebook, commented step by step.

The result? Without ever using the species label, the clustering recovers the 3 groups that match the 3 real species (Adelie, Chinstrap and Gentoo). The math rediscovers the biology.

## What's inside

The notebook [PCA_Clustering_Adolfo_Viguera.ipynb](PCA_Clustering_Adolfo_Viguera.ipynb) walks through the full pipeline:

1. **Exploration** — pairplot by species, distribution by island, NaN handling.
2. **Correlations** — correlation matrix exposing multicollinearity (body mass ↔ flipper length, high r): the ideal scenario for PCA.
3. **PCA** — on the 4 standardized numeric variables. The first 2 components explain **~88% of the variance** (68% + 19%).
4. **Component interpretation** — eigenvectors, cos², contributions, correlation circle and scatter plots in component space colored by species, island and sex.
5. **Hierarchical clustering** — Euclidean distance matrix + dendrogram (Ward) to get a feel for the optimal k.
6. **K-means** — k=3, contrasted with the **elbow method** and a k=5 trial.
7. **Validation** — **silhouette method** to confirm the choice of k.
8. **Conclusions** — hierarchical vs. K-means comparison and group interpretation: the clusters are the species.

The visualization helpers (explained variance, cos² heatmap, correlation circle, scatters with vectors and categories…) live in [functions/pca_plots.py](functions/pca_plots.py), documented with docstrings.

## How to run it

Requirements: Python 3.10+ and an internet connection the first time (the dataset is downloaded via `seaborn.load_dataset`; it gets cached afterwards).

```bash
git clone https://github.com/adolfoviguera/penguins_pca_clustering.git
cd penguins_pca_clustering

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook PCA_Clustering_Adolfo_Viguera.ipynb
```

Then run all cells (`Run All`). Verified end to end in a clean environment with current versions of every dependency.

## Structure

```
├── PCA_Clustering_Adolfo_Viguera.ipynb   # Full analysis
├── functions/
│   └── pca_plots.py                      # PCA visualization helpers
├── requirements.txt
└── LICENSE                               # MIT
```

## License

[MIT](LICENSE) — use it, copy it, learn from it.
