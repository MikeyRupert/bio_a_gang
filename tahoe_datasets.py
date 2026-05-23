# =========================================================
# Exploring Tahoe-100M: a giga-scale single-cell drug atlas
# =========================================================

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np


# ---------- 1. Stream 5 cells (no download) ----------
ds = load_dataset("tahoebio/Tahoe-100M", split="train", streaming=True)
sample = list(ds.take(5))


# ---------- 2. Schema of one cell ----------
row = sample[0]
row.keys()


# ---------- 3. Cell metadata ----------
row["cell_line_id"]        # Cellosaurus ID of the cancer cell line
row["drug"]                # small-molecule perturbation name
row["moa-fine"]            # mechanism of action
row["canonical_smiles"]    # SMILES string for the drug
row["BARCODE_SUB_LIB_ID"]  # unique cell barcode


# ---------- 4. Sparse expression: parallel arrays ----------
# `genes` holds integer token IDs, `expressions` holds values.
len(row["genes"])
list(zip(row["genes"][:10], row["expressions"][:10]))


# ---------- 5. Quick scan across 5 cells ----------
for r in sample:
    print(r["cell_line_id"], "|", r["drug"], "|", r["moa-fine"])


# ---------- 6. Download the gene vocabulary ----------
# Maps integer token IDs in `genes` -> human-readable gene symbols.
gene_path = hf_hub_download(
    repo_id="tahoebio/Tahoe-100M",
    filename="metadata/gene_metadata.parquet",
    repo_type="dataset",
)
gene_df = pd.read_parquet(gene_path)
gene_df.head()


# ---------- 7. Build the lookup ----------
id_to_symbol = dict(zip(gene_df["token_id"], gene_df["gene_symbol"]))


# ---------- 8. Decode this cell's top 10 expressed genes ----------
g = np.array(row["genes"])
e = np.array(row["expressions"])

# token IDs 0-2 are special (padding/mask/etc.) - drop them
mask = np.isin(g, list(id_to_symbol))
g = g[mask]
e = e[mask]

top = np.argsort(e)[::-1][:10]
[(id_to_symbol[g[i]], e[i]) for i in top]


# ---------- 9. Download one shard of pseudobulk DE ----------
de_path = hf_hub_download(
    repo_id="tahoebio/Tahoe-100M",
    filename="metadata/pseudobulk_differential_expression/train-00000-of-01026.parquet",
    repo_type="dataset",
)
de_df = pd.read_parquet(de_path)

de_df.head()
de_df.columns
de_df.shape


# ---------- 10. Genes related to one drug ----------
drug = "4EGI-1"   # change to whatever drug you want
drug_de = de_df[de_df["drug"] == drug]

drug_de.shape
drug_de.head(20)


# ---------- 11. Top up-regulated genes for this drug ----------
drug_de.sort_values("log2FoldChange", ascending=False).head(20)


# ---------- 12. Top down-regulated genes for this drug ----------
drug_de.sort_values("log2FoldChange", ascending=True).head(20)