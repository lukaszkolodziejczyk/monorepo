# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
%load_ext autoreload
%autoreload 2

# %%
import pandas as pd
import shutil
from pathlib import Path
from mostlyai.engine import split, analyze, encode, train, generate, init_logging

# %%
init_logging()

# %%
workspace_dir = Path("players_ws")
# shutil.rmtree(workspace_dir, ignore_errors=True)

tgt_encoding_types = {"nameFirst": "LANGUAGE_CATEGORICAL", "nameLast": "LANGUAGE_CATEGORICAL"}

# %%
def split_analyze_encode_train(tgt_encoding_types, workspace_dir):
    players_df = pd.read_csv("https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/baseball/players.csv.gz").sample(n=1_000).reset_index(drop=True)
    tgt_columns = list(tgt_encoding_types.keys()) + ["id"]
    ctx_columns = [c for c in players_df.columns if c not in tgt_columns] + ["id"]
    tgt_df = players_df[tgt_columns]
    ctx_df = players_df[ctx_columns]
    
    split(
        tgt_data=tgt_df,
        tgt_context_key="id",
        tgt_encoding_types=tgt_encoding_types,
        ctx_data=ctx_df,
        ctx_primary_key="id",
        ctx_encoding_types=None,
        workspace_dir=workspace_dir,
    )
    analyze(workspace_dir=workspace_dir)
    encode(workspace_dir=workspace_dir)
    train(model="HuggingFaceTB/SmolLM2-135M", max_training_time=0.1, workspace_dir=workspace_dir)
    return tgt_df, ctx_df

# %%
tgt_df, ctx_df = split_analyze_encode_train(tgt_encoding_types, workspace_dir)

# %%
gen_ctx_df = ctx_df.sample(500).reset_index(drop=True)
gen_ctx_df.head()

# %%
generate(ctx_data=gen_ctx_df, workspace_dir=workspace_dir, _use_xgrammar=True)
syn_df = pd.read_parquet(workspace_dir / "SyntheticData")
syn_df.head()

# %%
gen_seed_df = tgt_df[["nameFirst"]].sample(500).reset_index(drop=True)
gen_seed_df.head()

# %%
generate(ctx_data=gen_ctx_df, seed_data=gen_seed_df, workspace_dir=workspace_dir, _use_xgrammar=True)
