# from dbm import gnu
import sys
import modal

# -----------------------------
# Build Image
# -----------------------------

evo2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12"
    )
    .apt_install(
        [
            "build-essential",
            "cmake",
            "ninja-build",
            "libcudnn8",
            "libcudnn8-dev",
            "git",
            "gcc",
            "g++",
            "wget",
            "curl",
        ]
    )
    .env({"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++", "CACHE_BUST": "4", "GLIBCXX_USE_CXX11_ABI": "0"})
    # Install build tools and critical dependencies
    .run_commands("pip install ninja packaging wheel")
    .run_commands("pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124")
    .run_commands("pip install flash-attn==2.6.3 --no-build-isolation")
    .run_commands("pip install 'transformer_engine[pytorch]==1.13.0' --no-build-isolation")
    .run_commands("pip install stripedhyena --no-build-isolation")
    .run_commands("git clone https://github.com/Zymrael/vortex.git && cd vortex && pip install .")
    .run_commands("git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git && cd evo2 && pip install --no-deps .")

    .pip_install_from_requirements("requirements.txt")
)

# -----------------------------
# Modal App
# -----------------------------

app = modal.App("variant-analysis-evo2", image=evo2_image)

volume = modal.Volume.from_name("hf_cache",create_if_missing=True)
mount_path ="/root/.cache/huggingface"

@app.function(gpu="H100",volumes={mount_path:volume},timeout=1000)
def run_brca1_analysis():
    import base64
    from io import BytesIO
    from Bio import SeqIO
    import gzip
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    import seaborn as sns
    from sklearn.metrics import roc_auc_score

    import torch
    print(f"Torch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Device Count: {torch.cuda.device_count()}")
    
    from evo2 import Evo2
    WINDOW_SIZE=8192
    print("Loading evo2 Modal...")
    # model=Evo2("evo2_7b")
    # print("Evo2 model loaded")

    brca1_df = pd.read_excel(
     '/evo2/notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx',
    header=2,
    )
    brca1_df = brca1_df[[
        'chromosome', 'position (hg19)', 'reference', 'alt', 'function.score.mean', 'func.class',
    ]]
    brca1_df.rename(columns={
    'chromosome': 'chrom',
    'position (hg19)': 'pos',
    'reference': 'ref',
    'alt': 'alt',
    'function.score.mean': 'score',
    'func.class': 'class',
    }, inplace=True)

    brca1_df['class']=brca1_df["class"].replace(['FUNC','INT'],'FUNC/INT')

    with gzip.open('/evo2/notebooks/brca1/GRCh37.p13_chr17.fna.gz', "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_chr17 = str(record.seq)
            break
    ref_seqs = []
    ref_seq_to_index = {}

    # Parse sequences and store indexes
    ref_seq_indexes = []
    var_seqs = []

    brca1_subset = brca1_df.iloc[:500].copy()

    for _, row in brca1_subset.iterrows():
        p = row["pos"] - 1 # Convert to 0-indexed position
        full_seq = seq_chr17

        ref_seq_start = max(0, p - WINDOW_SIZE//2)
        ref_seq_end = min(len(full_seq), p + WINDOW_SIZE//2)
        ref_seq = seq_chr17[ref_seq_start:ref_seq_end]
        snv_pos_in_ref = min(WINDOW_SIZE//2, p)
        var_seq = ref_seq[:snv_pos_in_ref] + row["alt"] + ref_seq[snv_pos_in_ref+1:]

        

        

        # Get or create index for reference sequence
        if ref_seq not in ref_seq_to_index:
            ref_seq_to_index[ref_seq] = len(ref_seqs)
            ref_seqs.append(ref_seq)
        
        ref_seq_indexes.append(ref_seq_to_index[ref_seq])
        var_seqs.append(var_seq)

    ref_seq_indexes = np.array(ref_seq_indexes)

    print(f'Scoring likelihoods of {len(ref_seqs)} reference sequences with Evo 2...')
    ref_scores = model.score_sequences(ref_seqs)

    print(f'Scoring likelihoods of {len(var_seqs)} variant sequences with Evo 2...')
    var_scores = model.score_sequences(var_seqs)

    # Subtract score of corresponding reference sequences from scores of variant sequences
    delta_scores = np.array(var_scores) - np.array(ref_scores)[ref_seq_indexes]

    # Add delta scores to dataframe
    brca1_subset[f'evo2_delta_score'] = delta_scores

    # Calculate AUROC of zero-shot predictions
    y_true = (brca1_subset['class'] == 'LOF')
    auroc = roc_auc_score(y_true, -brca1_subset['evo2_delta_score'])

    plt.figure(figsize=(4, 2))

    # Plot stripplot of distributions
    p = sns.stripplot(
        data=brca1_subset,
        x='evo2_delta_score',
        y='class',
        hue='class',
        order=['FUNC/INT', 'LOF'],
        palette=['#777777', 'C3'],
        size=2,
        jitter=0.3,
    )

    # Mark medians from each distribution
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'visible': False},
                medianprops={'color': 'k', 'ls': '-', 'lw': 2},
                whiskerprops={'visible': False},
                zorder=10,
                x="evo2_delta_score",
                y="class",
                data=brca1_subset,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p)
    plt.xlabel('Delta likelihood score, Evo 2')
    plt.ylabel('BRCA1 SNV class')
    plt.tight_layout()

    buffer= BytesIO()
    plt.savefig(buffer,format="png")
    buffer.seek(0)
    plot_data=base64.b64encode(buffer.getvalue()).decode("utf-8")


    return{'VARIENTS':brca1_subset.to_dict(orient="record"),"plot":plot_data,"auroc":auroc}



@app.function()
def brca1_example():
    print("Running BRCA varient analysis with EVO2...")
    # run interface
    result = run_brca1_analysis.remote()

    # show plot from returned data


@app.local_entrypoint()
def main():
    brca1_example.local()
