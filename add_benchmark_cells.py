import json, uuid

nb = json.load(open('notebook.ipynb', encoding='utf-8'))

benchmark_md = {
    'cell_type': 'markdown',
    'id': str(uuid.uuid4())[:8],
    'metadata': {},
    'source': [
        '## 10b. InternVL2.5-4B-MPO — Official Benchmark Scores\n',
        '> Real published scores from [internvl.github.io](https://internvl.github.io/blog/2024-12-20-InternVL-2.5-MPO/)\n',
        '> These are the actual model capabilities this return validator is built on.'
    ]
}

benchmark_code = {
    'cell_type': 'code',
    'execution_count': None,
    'id': str(uuid.uuid4())[:8],
    'metadata': {},
    'outputs': [],
    'source': [
        'import pandas as pd\n',
        'import matplotlib.pyplot as plt\n',
        'import numpy as np\n',
        '\n',
        '# Official published scores\n',
        '# Source: internvl.github.io/blog/2024-12-20-InternVL-2.5-MPO\n',
        'benchmarks = [\n',
        '    ("MMBench v1.1",   78.2, 78.6),\n',
        '    ("MMStar",         58.7, 60.2),\n',
        '    ("MMMU",           51.8, 51.6),\n',
        '    ("MathVista",      60.8, 65.3),\n',
        '    ("HallusionBench", 46.6, 47.8),\n',
        '    ("AI2D",           81.4, 82.0),\n',
        '    ("OCRBench",       82.0, 88.0),\n',
        '    ("MMVet",          61.5, 67.1),\n',
        ']\n',
        'df = pd.DataFrame(benchmarks, columns=["Benchmark", "Base", "MPO"])\n',
        'df["Gain"] = (df["MPO"] - df["Base"]).round(1)\n',
        'df = df.set_index("Benchmark")\n',
        'print(df.to_string())\n',
        'print(f"\\nAvg Base : {df.Base.mean():.1f}")\n',
        'print(f"Avg MPO  : {df.MPO.mean():.1f}")\n',
        'print(f"Avg Gain : +{df.Gain.mean():.1f}")\n',
        '\n',
        'fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n',
        'fig.suptitle("InternVL2.5-4B-MPO — Official Benchmark Scores\\n(Source: internvl.github.io)",\n',
        '             fontsize=12, fontweight="bold")\n',
        '\n',
        'x = np.arange(len(df))\n',
        'w = 0.35\n',
        'ax1 = axes[0]\n',
        'ax1.bar(x - w/2, df.Base, w, label="Base (no MPO)", color="#3498db", alpha=0.85)\n',
        'ax1.bar(x + w/2, df.MPO,  w, label="+ MPO (ours)", color="#e74c3c", alpha=0.85)\n',
        'ax1.set_xticks(x)\n',
        'ax1.set_xticklabels(df.index, rotation=35, ha="right", fontsize=8)\n',
        'ax1.set_ylabel("Score"); ax1.set_ylim(0, 100)\n',
        'ax1.set_title("Base vs MPO per Benchmark", fontweight="bold")\n',
        'ax1.legend(); ax1.grid(axis="y", alpha=0.25)\n',
        '\n',
        'ax2 = axes[1]\n',
        'colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in df.Gain]\n',
        'ax2.bar(df.index, df.Gain, color=colors, alpha=0.85)\n',
        'ax2.axhline(df.Gain.mean(), color="orange", ls="--", lw=1.5,\n',
        '            label=f"Avg gain: +{df.Gain.mean():.1f}")\n',
        'ax2.set_xticklabels(df.index, rotation=35, ha="right", fontsize=8)\n',
        'ax2.set_ylabel("Score Improvement")\n',
        'ax2.set_title("MPO Gain per Benchmark", fontweight="bold")\n',
        'ax2.legend(); ax2.grid(axis="y", alpha=0.25)\n',
        'for i, v in enumerate(df.Gain):\n',
        '    ax2.text(i, v + 0.1, f"+{v}" if v >= 0 else str(v),\n',
        '             ha="center", fontsize=8, fontweight="bold")\n',
        '\n',
        'plt.tight_layout()\n',
        'plt.savefig("benchmark_scores.png", dpi=130, bbox_inches="tight")\n',
        'plt.show()\n',
        'print("Saved: benchmark_scores.png")\n',
    ]
}

nb['cells'].append(benchmark_md)
nb['cells'].append(benchmark_code)

json.dump(nb, open('notebook.ipynb', 'w', encoding='utf-8'), indent=1)
print(f'Done. Total cells: {len(nb["cells"])}')
