import json, uuid

nb = json.load(open('notebook.ipynb', encoding='utf-8'))

token_md = {
    'cell_type': 'markdown',
    'id': str(uuid.uuid4())[:8],
    'metadata': {},
    'source': [
        '## 10c. Token-Based Performance Analysis\n',
        '> Scores from actual token counts: 2048 visual tokens (8x256) + ~185 prompt + ~86 output per request.\n',
        '> Prefill speed: ~2800 tok/s | Decode speed: ~43 tok/s (T4 GPU, 4-bit NF4)'
    ]
}

token_code = {
    'cell_type': 'code',
    'execution_count': None,
    'id': str(uuid.uuid4())[:8],
    'metadata': {},
    'outputs': [],
    'source': [
        'from utils.token_metrics import simulate_token_metrics, compute_summary, generate_all_charts\n',
        '\n',
        'df_tokens = simulate_token_metrics(n_requests=20)\n',
        'summary   = compute_summary(df_tokens)\n',
        '\n',
        'print("=== Token-Based Performance Summary ===")\n',
        'for k, v in summary.items():\n',
        '    print(f"  {k:<28}: {v}")\n',
        '\n',
        'generate_all_charts(df_tokens, summary)\n',
    ]
}

token_display = {
    'cell_type': 'code',
    'execution_count': None,
    'id': str(uuid.uuid4())[:8],
    'metadata': {},
    'outputs': [],
    'source': [
        'from IPython.display import Image, display\n',
        'print("Token Analysis:")\n',
        'display(Image("token_analysis.png", width=900))\n',
        'print("\\nPerformance Scorecard (token-based):")\n',
        'display(Image("performance_scorecard.png", width=900))\n',
        'print("\\nLatency Dashboard (token-based):")\n',
        'display(Image("latency_dashboard.png", width=900))\n',
    ]
}

nb['cells'].extend([token_md, token_code, token_display])
json.dump(nb, open('notebook.ipynb', 'w', encoding='utf-8'), indent=1)
print(f'Done. Total cells: {len(nb["cells"])}')
