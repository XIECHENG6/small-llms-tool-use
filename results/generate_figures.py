"""
Generate publication-quality figures from experiment results.

Usage:
    python results/generate_figures.py
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False

COLORS = {
    'zero_shot': '#94a3b8',
    'fine_tuned': '#3b82f6',
    'qwen': '#3b82f6',
    'llama': '#ef4444',
    'phi': '#22c55e',
    'highlight': '#f59e0b',
    'bg': '#f8fafc',
}

def load_results():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, 'all_results.json'), 'r') as f:
        return json.load(f)


def fig_model_comparison(data, save_dir):
    """Bar chart comparing all models: zero-shot vs fine-tuned."""
    models = list(data['model_comparison'].keys())
    zero_shot = [data['model_comparison'][m]['zero_shot']['Exact Match Rate'] * 100 for m in models]
    fine_tuned = [data['model_comparison'][m]['fine_tuned']['Exact Match Rate'] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, zero_shot, width, label='Zero-Shot', 
                    color=COLORS['zero_shot'], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, fine_tuned, width, label='Fine-Tuned (QLoRA)', 
                    color=COLORS['fine_tuned'], edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.8, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=9, color='#64748b')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.8, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1e40af')

    # Add improvement arrows
    for i in range(len(models)):
        delta = fine_tuned[i] - zero_shot[i]
        mid_x = x[i] + width/2
        ax.annotate(f'+{delta:.1f}%', xy=(mid_x + 0.15, fine_tuned[i] - 2),
                    fontsize=8, color='#16a34a', fontweight='bold')

    ax.set_ylabel('Exact Match Rate (%)')
    ax.set_title('Function Calling Performance: Zero-Shot vs QLoRA Fine-Tuned', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: model_comparison.png")


def fig_data_scaling(data, save_dir):
    """Line chart showing data scaling curve."""
    scaling = data['data_scaling']
    samples = [int(k) for k in scaling.keys()]
    exact_match = [scaling[str(s)]['Exact Match Rate'] * 100 for s in samples]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(samples, exact_match, 'o-', color=COLORS['fine_tuned'], 
            linewidth=2.5, markersize=8, markeredgecolor='white', markeredgewidth=2)
    
    # Fill area under curve
    ax.fill_between(samples, exact_match, alpha=0.1, color=COLORS['fine_tuned'])

    # Annotate each point
    for s, em in zip(samples, exact_match):
        offset = 1.5
        ax.annotate(f'{em:.1f}%', (s, em + offset), ha='center', fontsize=9, fontweight='bold')

    # Mark the "sweet spot"
    ax.axvline(x=3000, color=COLORS['highlight'], linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(3200, 76, 'Sweet spot\n(3K samples)', fontsize=9, color=COLORS['highlight'], fontweight='bold')

    # Zero-shot baseline
    ax.axhline(y=72.2, color=COLORS['zero_shot'], linestyle=':', alpha=0.7, linewidth=1.5)
    ax.text(500, 73.0, 'Zero-shot baseline (72.2%)', fontsize=9, color='#64748b')

    ax.set_xscale('log')
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('Exact Match Rate (%)')
    ax.set_title('Data Scaling: Exact Match vs Training Data Size\n(Qwen2.5-3B, QLoRA rank=16)', 
                 fontweight='bold', pad=15)
    ax.set_xticks(samples)
    ax.set_xticklabels(['500', '1K', '3K', '10K', '30K'])
    ax.set_ylim(70, 95)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'data_scaling.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: data_scaling.png")


def fig_improvement_heatmap(data, save_dir):
    """Heatmap showing improvement from fine-tuning across models and metrics."""
    models = list(data['model_comparison'].keys())
    metrics = ['JSON Valid Rate', 'Function Name Acc', 'Arg Names Acc', 'Arg Values Acc', 'Exact Match Rate']
    metric_labels = ['JSON Valid', 'Func Name', 'Arg Names', 'Arg Values', 'Exact Match']

    improvements = []
    for m in models:
        row = []
        for metric in metrics:
            delta = (data['model_comparison'][m]['fine_tuned'][metric] - 
                    data['model_comparison'][m]['zero_shot'][metric]) * 100
            row.append(delta)
        improvements.append(row)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(improvements, cmap='Blues', aspect='auto', vmin=0, vmax=30)

    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)

    for i in range(len(models)):
        for j in range(len(metrics)):
            val = improvements[i][j]
            color = 'white' if val > 15 else 'black'
            ax.text(j, i, f'+{val:.1f}%', ha='center', va='center', fontsize=9, 
                   fontweight='bold', color=color)

    ax.set_title('Improvement from QLoRA Fine-Tuning (%)', fontweight='bold', pad=15)
    plt.colorbar(im, ax=ax, label='Improvement (%)', shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'improvement_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: improvement_heatmap.png")


def fig_convergence(data, save_dir):
    """Show how models converge after fine-tuning."""
    models = list(data['model_comparison'].keys())
    zero_shot = [data['model_comparison'][m]['zero_shot']['Exact Match Rate'] * 100 for m in models]
    fine_tuned = [data['model_comparison'][m]['fine_tuned']['Exact Match Rate'] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, model in enumerate(models):
        color = COLORS['llama'] if 'LLaMA' in model else (COLORS['phi'] if 'Phi' in model else COLORS['qwen'])
        ax.plot([0, 1], [zero_shot[i], fine_tuned[i]], 'o-', color=color, 
                linewidth=2, markersize=8, markeredgecolor='white', markeredgewidth=2, label=model)
        ax.text(-0.05, zero_shot[i], f'{zero_shot[i]:.1f}%', ha='right', va='center', fontsize=9, color=color)
        ax.text(1.05, fine_tuned[i], f'{fine_tuned[i]:.1f}%', ha='left', va='center', fontsize=9, color=color)

    # Highlight convergence zone
    ax.axhspan(min(fine_tuned) - 0.5, max(fine_tuned) + 0.5, xmin=0.6, xmax=1.0, 
               alpha=0.1, color=COLORS['highlight'])
    ax.text(0.85, max(fine_tuned) + 1.5, 'Convergence\nzone', ha='center', fontsize=9, 
            color=COLORS['highlight'], fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Zero-Shot', 'Fine-Tuned'], fontsize=12)
    ax.set_ylabel('Exact Match Rate (%)')
    ax.set_title('Architecture Convergence After Fine-Tuning', fontweight='bold', pad=15)
    ax.legend(loc='center left', fontsize=9)
    ax.set_xlim(-0.2, 1.3)
    ax.set_ylim(55, 95)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: convergence.png")


def main():
    data = load_results()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'figures')
    os.makedirs(save_dir, exist_ok=True)

    fig_model_comparison(data, save_dir)
    fig_data_scaling(data, save_dir)
    fig_improvement_heatmap(data, save_dir)
    fig_convergence(data, save_dir)
    
    print(f"\nAll figures saved to: {save_dir}/")


if __name__ == "__main__":
    main()
