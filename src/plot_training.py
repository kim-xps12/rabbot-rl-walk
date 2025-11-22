"""
TensorBoardログからトレーニング結果をプロットするスクリプト
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_tensorboard_logs(log_dir):
    """
    TensorBoardログファイルからデータを読み込む
    """
    # イベントファイルを検索
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return None
    
    print(f"Found {len(event_files)} event file(s)")
    
    # イベントアキュムレータを使用してデータを読み込む
    ea = event_accumulator.EventAccumulator(os.path.dirname(event_files[0]))
    ea.Reload()
    
    # 利用可能なタグを取得
    available_tags = ea.Tags()
    print(f"Available scalar tags: {available_tags.get('scalars', [])}")
    
    return ea, available_tags

def plot_training_metrics(log_dir, save_path="training_plots.png"):
    """
    学習メトリクスをプロット
    """
    result = load_tensorboard_logs(log_dir)
    if result is None:
        return
    
    ea, available_tags = result
    scalar_tags = available_tags.get('scalars', [])
    
    # プロットするメトリクス
    metrics_to_plot = {
        'rollout/ep_rew_mean': 'Episode Reward Mean',
        'rollout/ep_len_mean': 'Episode Length Mean',
        'train/loss': 'Total Loss',
        'train/policy_gradient_loss': 'Policy Gradient Loss',
        'train/value_loss': 'Value Loss',
        'train/entropy_loss': 'Entropy Loss',
        'train/approx_kl': 'Approximate KL Divergence',
        'train/clip_fraction': 'Clip Fraction',
    }
    
    # 利用可能なメトリクスのみをフィルタリング
    available_metrics = {k: v for k, v in metrics_to_plot.items() if k in scalar_tags}
    
    if not available_metrics:
        print("No metrics found to plot")
        return
    
    # サブプロットの数を計算
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (tag, title) in enumerate(available_metrics.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # データを取得
        try:
            scalar_events = ea.Scalars(tag)
            steps = [e.step for e in scalar_events]
            values = [e.value for e in scalar_events]
            
            # プロット
            ax.plot(steps, values, linewidth=1.5, alpha=0.8)
            ax.set_xlabel('Timesteps')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error plotting {tag}: {e}")
    
    # 使用していないサブプロットを非表示
    for idx in range(n_metrics, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training plots saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        # 最新のログディレクトリを自動検出
        log_dirs = glob.glob("logs/quadruped_ppo_*")
        if not log_dirs:
            print("No log directories found. Please specify a log directory.")
            print("Usage: python plot_training.py <log_directory>")
            sys.exit(1)
        log_dir = max(log_dirs, key=os.path.getmtime)
        print(f"Using latest log directory: {log_dir}")
    
    plot_training_metrics(log_dir)
