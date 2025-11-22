"""
MuJoCo四脚ロボットのPPO学習スクリプト

並列環境とプログレストラッキングを使用した効率的な学習を実行します。
"""
import os
from datetime import datetime
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from quadruped_air_env import QuadrupedAirEnv

# 並列環境数
N_ENVS = 4

class ProgressCallback(BaseCallback):
    """学習進捗を10%刻みで表示するコールバック"""
    
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_milestone = 0
        
    def _on_step(self) -> bool:
        # 進捗率を計算（10%刻み）
        progress = int((self.num_timesteps / self.total_timesteps) * 100)
        milestone = (progress // 10) * 10
        
        # 10%刻みのマイルストーンに到達したら表示
        if milestone > self.last_milestone and milestone <= 100:
            print(f"\n{'='*60}")
            print(f"学習進捗: {milestone}% ({self.num_timesteps:,}/{self.total_timesteps:,} steps)")
            print(f"{'='*60}\n")
            self.last_milestone = milestone
            
        return True

def make_env():
    """環境生成関数（並列化用）"""
    def _init():
        return QuadrupedAirEnv()
    return _init

def main():
    # デバイス設定（CPUを推奨）
    device = "cpu"  # PPO + MlpPolicyはCPUが最適
    # device = "mps"   # Apple Silicon GPU（実験的）
    # device = "cuda"  # NVIDIA GPU（実験的）
    
    # タイムスタンプ生成（ログとモデル保存で共通利用）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("="*60)
    print("Quadruped Air PPO Training")
    print("="*60)
    print(f"Device: {device}")
    print(f"Parallel Environments: {N_ENVS}")
    print(f"PyTorch Threads: {torch.get_num_threads()}")
    print(f"Timestamp: {timestamp}")
    print("="*60 + "\n")
    
    # 並列環境の作成
    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    
    # ログディレクトリの設定
    log_dir = f"logs/quadruped_ppo_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 出力ディレクトリの設定
    output_dir = f"output/quadruped_ppo_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # PPOモデルの作成
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        device=device,
        n_epochs=10,
        tensorboard_log=log_dir,
    )
    
    # 学習の実行
    total_timesteps = 2_000_000
    progress_callback = ProgressCallback(total_timesteps)
    
    print(f"Starting training for {total_timesteps:,} timesteps...")
    print(f"TensorBoard logs: {log_dir}")
    print(f"Model output: {output_dir}")
    print(f"Run: uv run tensorboard --logdir=logs\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=progress_callback,
        progress_bar=True,
    )
    
    # モデルの保存（タイムスタンプ付きディレクトリに保存）
    model_path = os.path.join(output_dir, "quadruped_air_ppo")
    model.save(model_path)
    
    print("\n" + "="*60)
    print("✓ Training completed!")
    print(f"✓ Model saved: {model_path}.zip")
    print(f"✓ Logs saved: {log_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

