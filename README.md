# Quadruped Air PPO

MuJoCoを使用した四脚ロボットの歩容獲得プロジェクト。エアシリンダー駆動の四脚ロボットをPPO（Proximal Policy Optimization）アルゴリズムで学習させます。

## プロジェクト概要

このプロジェクトは、以下の要素で構成されています：

- MuJoCoシミュレーション環境: エアシリンダー駆動の四脚ロボットモデル
- Gymnasium互換の環境: `QuadrupedAirEnv`
- PPOによる強化学習: Stable-Baselines3を使用した学習
- 学習済みモデルの可視化: インタラクティブなビューワーでの動作確認

### ロボット仕様

- 自由度: 8（各脚2関節 × 4脚）
- アクチュエーター: エアシリンダーを模したモーター
- 観測空間: 20次元（胴体姿勢クォータニオン4次元 + 関節角度8次元 + 関節速度8次元）
- 行動空間: 8次元（各関節への制御信号 -1〜1）

### 報酬設計

- 前進速度報酬: x方向の速度を最大化
- 高さペナルティ: 目標高さ（0.45m）からの偏差にペナルティ
- 姿勢ペナルティ: 胴体の傾きにペナルティ

## セットアップ

### 前提条件

- 動作確認済み: macOS 
    - Linux / Windows は未検証
- [uv](https://docs.astral.sh/uv/) がインストールされていること

### uvのインストール

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# またはHomebrewで
brew install uv

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### セットアップ

1. リポジトリのクローン

```bash
git clone https://github.com/kim-xps12/rabbot-rl-walk.git
cd rabbot-rl-walk
```

2. 依存関係の同期

```bash
uv sync
```

このコマンドで、`pyproject.toml`に記載された依存関係がインストールされ、ロックファイル（`uv.lock`）が作成されます。

### 必要なパッケージ

プロジェクトで使用する主要なパッケージ：

- `gymnasium`: OpenAI Gym互換の環境インターフェース
- `mujoco`: 物理シミュレーションエンジン
- `stable-baselines3>=2.7.0`: PPOアルゴリズムの実装
- `torch>`: PyTorchディープラーニングフレームワーク
- `numpy`: 数値計算ライブラリ

## 実行方法

### 学習の実行

四脚ロボットの歩容学習を開始します（デフォルトの学習期間: 200万ステップ）：

```bash
uv run src/train_quadruped.py
```

`uv run`は仮想環境内でスクリプトを実行します。初回実行時や依存関係の変更後は、自動的に環境を更新します。

学習が完了すると、`quadruped_air_ppo.zip`として学習済みモデルが保存されます。

### 学習経過の可視化

学習を開始したのとは別のterminalを開いて，以下を実行することでブラウザからtensroboardを利用して経過を確認できます，

```bash
uv run python -m tensorboard.main --logdir=logs --port=6006
```

**注意**: 学習には時間がかかります。

### 学習済みモデルの再生

学習済みモデルを使用してロボットの動作を可視化します：

```bash
uv run src/visualize.py  
```

MuJoCoのインタラクティブビューワーが起動し、学習済みモデルによるロボットの歩行が表示されます。

## ファイル構成

```
quadruped_air_ppo/
├── README.md                        # このファイル
├── pyproject.toml                   # プロジェクト設定と依存関係
├── uv.lock                          # 依存関係のロックファイル（uv syncで生成）
├── model/
│   └── quadruped_air.xml           # MuJoCoロボットモデル定義
├── src/
│   ├── quadruped_air_env.py        # Gymnasium環境実装
│   ├── train_quadruped.py          # 学習スクリプト
│   ├── visualize.py                # 可視化スクリプト（最新）
│   └── plot_training.py            # 学習結果グラフ化スクリプト
├── logs/                            # 学習ログ（自動生成）
│   └── quadruped_ppo_YYYYMMDD_HHMMSS/
│       └── events.out.tfevents.*   # TensorBoardログ
└── output/                          # 学習済みモデル保存先（自動生成）
    └── quadruped_ppo_YYYYMMDD_HHMMSS/
        └── quadruped_air_ppo.zip   # 学習済みモデル
```

## カスタマイズ

### GPU/CPU設定とパフォーマンス最適化

`train_quadruped.py`の`main()`関数内で計算デバイスと並列環境数を設定できます：

```python
# デバイス設定（main関数内）
device = "cpu"   # 推奨: CPUでの実行（PPO + MlpPolicyの場合）
# device = "mps"   # Apple Silicon GPU（実験的、速度向上は限定的）
# device = "cuda"  # NVIDIA GPU（実験的、速度向上は限定的）
# device = "auto"  # 自動検出（GPU優先、非推奨）

# 並列環境数（ファイル冒頭）
N_ENVS = 4
```

**重要:** PPOアルゴリズムは小規模なMlpPolicyを使用する場合、**CPUでの実行が効率的**です。GPUは大規模なCNNポリシーで効果を発揮します。

### ハイパーパラメータの調整

`train_quadruped.py`で以下のパラメータを調整できます：

```python
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,      # 学習率
    n_steps=2048,            # ロールアウトステップ数
    batch_size=64,           # バッチサイズ
    device=device,           # 使用デバイス (mps/cuda/cpu)
    n_epochs=10,             # 学習エポック数
)

model.learn(total_timesteps=2_000_000)  # 総学習ステップ数
```

### 報酬関数の変更

`quadruped_air_env.py`の`step()`メソッド内で報酬関数を調整できます：

```python
forward_reward = self.data.qvel[0]     # x方向速度
height_penalty = -abs(0.45 - self.data.qpos[2]) * 1.0
tilt_penalty = -np.linalg.norm(self.data.qpos[3:7] - np.array([1,0,0,0])) * 2.0

reward = forward_reward + height_penalty + tilt_penalty
```

### ロボットモデルの変更

`model/quadruped_air.xml`でロボットの物理パラメータを変更できます：

- ジョイントの減衰係数（`damping`）
- モーターのギア比（`gear`）
- リンクのサイズや質量
- 関節可動域（`range`）

## トラブルシューティング

### MuJoCoのインストールエラー

MuJoCoのインストールで問題が発生した場合、`pyproject.toml`のバージョン指定を調整してください。

### 依存関係の手動インストール

`uv run`を使わずに仮想環境で実行したい場合：

```bash
# 仮想環境を作成
uv venv --python 3.11

# 仮想環境を有効化
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# 依存関係をインストール
uv pip install -e .

# スクリプトを実行
cd src
python train_quadruped.py
```

### 学習が進まない場合

- 報酬関数のバランスを調整
- 学習率を変更（`learning_rate=1e-4`など）
- ネットワークアーキテクチャを変更（`policy_kwargs`で設定可能）

## 参考情報

- [Stable-Baselines3 ドキュメント](https://stable-baselines3.readthedocs.io/)
- [MuJoCo Python バインディング](https://mujoco.readthedocs.io/)
- [Gymnasium ドキュメント](https://gymnasium.farama.org/)
- [PPO論文](https://arxiv.org/abs/1707.06347)

## ライセンス

プロジェクト固有のライセンスを記載してください。

## 開発環境

このプロジェクトは以下の環境で開発・テストされています：

- Python 3.11.9
- Stable-Baselines3 2.7.0
- PyTorch 2.9.1+cpu
- Gymnasium 1.2.2
- NumPy 2.3.5

## Author

Rabbot Laboratory Team.
