"""
学習済みモデルを可視化するシンプルなスクリプト
"""
import os
import sys
import glob
os.environ['MUJOCO_GL'] = 'glfw'  # macOSで必要

import mujoco
import mujoco_viewer
from stable_baselines3 import PPO
from quadruped_air_env import QuadrupedAirEnv
import time

def find_latest_model():
    """最新の学習済みモデルを検索"""
    # output/ディレクトリ内の全モデルを検索
    model_paths = glob.glob("output/quadruped_ppo_*/quadruped_air_ppo.zip")
    
    if not model_paths:
        # output/がない場合はルートディレクトリを確認
        if os.path.exists("quadruped_air_ppo.zip"):
            return "quadruped_air_ppo"
        return None
    
    # タイムスタンプでソート（最新を取得）
    model_paths.sort(reverse=True)
    # .zipを除いたパスを返す
    return model_paths[0].replace(".zip", "")

def main():
    # 環境を作成
    env = QuadrupedAirEnv()
    
    # コマンドライン引数からモデルパスを取得、なければ最新を検索
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = find_latest_model()
    
    if model_path is None:
        print("✗ No trained model found.")
        print("  Please train the model first: uv run src/train_quadruped.py")
        print("\nAlternatively, specify model path:")
        print("  uv run src/visualize.py output/quadruped_ppo_YYYYMMDD_HHMMSS/quadruped_air_ppo")
        return
    
    # モデルを読み込み
    try:
        model = PPO.load(model_path, env=env)
        print(f"✓ Loaded trained model: {model_path}.zip")
    except FileNotFoundError:
        print(f"✗ Model file not found: {model_path}.zip")
        print("  Please check the path or train a new model.")
        return
    
    # 初期状態をリセット
    obs, _ = env.reset()
    
    print("\nStarting visualization...")
    print("Running trained policy on quadruped robot")
    print("Press Ctrl+C to stop\n")
    
    # MuJoCoビューワーを起動（変更: mujoco_viewerを使用）
    viewer = mujoco_viewer.MujocoViewer(env.model, env.data)
    
    # カメラ設定
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90
    
    # 視覚化オプション設定
    viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0  # 関節非表示
    viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 0  # アクチュエータ非表示
    viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1  # 接触点表示
    
    # レンダリング設定（影と反射を有効化）
    viewer.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1  # 反射
    viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1  # 影
    viewer.scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0  # 霧効果オフ
    
    step_count = 0
    episode_count = 0
    total_reward = 0
    
    try:
        while viewer.is_alive:  # 変更: is_running() -> is_alive
            # 学習済みポリシーから行動を取得
            action, _ = model.predict(obs, deterministic=True)
            
            # 環境を1ステップ進める
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step_count += 1
            
            # ビューワーを更新（変更: render()を使用）
            viewer.render()
            
            # フレームレート制御（リアルタイム速度）
            time.sleep(env.dt)
            
            # エピソード終了時
            if terminated or truncated:
                episode_count += 1
                print(f"Episode {episode_count}: Steps={step_count}, Total Reward={total_reward:.2f}")
                obs, _ = env.reset()
                step_count = 0
                total_reward = 0
                
    except KeyboardInterrupt:
        print("\n\n✓ Visualization stopped by user")
    finally:
        viewer.close()  # ビューワーを明示的に閉じる

if __name__ == "__main__":
    main()
