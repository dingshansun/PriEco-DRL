# train_rl_par.py
# -*- coding: utf-8 -*-
import os
# ✅ 多进程时一定要限制线程，否则每个进程又开很多线程，反而更慢
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize

from sutraffic_env import SUTrafficEnv
import gc, time

# ---------- 你的数据读取保持不动 ----------
link_start_edge = pd.read_csv('For_correcting_LTT.txt',header=None,sep=' ').values
bus_stop_position = pd.read_csv('Bus_stop_position.txt',header=None,sep=' ').values
edge_length_speed = pd.read_csv('Edge_length_speed.txt', header=None, sep=' ').values
in_out = pd.read_csv('In_out_pairs.txt', header=None, sep=' ').values
signal_phase = pd.read_csv('Signal_phase.txt', header=None, sep=' ').values
sourced_link = pd.read_csv('Sourced_link.txt', header=None, sep=' ').values
corridor_edges = pd.read_csv('Corridor_edge.txt', header=None, sep=' ').values

sumo_cfg_path = "1_osm.sumocfg"
corridor_edges = set(corridor_edges[:,0].tolist())
edge_length_speed = {str(e): (float(L), float(v)) for e,L,v in edge_length_speed}
link_start_edge = {str(e): str(s) for e,s in link_start_edge}
in_out = {row[0]: (row[1], row[2].split('_')) for row in in_out}


# ---------- 回调保持不动 ----------
class EpisodeTBCallback(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                r = float(info["episode"]["r"])
                l = int(info["episode"]["l"])
                self.logger.record("episode/return", r, exclude=("stdout",))
                self.logger.record("episode/length", l,  exclude=("stdout",))
                self.logger.dump(self.num_timesteps)
        return True

class ReplayBufferSaveCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self.model.save_replay_buffer(self.save_path)
        return True


# ✅ 用“工厂函数”创建 env，给每个进程独立 seed/port
def make_env(rank: int, base_seed: int = 12, base_port: int = 8813):
    def _init():
        env = SUTrafficEnv(
            sumo_cfg_path, link_start_edge, edge_length_speed,
            bus_stop_position, corridor_edges, sourced_link, in_out, signal_phase,
        )
        # gymnasium 风格可 seed：如果你 env 支持 reset(seed=...)
        try:
            env.reset(seed=base_seed + rank)
        except TypeError:
            pass
        return env
    return _init


def main(run_idx=0, base_seed=12):
    n_envs = 10   # 先试 4，再到 6/8/10，别一上来开满 12
    # base_seed = 13

    log_dir   = f"runs/{datetime.now():%Y%m%d_%H%M%S}_run{run_idx:02d}_seed{base_seed}"
    model_dir = f"models/{datetime.now():%Y%m%d_%H%M%S}_run{run_idx:02d}_seed{base_seed}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ✅ 并行环境
    env = SubprocVecEnv([make_env(i, base_seed=base_seed) for i in range(n_envs)], start_method="spawn")
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=10.0)

    # ✅ eval 环境建议单独一个（不并行），保持你原逻辑
    eval_env = DummyVecEnv([make_env(0, base_seed=9999)])  # 给 eval 一个固定 seed
    eval_env = VecMonitor(eval_env, filename=os.path.join(log_dir, "eval_monitor.csv"))
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms

    policy_kwargs = dict(net_arch=[256, 256])

    # ----------- 关键：并行后如何“不改变训练强度” -----------
    # 你原来是：每 1 条 transition -> gradient_steps=2 次更新
    # 现在每次 env.step() 会产生 n_envs 条 transition
    # 如果你仍用 gradient_steps=2，会变成每条 transition 更新更少（学习可能变慢）
    # ✅ 想“尽量保持原更新/样本比”，就把 gradient_steps 乘以 n_envs：
    gradient_steps = n_envs

    model = SAC(
        "MlpPolicy",
        env,
        seed=base_seed,
        learning_rate=3e-4,
        learning_starts=10_000,
        buffer_size=200_000,
        batch_size=512,
        tau=0.002,
        gamma=0.99,
        train_freq=n_envs,
        gradient_steps=gradient_steps,   # ✅ 这里改成 2*n_envs
        ent_coef="auto",
        target_entropy="auto",
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
    )

    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    ckpt_cb = CheckpointCallback(save_freq=5_000, save_path=model_dir, name_prefix="sac")
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=5_000, n_eval_episodes=20, deterministic=True
    )
    ep_cb = EpisodeTBCallback()
    rb_cb = ReplayBufferSaveCallback(save_freq=50_000, save_path=os.path.join(model_dir, "replay_buffer.pkl"))

    model.learn(total_timesteps=500_000, callback=[ep_cb, ckpt_cb, eval_cb, rb_cb])

    model.save(os.path.join(model_dir, "sac_last"))
    env.save(os.path.join(model_dir, "vecnorm.pkl"))
    env.close(); eval_env.close()

    # 训练曲线：每集回报/秒（或每步回报）+ 滑动平均
    m = pd.read_csv(os.path.join(log_dir, "monitor.csv"), comment="#")
    dt = 1.0  # env 步长是 1s
    m["r_per_s"] = m["r"] / (m["l"] * dt)
    m["cum_steps"] = m["l"].cumsum()
    m["r_per_s_ma"] = m["r_per_s"].rolling(200, min_periods=1).mean()

    plt.figure(); plt.plot(m["cum_steps"], m["r_per_s_ma"])
    plt.xlabel("timesteps"); plt.ylabel("reward per second (200-ep MA)")
    plt.title("Training (smoothed)"); plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "train_reward_per_s.png"), dpi=160)

    try:
        em = pd.read_csv(os.path.join(log_dir, "eval_monitor.csv"), comment="#")
        em["r_per_s"] = em["r"] / (em["l"] * dt)
        em["episode"] = range(len(em))
        plt.figure(); plt.plot(em["episode"], em["r_per_s"].rolling(10, min_periods=1).mean(), marker='o')
        plt.xlabel("eval episode (blocks)"); plt.ylabel("eval reward per second")
        plt.title("Evaluation (fixed seeds)"); plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "eval_reward_per_s.png"), dpi=160)
    except FileNotFoundError:
        pass
    plt.close("all")   # 关掉所有 figure，防止多轮内存涨
    del model, env, eval_env
    gc.collect()

def run_many(n_runs=5, base_seed=18, seed_step=100):
    for k in range(n_runs):
        print(f"\n========== Run {k+1}/{n_runs} ==========")
        main(run_idx=k, base_seed=base_seed + k * seed_step)
        gc.collect()
        time.sleep(3)   # 给 SubprocVecEnv 子进程一点时间彻底退出

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    run_many(n_runs=2, base_seed=26, seed_step=100)
