# eval_deploy.py
import os, sys, pandas as pd, numpy as np
from stable_baselines3 import SAC
from sutraffic_env import SUTrafficEnv
from extract_CV_information import bus_in_dwell
from datetime import datetime
import pickle
# ===== 1) SUMO 环境变量 & 工具路径 =====
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci  # noqa

# ===== 2) 与训练脚本一致的数据加载 & 预处理 =====
sumo_cfg_path = "1_osm.sumocfg"

# 直接复制你 train_rl.py 里的这一段：
link_start_edge = pd.read_csv('For_correcting_LTT.txt',header=None,sep=' ').values
bus_stop_position = pd.read_csv('Bus_stop_position.txt',header=None,sep=' ').values
edge_length_speed = pd.read_csv('Edge_length_speed.txt', header = None, sep=' ').values
in_out           = pd.read_csv('In_out_pairs.txt', header = None, sep=' ').values
signal_phase     = pd.read_csv('Signal_phase.txt', header = None, sep=' ').values
sourced_link     = pd.read_csv('Sourced_link.txt', header = None, sep=' ').values
corridor_edges   = pd.read_csv('Corridor_edge.txt', header = None, sep=' ').values

# 与训练脚本保持一致的预处理：
corridor_edges   = set(corridor_edges[:,0].tolist())
edge_length_speed = {str(e): (float(L), float(v)) for e,L,v in edge_length_speed}
link_start_edge   = {str(e): str(s) for e,s in link_start_edge}
in_out            = {row[0]: (row[1], row[2].split('_')) for row in in_out}

class ObsNormalizer:
    """从 VecNormalize 保存的 vecnorm.pkl 中提取 obs_rms，并对 obs 做同款 normalize."""
    def __init__(self, vecnorm_pkl: str):
        if not os.path.exists(vecnorm_pkl):
            raise FileNotFoundError(f"vecnorm.pkl not found: {vecnorm_pkl}")

        with open(vecnorm_pkl, "rb") as f:
            vn = pickle.load(f)  # VecNormalize 对象（通过 pickle 保存）

        # 注意：obs_rms 里是 mean/var
        self.mean = np.array(vn.obs_rms.mean, dtype=np.float32)
        self.var  = np.array(vn.obs_rms.var,  dtype=np.float32)
        self.eps  = float(getattr(vn, "epsilon", 1e-8))
        self.clip = float(getattr(vn, "clip_obs", 10.0))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)

        # 维度检查（防止你 default_state 维度变化导致 silent bug）
        if obs.shape[-1] != self.mean.shape[-1]:
            raise ValueError(
                f"Obs dim mismatch: obs has {obs.shape[-1]}, vecnorm expects {self.mean.shape[-1]}"
            )

        obs_norm = (obs - self.mean) / np.sqrt(self.var + self.eps)
        obs_norm = np.clip(obs_norm, -self.clip, self.clip)
        return obs_norm

# ===== 3) 定义 BusPolicy：SAC 控制所有 corridor 公交 =====
class SACBusPolicy:
    def __init__(self, model: SAC, vecnorm_pkl: str, debug=False):
        self.model = model
        self.norm = ObsNormalizer(vecnorm_pkl)
        self.debug = debug

    def step(self, env, t, bus_ids, bus_info):
        """
        返回: {bus_id: acceleration}
        """
        actions = {}

        # bus_info -> dict: id -> row
        info_by_id = {row[0]: row for row in (bus_info or [])}

        for bid in bus_ids:
            row = info_by_id.get(bid, None)
            if row is None:
                continue

            # 1) 构造原始观测（1D）
            obs_raw = env._build_obs_for_eval(bid, [row])
            obs_raw = np.asarray(obs_raw, dtype=np.float32)

            # 兜底：防 NaN/inf（一旦出现，policy 很容易“发疯”）
            if not np.all(np.isfinite(obs_raw)):
                if self.debug:
                    print(f"[WARN] non-finite obs for {bid} at t={t}, fallback to default_state")
                obs_raw = np.asarray(env.default_state, dtype=np.float32)

            # 2) 用训练时的 VecNormalize 统计量做同款 normalize + clip
            obs = self.norm(obs_raw)

            # 3) predict：SB3 建议输入 shape = (n_env, obs_dim)
            obs_in = obs.reshape(1, -1)
            act, _ = self.model.predict(obs_in, deterministic=True)

            # 4) act shape 兼容：可能是 (1,1) 或 (1,) 或 (1,action_dim)
            act = np.asarray(act).squeeze()
            a_cmd = float(act) if act.ndim == 0 else float(act[0])

            actions[bid] = a_cmd

            # 你原来的 debug（建议传 raw 和 normalized 都打印更好）
            if self.debug and hasattr(env, "debug_log_stop_reward"):
                env.debug_log_stop_reward(obs_raw, bid, a_cmd)  # 这里传 raw，方便你对照
            # print("raw min/max:", obs_raw.min(), obs_raw.max(),
            #     "norm min/max:", obs.min(), obs.max())
        return actions

class G2_BusPolicy:
    """
    GLOSA+GLODTA 结合的公交控制策略：

    - 远离路口：用 GLOSA 型速度建议（你的公式），给一个合适的 v_target；
    - 靠近路口且在站台 dwell：用 GLODTA 型逻辑，红灯时“hold 在站台”，绿灯放行；
    - 最终输出加速度 a = (v_target - v_cur) / tau。
    """

    def __init__(
        self,
        v_cruise=8.9,       # ~50 km/h
        v_min=1.0,           # 最低有意义的巡航速度
        v_max=13.9,          # ~60 km/h
        a_min=-2.6,          # 舒适减速度 [m/s^2]
        a_max=2.6,           # 最大加速度 [m/s^2]
        tau=2.0,             # 速度跟踪时间常数
        glodta_zone=120.0,   # [m] 进入这个距离以内视为“接近路口”
        dwell_zone=60.0      # [m] 车站与路口距离小于这个值 → 视为“near-intersection stop”
    ):
        self.v_cruise = v_cruise
        self.v_min    = v_min
        self.v_max    = v_max
        self.a_min    = a_min
        self.a_max    = a_max
        self.tau      = tau
        self.glodta_zone = glodta_zone
        self.dwell_zone  = dwell_zone

    def step(self, env, t, bus_ids, bus_info):
        """
        env      : SUTrafficEnv
        t        : 当前仿真时刻
        bus_ids  : 当前走廊公交列表
        bus_info : 你在 eval_step 里准备好的 CV 行（这里主要用来喂 _build_obs_for）
        """
        actions = {}
        info_by_id = {row[0]: row for row in bus_info} if bus_info else {}

        for bid in bus_ids:
            # 当前速度
            try:
                v_cur = traci.vehicle.getSpeed(bid)
            except traci.TraCIException:
                continue

            # 通过你现有的观测构造函数拿“距路口距离”
            try:
                row = info_by_id.get(bid, None)
                if row is not None:
                    obs = env._build_obs_for_eval(bid, [row])
                else:
                    obs = env._build_obs_for_eval(bid, [])
                d_int = float(obs[0])  # 第一个元素：距路口距离
            except Exception:
                d_int = 1e9  # 有问题就当离路口很远

            # 获取最近信号灯信息: (tlsID, tlsIndex, dist, state, nextSwitch)
            try:
                tls_list = traci.vehicle.getNextTLS(bid)
            except traci.TraCIException:
                tls_list = []

            if not tls_list:
                # 没有信号灯：简单巡航
                v_target = self.v_cruise
            else:
                first = tls_list[0]
                if len(first) == 4:
                    tls_id, tls_index, dist_tls, tls_state = first
                    next_switch = None
                else:
                    tls_id, tls_index, dist_tls, tls_state, next_switch = first

                d_tls = float(dist_tls)

                # 是否在靠近路口的车站 dwell？
                try:
                    in_dwell = bool(bus_in_dwell(bid))
                except Exception:
                    in_dwell = False
                close_to_tls = (d_tls <= self.dwell_zone)

                # 1) GLODTA：near-intersection stop + dwell → 红灯 hold，绿灯放行
                if in_dwell and close_to_tls:
                    v_target = self._glodta_hold_speed(bid, tls_id, tls_index, v_cur)

                # 2) 否则走 GLOSA
                else:
                    sim_time = traci.simulation.getTime()

                    if next_switch is None:
                        try:
                            next_switch = traci.trafficlight.getNextSwitch(tls_id)
                        except traci.TraCIException:
                            next_switch = sim_time + 5.0

                    time_to_switch = max(float(next_switch) - sim_time, 0.1)

                    phase_len = 10.0
                    plan_t = max(0.0, min(phase_len, phase_len - time_to_switch))
                    v_plan = min(self.v_cruise, self.v_max)
                    v_target = self._glosa_speed_actuated(
                        bid=bid,
                        tls_id=tls_id,
                        tls_index=tls_index,
                        d_tls=float(d_tls),
                        v_cur=float(v_cur),
                        v_plan=float(v_plan),
                        a_max=self.a_max,
                        d_max=self.a_min,
                    )
                
                # print(f"  v_cur={v_cur:.2f}, time_to_switch={time_to_switch:.2f}, plan-t={plan_t:.2f}  ")
                # print("-" * 60)
            # 如果 advisory 无效（返回 -1），就用“当前速度/巡航速度”兜底
            if v_target < 0:
                v_target = min(self.v_cruise, self.v_max)

            # 把目标速度转成这一秒的加速度
            a = (v_target - v_cur) / max(self.tau, 1e-3)
            a = float(np.clip(a, self.a_min, self.a_max))
            actions[bid] = a

        return actions

    # ---------- 简化版 GLODTA：红灯 hold，绿灯释放 ----------
    def _glodta_hold_speed(self, vehicle_id, tls_id, tls_index, v_cur):
        """
        简化版 GLODTA：
        - 如果当前灯色对该 lane 是红灯 → 目标速度=0，继续停站；
        - 变成绿灯/黄灯 → 给一个接近巡航的速度，让车开始离站。
        """
        try:
            state_str = traci.trafficlight.getRedYellowGreenState(tls_id)
            current_state = state_str[tls_index]
        except traci.TraCIException:
            current_state = 'g'

        if current_state == 'r':
            return 0.0
        else:
            # 绿灯/黄灯：允许离站，直接朝 v_cruise 靠
            return max(self.v_min, min(self.v_cruise, self.v_max))

    # ---------- GLOSA：基本照搬你给的那段逻辑 ----------
    def _glosa_speed_actuated(self, bid, tls_id, tls_index, d_tls, v_cur, v_plan,
                            a_max, d_max, lane_vmax=13.89, pass_margin=0.5):
        """
        鲁棒版：只用当前绿剩余时间，不预测下一次绿（适合 actuated）
        返回 v_target；若无法给出有效建议则返回 v_plan（论文 Step4B/5B）
        """
        sim_time = traci.simulation.getTime()
        try:
            state_str = traci.trafficlight.getRedYellowGreenState(tls_id)
            cur_color = state_str[tls_index] if len(state_str) > tls_index else 'r'
            next_switch = traci.trafficlight.getNextSwitch(tls_id)
            t_left = max(float(next_switch) - float(sim_time), 0.0)
        except traci.TraCIException:
            return v_plan

        # --- 当前是绿：判断“能不能在剩余绿内通过” ---
        if cur_color in ('G', 'g'):
            # 若绿剩余太短，别算 d/t_left（会爆），直接走 v_plan
            if t_left <= pass_margin:
                return v_plan

            # 以最大可加速度能否赶上？（粗判）
            # 最快到达时间：用 v_cur + a_max 加速的下界估计（简化）
            # 这里给个很保守的近似：至少要 d / min(lane_vmax, v_cur + a_max) 秒
            v_fast = min(lane_vmax, v_cur + a_max)
            t_need = d_tls / max(v_fast, 1e-3)

            if t_need <= (t_left - pass_margin):
                # 能过：给一个刚好能过/略快的速度（但不超限速）
                v_req = d_tls / max(t_left - pass_margin, 0.1)
                return float(min(lane_vmax, self.v_max, max(v_cur, v_req)))
            else:
                # 赶不上这波绿：论文做法是回到 v_plan（不慢爬）
                return v_plan

        # --- 当前是红/黄：不预测下一绿，走 v_plan 或轻微 eco 减速 ---
        # 你也可以在接近路口时用 sqrt(2ad) 做一个舒适减速参考（可选）
        return v_plan



# 如果你想要“默认公交驾驶”的 baseline，就再加一个策略：
class DefaultBusPolicy:
    def step(self, env, t, bus_ids, bus_info):
        # 返回空 dict → 所有公交用 SUMO 默认跟驰模型
        return {}

# ===== 4) 评估主函数 =====
def run_eval(method_name, bus_policy, signal_policy, seed=123, horizon_s=3*3600):
    # 建环境（不启动 SUMO；启动在 reset() 中完成）
    env = SUTrafficEnv(
        sumo_cfg_path,
        link_start_edge,
        edge_length_speed,
        bus_stop_position,
        corridor_edges,
        sourced_link,
        in_out,
        signal_phase
    )

    # 第一次 reset：内部会 _do_hard_reset() + traci.start(...)
    obs, info = env.reset(seed=seed)

    rows = []
    for _ in range(horizon_s):
        # signal_policy=None → 用 eval_step 里的默认 fixed 信号逻辑
        metrics, world_done = env.eval_step(
            signal_policy,
            bus_policy=bus_policy
        )
        rows.append(metrics)
        if world_done:
            break

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"eval_runs/{ts}"
    os.makedirs(log_dir, exist_ok=True)

    out_csv = os.path.join(log_dir, f"{method_name}_{ts}_timeseries.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_bus_csv = os.path.join(log_dir, f"{method_name}_{ts}_bus_traj.csv")
    env.export_bus_traj(out_bus_csv)

    out_sum_csv = os.path.join(log_dir, f"{method_name}_{ts}_bus_summary.csv")
    env.export_bus_summary(out_sum_csv)

    print(f"[{method_name}] saved metrics to:", out_csv)
    print(f"[{method_name}] saved bus traj to:", out_bus_csv)
    env.close()



if __name__ == "__main__":
    # 1) SAC 控制所有公交（fixed 信号），例如使用你 best model 的路径
    model_dir = os.environ.get("PRIECO_MODEL_DIR", "models/your-trained-model")
    sac_model_path = os.path.join(model_dir, "best_model.zip")
    vecnorm_path   = os.path.join(model_dir, "vecnorm.pkl")
    
    sac_model = SAC.load(sac_model_path, device="cpu")
    # run_eval("SAC_PMP", bus_policy=SACBusPolicy(sac_model), signal_policy="PMP", seed=123)
    ########## method name 仅用于保存数据；可以适当修改，保存多次运行的数据
    g2_bus_policy = G2_BusPolicy()
    # run_eval("SAC_G2", bus_policy=g2_bus_policy, signal_policy="PMP", seed=123)
    # ########  baseline：固定信号 + 默认公交驾驶（可选）
    run_eval(
        "SAC_PMPTLS",
        bus_policy=SACBusPolicy(sac_model, vecnorm_path, debug=False),
        signal_policy="PMP",
        seed=1
    )
    # run_eval(
    #     "FT-IDM",
    #     bus_policy=None,
    #     signal_policy=None,
    #     seed=1
    # )
    # run_eval(
    #     "AC-IDM",
    #     bus_policy=None,
    #     signal_policy=None,
    #     seed=1
    # )
    # run_eval(
    #     "AC-G2",
    #     bus_policy=g2_bus_policy,
    #     signal_policy=None,
    #     seed=1
    # )
    # run_eval(
    #     "PMP-IDM",
    #     bus_policy=None,
    #     signal_policy="PMP",
    #     seed=1
    # )
    # run_eval(
    #     "PMP-G2",
    #     bus_policy=g2_bus_policy,
    #     signal_policy="PMP",
    #     seed=1
    # )
