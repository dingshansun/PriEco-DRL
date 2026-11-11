import gymnasium as gym
import numpy as np
import traci
import random, math
from gymnasium import spaces
from extract_CV_information import extract_EB_information, extract_CV_information, bus_in_dwell
from pressure_controller import compute_pressure_weights, SignalController
# from eval import SACBusPolicy


def continuous_slack_reward(slack: float, *, tau: float,
                            r_pos: float = 3.0,
                            r_neg: float = -3.0) -> float:
    """Continuous, bounded reward for the green-window slack boundary.

    The reward is exactly zero at ``slack == 0``.  Positive and negative
    slack use their respective amplitudes, while sharing the same time scale.
    With the default symmetric amplitudes the value and first derivative are
    continuous at the switching point.
    """
    tau = max(float(tau), 1e-6)
    slack = float(slack)
    amplitude = float(r_pos) if slack >= 0.0 else abs(float(r_neg))
    return amplitude * math.tanh(slack / tau)


class SUTrafficEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, sumo_cfg_path, link_start_edge, edge_length_speed, bus_stop_position, corridor_edges, sourced_link, in_out, signal_phase):
        super(SUTrafficEnv, self).__init__()
        self.sumo_cfg_path = sumo_cfg_path
        self.link_start_edge = link_start_edge
        self.edge_length_speed = edge_length_speed
        self.bus_stop_position = bus_stop_position
        self.corridor_edges = corridor_edges
        self.sourced_link = sourced_link
        self.in_out = in_out
        self.signal_phase = signal_phase
        self.vehicle_edges = dict()
        self.time_entering_edge = {}
        self.state_dimension = 15
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.state_dimension,), dtype=np.float32)
        self.action_space = spaces.Box(low=-2.6, high=2.6, shape=(1,), dtype=np.float32)
        self.startopt = 0
        self.T0 = 10
        self.t0 = 0
        self.losstime = 3 # yellow time
        self.signal_controller = SignalController(self.losstime)
        self.default_state = np.array([-1, # distance_to_intersection
                                       0, # current_speed
                                       0, # current_acceleration
                                       0, # light_state
                                       -1, # 当前决策周期剩余时间
                                       0, # current_phase_pressure
                                       0, # max_other_phase_pressure
                                       0, # gap between these two pressure
                                       -1, # distance_to_lead_vehicle
                                       -1, # lead_vehicle_speed
                                       0, # lead_vehicle_acceleration
                                       -1, # distance_to_next_bus_stop
                                       0, # 公交是否处于停站状态：0是未停站，1是停站，默认未停站
                                       1.5, # 停站duration：15→0，归一化/10
                                       0], # depart_index
                                       dtype=np.float32)
        self._sticky_steps = 500 # 粘连步数
        self._focus_bus_id = None # 当前粘连的“焦点公交”与剩余步数
        self._focus_left = 0
        self.break_on_dwell = False # 是否在停站时立即打断粘连，改选别的车（建议 True）
        # 逻辑截断用：下一次 reset() 的起始观测和焦点id（不重启SUMO）
        self._pending_reset_obs = None
        self._next_focus_id = None
        # 缓存：本步用于构造观测的 CV 行（避免重复扫描）
        self._bus_info = []
        self._switch_truncated = False     # 上一步是否为“切车截断”
        self.max_wait_steps = 1000          # 逻辑/硬重置时最多等车步数
        # self._hard_reset = True
        self.max_sim_time = 11000  # 仿真上限
        self.clean_time_window=600
        self.amax = 2.6
        self.vmax = 13.9
        self.a_prev = 0.0
        self.v_prev = 0.0
        # self.v_prev_by_bus = {} 
        self.tls_prev = 0
        self.dtls_prev = -1
        self.dtbstop_prev = -1
        self.isatstop = False
        self.ego_stuck_time = 0
        # self.v_now = 0.0
        self.Bus_ids = []
        self.run = 0
        self.bug = 0
        # metrics 收集
        self._max_veh_cnt = 0
        self._max_queue_cnt = 0
        self._max_spillover_cnt = 0
        # 网络 delay 累积（用于算平均）
        self._delay_sum = 0.0
        self._delay_steps = 0
        # 轨迹记录：每步每辆公交一条记录（后面直接写 CSV）
        self._bus_traj_rows = []   # list[dict]

        # per-bus 历史量（用于 accel / jerk / TTC duration）
        self.v_prev_by_bus = {}    # 你已经在做
        self.a_prev_by_bus = {}    # 新增：用于 jerk
        self.ttc_under_by_bus = {} # 新增：累计 TTC<T_hard 的时长（秒）
        self.collision_by_bus = {} # 新增：累计碰撞次数（或是否发生过）
        self.energy_by_bus = {}    # 新增：累计能耗（J 或 kWh，按你能量模型单位）
        self.dist_by_bus = {}      # 新增：累计行驶距离（m）
        self.time_by_bus = {}      # 新增：累计存在时间（s）
        self.abs_acc_sum_by_bus = {}  # 新增：累计 |a|
        self.abs_jerk_sum_by_bus = {} # 新增：累计 |jerk|

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(int(seed))
            np.random.seed(int(seed) % (2**32 - 1))
        # ===== 逻辑 reset：上一步是“切车截断”，不重启 SUMO =====
        if self._switch_truncated and traci.isLoaded():
            self._switch_truncated = False

            # 1) 如果 step 已经准备好了新车观测，就直接返回
            if self._pending_reset_obs is not None:
                obs = np.asarray(self._pending_reset_obs, dtype=np.float32)
                self._pending_reset_obs = None
                # self._focus_bus_id = self._next_focus_id  # _focus_bus_id已经更新过了
                return obs, {}

            # 2) 否则进入等待逻辑：推进仿真直到有公交
            wait = 0
            while wait < self.max_wait_steps:
                if self.t0 >= self.max_sim_time:
                    return self._do_hard_reset()
                vehicle_ids = traci.vehicle.getIDList()
                self.Bus_ids = self._get_all_buses(vehicle_ids)
                if self.Bus_ids:
                    # 选新焦点并返回其观测（注意：等待后要重新拉 vehicle_ids / CV_ids）
                    self._choose_new_focus(self.Bus_ids)
                    desired_type = ['bus_bus', 'tram_tram', 'veh_CV']  # 或 ['bus_bus','tram_tram','veh_CV'] 看你需求
                    CV_ids = vehicle_ids
                    # CV_ids = [veh for veh in vehicle_ids if traci.vehicle.getTypeID(veh) in desired_type]
                    CV_info = extract_CV_information(
                        CV_ids, self.link_start_edge, self.edge_length_speed, self.bus_stop_position,
                        self.vehicle_edges, self.time_entering_edge, self.t0, self.T0, considerdwell=1
                    )
                    cv_row = {row[0]: row for row in CV_info}
                    self._bus_info = [cv_row[bid] for bid in self.Bus_ids if bid in cv_row]
                    weightZZ, weightIM, weightWL = compute_pressure_weights(CV_info, self.in_out, self.sourced_link)
                    # 根据CV计算压力
                    self.signal_phase = self.signal_controller.update_signal_phase(
                        self.t0, self.startopt, self.T0, self.signal_phase, weightZZ, weightIM, weightWL,
                        pressure=True, signal=False
                    )
                    state = self._build_obs_for(self._focus_bus_id, self._bus_info)
                    self.a_prev = state[2]*2
                    self.v_prev = state[1]*10
                    self.tls_prev = state[3]
                    self.dtls_prev = state[0]*500
                    self.dtbstop_prev = state[11]*500
                    self.isatstop = False
                    self.ego_stuck_time = 0
                    # self.dtls_prev = 9998
                    return np.asarray(state, dtype=np.float32), {}

                traci.simulationStep(); self.t0 += 1
                wait += 1

            # 没等到车：兜底返回默认
            return np.asarray(self.default_state, dtype=np.float32), {}

        # ===== 硬 reset：首次或仿真结束，需要重启 SUMO =====
        return self._do_hard_reset()
    
    def _do_hard_reset(self):
        if traci.isLoaded():
            print("Before close: sim_t =", self.t0)
            traci.close()
        sumo_cmd = ["sumo", "-c", self.sumo_cfg_path, "--start", "--time-to-teleport", "600",
                      "--collision.action", "warn"]
        traci.start(sumo_cmd)
        # 重新初始化运行态 #### 
        self.run += 1
        self.t0 = 0
        self._focus_bus_id = None
        self._focus_left = 0
        self._pending_reset_obs = None
        self._next_focus_id = None
        self.vehicle_edges.clear()   # 硬重置才清；逻辑重置不要清
        self.time_entering_edge.clear()
        self._bus_info.clear()
        self.Bus_ids.clear()
        self.a_prev = 0.0
        self.v_prev = 0.0
        self.tls_prev = 0
        self.dtls_prev = -1
        self.dtbstop_prev = -1
        self.isatstop = False
        self.ego_stuck_time = 0
        self._bus_traj_rows.clear()
        self.v_prev_by_bus.clear() 
        self.a_prev_by_bus.clear()
        self.ttc_under_by_bus.clear()
        self.collision_by_bus.clear()
        self.energy_by_bus.clear()
        self.dist_by_bus.clear()
        self.time_by_bus.clear()
        self.abs_acc_sum_by_bus.clear()
        self.abs_jerk_sum_by_bus.clear()
        # 等待到有公交再给初始观测；这部分只针对训练过程，eval时注释掉
        wait = 0
        while wait < self.max_wait_steps:
            vehicle_ids = traci.vehicle.getIDList()
            self.Bus_ids = self._get_all_buses(vehicle_ids)
            if self.Bus_ids and self.t0 >= 0:
                self._choose_new_focus(self.Bus_ids)
                desired_type = ['bus_bus', 'tram_tram', 'veh_CV']
                CV_ids = vehicle_ids
                # CV_ids = [v for v in vehicle_ids if traci.vehicle.getTypeID(v) in desired_type]
                CV_info = extract_CV_information(
                    CV_ids, self.link_start_edge, self.edge_length_speed, self.bus_stop_position,
                    self.vehicle_edges, self.time_entering_edge, self.t0, self.T0, considerdwell=1
                )
                cv_row = {row[0]: row for row in CV_info}
                self._bus_info = [cv_row[bid] for bid in self.Bus_ids if bid in cv_row]
                weightZZ, weightIM, weightWL = compute_pressure_weights(CV_info, self.in_out, self.sourced_link)
                    # 根据CV计算压力
                self.signal_phase = self.signal_controller.update_signal_phase(
                    self.t0, self.startopt, self.T0, self.signal_phase, weightZZ, weightIM, weightWL,
                    pressure=True, signal=False
                )
                state = self._build_obs_for(self._focus_bus_id, self._bus_info)
                self.a_prev = state[2]*2
                self.v_prev = state[1]*10
                self.dtls_prev = state[0]*500
                # self.dtls_prev = 9997
                return np.asarray(state, dtype=np.float32), {}

            traci.simulationStep(); self.t0 += 1
            wait += 1

        # 仍没有公交：兜底
        return np.asarray(self.default_state, dtype=np.float32), {}

    def step(self, action):
        # self.Bus_ids = self._get_corridor_buses()
        # —— 早截断兜底：焦点车已经不在场（或 None）——
        focus_id = self._focus_bus_id
        if (focus_id is None) or (focus_id not in self.Bus_ids):
            # 标记为“切车截断”，不推进仿真
            self._switch_truncated = True
            # 只有在确有候选车时，才预计算下一回合的起始观测
            if self.Bus_ids:
                self._choose_new_focus(self.Bus_ids)
                self._next_focus_id = self._focus_bus_id
                self._pending_reset_obs = self._build_obs_for(self._next_focus_id, self._bus_info)
            else:
                self._next_focus_id = None
                self._pending_reset_obs = None

            info = {
                "TimeLimit.truncated": True,
                "terminal_reason": "focus_invalid",
                "switch_focus_to": self._next_focus_id
            }
            # 返回一个干净的截断（reward=0, 观测无所谓，给默认即可）
            return np.asarray(self.default_state, dtype=np.float32), 0.0, False, True, info
        
        # ===== 1) 进行信号灯的更新
        if (self.t0 > self.startopt): # and ((self.t0-self.startopt) % self.T0 == 0):
            self.signal_controller.update_signal_phase(
                self.t0, self.startopt, self.T0, self.signal_phase, weightZZ = np.zeros((22, 21), dtype=np.float64), 
                weightIM = np.zeros((22, 21), dtype=np.float64), weightWL = np.zeros((22, 21), dtype=np.float64),
                pressure=False, signal=True
            )

        if self.t0 % self.clean_time_window == 0:  # 每 - 分钟清理一次
            self._cleanup_old_records()

        
        # ===== 2) 停站门控 + 动作与奖励 =====
        # dwell_reward = 0.0
        try:
            in_dwell_before = bus_in_dwell(focus_id) # 只包含上下客阶段
        except traci.TraCIException:
            self._switch_truncated = True
            self.bug += 1
            return np.asarray(self.default_state, dtype=np.float32), 0.0, False, True, {}

        # self.a_prev = traci.vehicle.getAcceleration(focus_id) # 仿真前一步的加速度
        # self.v_prev = self.v_now  # 上一步保存的速度
        try:
            if in_dwell_before:    # 停站期间不施加动作
                # traci.vehicle.setAcceleration(focus_id, 0.0, 1.0)
                applied = 0.0
            else:
                a = float(action[0])       # 动作范围[-2.6, 2.6] m/s^2
                applied = a
                traci.vehicle.setAcceleration(focus_id, applied, 1.0)
            # 奖励：停站不给梯度；否则基于“实际下发”的 applied
            # reward = float(dwell_reward) if in_dwell_before else - (applied ** 2)
        except traci.TraCIException:
            applied = 0.0
            # reward = 0.0
        

        # ===== 3) 推进一步仿真，构造 s_{t+1}（仍是同一辆旧车） =====
        traci.simulationStep()
        self.t0 += 1
        ######################### reward相关状态搜集及计算 ###########################
        # colliding_ids = set(traci.simulation.getCollidingVehiclesIDList())
        # collided = (focus_id in colliding_ids)
        if traci.simulation.getCollidingVehiclesNumber() > 0:
            collided = focus_id in traci.simulation.getCollidingVehiclesIDList()
        else:
            collided = False

        # 提取最新的状态，并缓存作为下一步初始使用
        # 判定公交是否还在网络，否则直接开启下一个episode
        try:
            current_lane = traci.vehicle.getLaneID(focus_id) # 有可能公交已经离开了网络
        except traci.TraCIException:
            self._switch_truncated = True
            return np.asarray(self.default_state, dtype=np.float32), 3.0, False, True, {}

        desired_type = ['bus_bus', 'tram_tram', 'veh_CV']
        vehicle_ids2 = traci.vehicle.getIDList()
        CV_ids2 = vehicle_ids2
        # CV_ids2 = [veh for veh in vehicle_ids2 if traci.vehicle.getTypeID(veh) in desired_type]
        self.Bus_ids = self._get_all_buses(vehicle_ids2) # 更新目标bus集
        CV_info_after = extract_CV_information(
            CV_ids2, self.link_start_edge, self.edge_length_speed, self.bus_stop_position,
            self.vehicle_edges, self.time_entering_edge, self.t0, self.T0, considerdwell=1
        )
        cv_row = {row[0]: row for row in CV_info_after}
        self._bus_info = [cv_row[bid] for bid in self.Bus_ids if bid in cv_row]
        weightZZ, weightIM, weightWL = compute_pressure_weights(CV_info_after, self.in_out, self.sourced_link)
        # 根据CV计算压力
        self.signal_phase = self.signal_controller.update_signal_phase(
            self.t0, self.startopt, self.T0, self.signal_phase, weightZZ, weightIM, weightWL,
            pressure=True, signal=False
        )
        next_state = self._build_obs_for(focus_id, self._bus_info)
        
        # 获取状态，计算奖励
        distance_to_lead_vehicle = next_state[8]*10
        lead_vehicle_speed = next_state[9]*10
        current_speed = next_state[1]*10
        current_acceleration = next_state[2]*2
        distance_to_intersection = next_state[0]*500
        distance_to_next_bus_stop = next_state[11]*500
        pressure_info = [next_state[5]*10, next_state[6]*10, next_state[7]*10]
        dec_remain = next_state[4]*10
        duration = next_state[13]*10
        is_at_stop = next_state[12]
        # 信号灯状态判定
        light_state = next_state[3]
        if light_state == 0:
            tls_state = 'G'
        elif light_state == 1:
            tls_state = 'Y'
        elif light_state == 2:
            tls_state = 'R'
        # 是否在站内停车判定
        if is_at_stop:
            self.isatstop = True
        elif self.isatstop and current_speed == 0:
            self.isatstop = True
        else:
            self.isatstop = False
        # 碰撞风险判定
        dv = current_speed - lead_vehicle_speed
        if distance_to_lead_vehicle < 0:
            TTC = float('inf')
        else:
            d_eff = max(distance_to_lead_vehicle - 1.0, 0.0)
            TTC = d_eff / max(dv, 0.1) if dv > 0 else float('inf')
        TTC = min(TTC, 10.0)
        # 距离
        d_to_isec = None if distance_to_intersection < 0  else distance_to_intersection
        if distance_to_next_bus_stop < 0:
            d_to_stop = None
            has_stop_ahead = False
        else:
            d_to_stop = distance_to_next_bus_stop
            has_stop_ahead = True
        # 绿灯过路口判定
        passed_on_green = False # 默认为否 # 这个判定有问题，通过路口后当前路口会被移除-->已修改为判定前后所在的位置
        if (current_lane.startswith(":")) and (self.dtls_prev >= 0 and self.dtls_prev < 20) and (self.tls_prev in (0,1))\
              and (current_speed > 2.0):
            passed_on_green = True
        # 判定公交的阶段
        current_stage = self.classify_stage(d_to_isec, d_to_stop, has_stop_ahead, current_speed, duration, self.isatstop)
        stop_intersection_overlap = self.is_stop_intersection_overlap(
            d_to_isec, d_to_stop, current_speed, duration, self.isatstop
        )
        # 速度饱和/碰撞/平滑/停车/安全/能耗/效率
        R_sat = self.reward_speed_saturation(current_speed, self.vmax, applied)
        Rcol = self.reward_collision(collided, w_collision=12.0)
        Rj  = self.reward_jerk(current_acceleration, self.a_prev)
        Rsf = self.reward_safety_brake(current_acceleration)
        Rst = self.reward_stop_penalty(current_speed, lead_vehicle_speed, distance_to_lead_vehicle, 
                                       is_at_stop, tls_state, d_to_isec)
        Reng = self.energy_reward(
            v_pre=self.v_prev, v_now=max(0.0, current_speed), a=current_acceleration
        )
        Rgreen = self.reward_stage2_events(passed_on_green)
        r_time = -0.1
        R_stage_2 = R_stage_3 = R_stage_4 = 0
        if current_stage == 2: # 过路口
            self.update_bus_speed_mode(focus_id, d_to_isec)
            obs = dict(
                v=current_speed,
                d_to_isec=d_to_isec,
                tls_state=tls_state,
                tls_t_rem=dec_remain,  # 在stage2则是正常数字
                pressure_info=pressure_info,
                distance_to_lead_vehicle=distance_to_lead_vehicle
            )
            R_stage_2 = self.compute_stage2_reward(obs)
            # The intersection is the primary reward stage, but a nearby stop
            # retains its approach-speed guidance in the overlap region.
            if stop_intersection_overlap and not self.in_auto_stop_zone(d_to_stop, current_speed):
                R_stage_4 = self.reward_stage4(
                    current_speed, current_acceleration, d_to_stop, duration,
                    distance_to_lead_vehicle, is_at_stop
                )
        elif current_stage == 3:  # 自由行驶
            R_stage_3 = self.reward_stage3_TTC(TTC) + self.reward_stage3_progress(current_speed)
        elif current_stage == 4: # 进站
            auto_stop_zone = self.in_auto_stop_zone(d_to_stop, current_speed)
            if auto_stop_zone:
                R_stage_4 = 0
            else:
                R_stage_4 = self.reward_stage4(current_speed, current_acceleration,
                                               d_to_stop, duration,
                                               distance_to_lead_vehicle, is_at_stop)
        else:  # stage 1
            pass
        Reff = R_stage_2 + R_stage_3 + R_stage_4
        R_eff = Reff + Rgreen + r_time
        w_energy = 2
        w_eff = 1
        reward = w_eff*R_eff + w_energy*Reng + Rcol + Rj + Rsf + Rst + R_sat
        # # 打印所有的状态以及reward
        # print(f"[DEBUG STOP] t={self.t0:.1f} veh={focus_id} lane={current_lane}")
        # print(
        #     f"  v={current_speed:.2f}, a_rl={current_acceleration:.2f}, "
        #     f"a_proj={current_acceleration:.2f}, "
        #     f"d_stop={distance_to_next_bus_stop:.1f}, d_tls={distance_to_intersection:.1f}, "
        #     f"tls={tls_state}, t_rem={dec_remain:.1f}"
        # )
        # a = float(action[0])
        # print(
        #     f"  Stage={current_stage}, Action={a:.2f}, "
        #     f"Pressure=[{', '.join(f'{p:.4f}' for p in pressure_info)}]"
        # )
        # print(
        #     f"  r_stop={Rst:.2f}, r_stage_2={R_stage_2:.2f}, r_stage_3={R_stage_3:.2f}, "
        #     f"r_stage_4={R_stage_4:.2f}, r_stage={Reff:.2f},  r_sat={R_sat:.2f} "
        # )
        # print(
        #     f"  r_energy={Reng:.2f}, r_brake={Rsf:.2f}, r_collision={Rcol:.2f}, r_jerk={Rj:.2f}, "
        #     f"r_pass_green={Rgreen:.2f}, R_total={reward:.2f}"
        # )
        # print(
        #     f"  lead_speed={lead_vehicle_speed:.2f}, d_lead={distance_to_lead_vehicle:.2f},"
        #     f" lead_acce={next_state[10]*2:.2f}"
        # )
        # print(f"  green_pass={passed_on_green}, duration={duration:.1f}," 
        #       f" is_at_stop={is_at_stop}, depart_index={next_state[14]}")
        # print("-" * 60)
        world_done = (self.t0 >= self.max_sim_time)
        ##### 更新状态
        self.a_prev = current_acceleration
        self.v_prev = current_speed
        self.dtls_prev = distance_to_intersection
        self.dtbstop_prev = d_to_stop
        self.tls_prev = light_state
        # self.dtls_prev = 9996
        # ===== 4) 粘连计数 & 是否切车 =====
        # 粘连步数递减
        if self._focus_left > 0:
            self._focus_left -= 1
        # 公交卡死情况的核查
        if current_speed <= 0.5 and not self.isatstop:
            self.ego_stuck_time += 1
        else:
            self.ego_stuck_time = 0
        if self.ego_stuck_time >= 120:
            reward -= 50
            world_done = True
        # 触发换车的条件
        still_alive = (focus_id in self.Bus_ids)   # 关键：先判断在不在网
        # in_dwell_after = bus_in_dwell(focus_id) if still_alive else 0

        should_switch = (
            (self._focus_left <= 0) or
            (not still_alive)
            # (self.break_on_dwell and (in_dwell_before or in_dwell_after))
        )

        if (not world_done) and should_switch:
            # 标记为“切车截断”
            self._switch_truncated = True

            # 在“下一回合”启用的新车与其起始观测（仅当确有候选车）
            self.Bus_ids = self._get_all_buses(vehicle_ids2)
            if self.Bus_ids:
                self._choose_new_focus(self.Bus_ids) # 此处已经更新了focus bus id
                self._next_focus_id = self._focus_bus_id
                self._pending_reset_obs = self._build_obs_for(self._next_focus_id, self._bus_info)
                self.a_prev = self._pending_reset_obs[2]*2
                self.v_prev = self._pending_reset_obs[1]*10
                self.dtls_prev = self._pending_reset_obs[0]*500
                self.dtbstop_prev = self._pending_reset_obs[11]*500
                self.tls_prev = self._pending_reset_obs[3]
                self.ego_stuck_time = 0
                # self.dtls_prev = 9995
            else:
                self._next_focus_id = None
                self._pending_reset_obs = None
                self.a_prev = 0
                self.v_prev = 0
                self.dtls_prev = -1
                self.dtbstop_prev = -1
                self.tls_prev = 0
                self.isatstop = False
                self.ego_stuck_time = 0
            info = {
                "focus_id": focus_id,
                "switch_focus_to": self._next_focus_id,
                "sticky_left": self._focus_left,
                "TimeLimit.truncated": True,   # 这是一次“截断”而非任务终结
                "terminal_reason": "switch_focus",
                "applied": applied,
                "in_dwell_before": in_dwell_before,
                "stop_intersection_overlap": stop_intersection_overlap,
                # "in_dwell_after": in_dwell_after
            }
            return np.asarray(next_state, dtype=np.float32), float(reward), bool(world_done), bool(should_switch), info
        if world_done:
            # 标记为“硬重置”，不是切车截断
            self._switch_truncated = False
            self._pending_reset_obs = None
            self._next_focus_id = None
            # next_state 已经算好了，正常返回
            return np.asarray(next_state, dtype=np.float32), float(reward), bool(world_done), bool(should_switch), {"world_done": True}
        # 不切车：继续本 episode
        info = {
            "focus_id": focus_id,
            "sticky_left": self._focus_left,
            "applied": applied,
            "in_dwell_before": in_dwell_before,
            "stop_intersection_overlap": stop_intersection_overlap,
        }
        return np.asarray(next_state, dtype=np.float32), float(reward), bool(world_done), bool(should_switch), info

    
    def close(self):
        try:
            # 调任何需要连接的 API 试探（比如 getVersion）；若断了会抛 FatalTraCIError
            traci.getVersion()
        except traci.exceptions.FatalTraCIError:
            return  # 已经断开，什么都不做，幂等

        try:
            traci.close(False)   # False=不要阻塞等待
        except traci.exceptions.TraCIException:
            pass  # 这里再兜底一次，确保不影响主进程退出

    def _get_all_buses(self, all_vehicles):
        all_bus = []
        # all_vehicles = traci.vehicle.getIDList()
        for vid in all_vehicles:
            # 只要公交/电车
            if ("bus" not in vid) and ("tram" not in vid):
                continue
            # if ("bus_65KNS" not in vid):  # 挑选指定的公交进行训练
            if ("bus_37Ams" not in vid) and ("bus_40Ams" not in vid) and ("bus_22Slo" not in vid) and ("bus_65KNS" not in vid):
                continue
            all_bus.append(vid)
        return all_bus

    def _get_corridor_buses(self):
        corridor_buses = []
        try:
            all_vehicles = traci.vehicle.getIDList()
        except traci.TraCIException:
            return corridor_buses

        for vid in all_vehicles:
            # 只要公交/电车
            if ("bus" not in vid) and ("tram" not in vid):
                continue

            try:
                edge = traci.vehicle.getRoadID(vid)  # 当前边（可能是内部边）
            except traci.TraCIException:
                continue

            # # 记录“最后一个真实边”（不以 ':' 开头的才更新）
            # if edge and not edge.startswith(':'):
            #     self.vehicle_edges[vid] = edge

            in_corridor = False

            # 1) 当前就是走廊的真实边
            if edge in self.corridor_edges:
                in_corridor = True

            # 2) 当前在内部边：看前一条/下一条真实边是否在走廊
            elif edge and edge.startswith(':'):
                in_corridor = True
                # prev_real = self.vehicle_edges.get(vid, None)
                # # 从 route 推断下一条真实边
                # next_real = None
                # try:
                #     route = traci.vehicle.getRoute(vid)           # 只包含真实边
                #     idx   = traci.vehicle.getRouteIndex(vid)      # 指向“下一条将驶入的真实边”
                #     if 0 <= idx < len(route) - 1:
                #         next_real = route[idx + 1]
                # except traci.TraCIException:
                #     pass

                # cand = [prev_real, next_real]
                # in_corridor = any((e in self.corridor_edges) for e in cand if e)
            # 命中就加入
            if in_corridor:
                corridor_buses.append(vid)

        return corridor_buses
    
    def _cleanup_old_records(self):
        self.time_entering_edge = {
            key: t_enter
            for key, t_enter in self.time_entering_edge.items()
            if self.t0 - t_enter <= self.clean_time_window
        }

        valid_vehicle_ids = {key[0] for key in self.time_entering_edge.keys()}
        self.vehicle_edges = {
            vid: edge for vid, edge in self.vehicle_edges.items()
            if vid in valid_vehicle_ids
        }
    
    def _choose_new_focus(self, Bus_ids):
        """Choose a new focus bus; softly prioritize buses on a target line."""
        if not Bus_ids:
            self._focus_bus_id = None
            self._focus_left = 0
            return None

        candidates = list(Bus_ids)

        prefer_key=""
        p_prefer=0.5

        preferred = [vid for vid in candidates if prefer_key in vid]
        others = [vid for vid in candidates if prefer_key not in vid]

        if preferred and (not others or random.random() < p_prefer):
            self._focus_bus_id = random.choice(preferred)
        else:
            self._focus_bus_id = random.choice(others if others else candidates)

        self._focus_left = max(1, int(self._sticky_steps))
        return self._focus_bus_id
    
    def _build_obs_for(self, vid, CV_info):
        """从 CV_info 取 vid 的一行，调用 extract_EB_information 构造观测"""
        if vid is None:
            return self.default_state.copy()
        row = [r for r in CV_info if r[0] == vid]
        if not row:
            return self.default_state.copy()
        S = extract_EB_information(
            row, self.link_start_edge, self.edge_length_speed,
            self.bus_stop_position, self.corridor_edges,
            self.signal_phase, self.t0, self.T0, self.startopt, self.v_prev
        )
        if hasattr(S, 'shape') and S.shape[0] > 0:
            return S[0].astype(np.float32)
        return self.default_state.copy()
    
    def _build_obs_for_eval(self, vid, CV_info):
        """从 CV_info 取 vid 的一行，调用 extract_EB_information 构造观测"""
        if vid is None:
            return self.default_state.copy()
        row = [r for r in CV_info if r[0] == vid]
        if not row:
            return self.default_state.copy()
        
        if vid not in self.v_prev_by_bus:
            v_init = None
            try:
                v_init = traci.vehicle.getSpeed(vid)
            except Exception:
                # 兜底：如果你知道 CV_info 的某一列就是速度，可以从 row[0][idx] 取
                # 例如：v_init = float(row[0][SPEED_IDX])
                pass
            if v_init is None:
                v_init = 0.0
            self.v_prev_by_bus[vid] = float(v_init)

        v_prev = float(self.v_prev_by_bus[vid])

        S = extract_EB_information(
            row, self.link_start_edge, self.edge_length_speed,
            self.bus_stop_position, self.corridor_edges,
            self.signal_phase, self.t0, self.T0, self.startopt, v_prev
        )
        if hasattr(S, 'shape') and S.shape[0] > 0:
            return S[0].astype(np.float32)
        return self.default_state.copy()
    def update_bus_speed_mode(self, bus_id, d_tls=None):
        """Retain the original design that lets the agent learn signal risk."""
        HARD_ZONE = 0.0
        if d_tls is not None and d_tls > HARD_ZONE:
            # Disable SUMO's red-light hard braking before the stop line so
            # unsafe decisions remain visible to the learning signal.
            traci.vehicle.setSpeedMode(bus_id, 15)
        else:
            traci.vehicle.setSpeedMode(bus_id, 31)

    def reward_speed_saturation(self, v, v_max, a_cmd, eps=0.5, w=0.2):
        # v_max 来自 lane max speed
        if v_max <= 0:
            return 0.0
        if v >= v_max - eps and a_cmd > 0:
            return -w * (a_cmd**2)
        return 0.0

    def reward_jerk(self, a_now, a_prev, dt=1.0):
        jerk = (a_now - a_prev) / dt
        return -(jerk**2) / 50 
    
    def reward_collision(self, collided: bool, w_collision: float = 10.0) -> float:
        """
        碰撞即罚；无碰撞为0。
        参数:
        collided    - 当前步是否发生碰撞（bool）
        w_collision - 碰撞惩罚权重（默认 10）
        返回: ≤0 的惩罚
        """
        return -w_collision if collided else 0.0
    
    def energy_reward(self, v_pre, v_now, a, mass_bus=11350, mass_pass=70*20, rho_air=1.225, Sd=8, cd=1.17,
                  Pv=0.703, Pre=0.20, cr=0.01, theta=0.0, T_amb=20.0, p_cnt=20, dt=1.0, E_ref=300000.0,     # 归一化参考能量：~30kJ/步（见下）
                  E_cap=300000.0):
        """
        返回 R_energy = -E_bus / Emax
        这里把纵向能耗 Erun 与 HVAC 能耗 EHVAC 相加作为 E_bus（论文 Eq.15, 18, 19）。
        """
        EPS = 1e-3
        M = mass_bus + mass_pass     # kg
        v_avg = 0.5*(v_pre + v_now)
        dS = v_avg * dt                                 # Eq.(11) 的离散近似

        # 四力（Eq.6~10，坡度近似为0则 sin/cos 退化）
        Facc = M * 1.05 * a                             # f≈1.05 的旋转质量系数
        Froll = M * 9.81 * cr
        Fair  = 0.5 * rho_air * Sd * cd * (v_avg**2)
        Fclimb= M * 9.81 * 0.0                          # theta≈0

        if v_now > v_pre:   # 加速（Eq.12）
            Erun = (Facc + Froll + Fair + Fclimb) * dS / max(Pv, EPS)
        elif v_now < v_pre: # 减速（Eq.13）动能回收
            Ekin = 0.5 * M * (v_now**2 - v_pre**2) * Pre
            Erun = (Froll + Fair + Fclimb) * dS / max(Pv, EPS) + Ekin
        else:               # 匀速（Eq.14）
            Erun = (Froll + Fair + Fclimb) * dS / max(Pv, EPS)

        # HVAC（Eq.16~18），冷却负载线性叠加；简单离散化积分
        PHVAC_T = 0.0264*(T_amb**2) - 1.046*T_amb + 11.13
        PHVAC_p = 0.064*max(p_cnt, 0) + 2.49
        EHVAC = (PHVAC_T + PHVAC_p) * dt

        E_bus = max(Erun + EHVAC, 0.0)
        # E_bus = min(E_bus, float(E_cap))
        reward = - E_bus / max(float(E_ref), EPS)
        return reward

    def energy_consumption(self, v_pre, v_now, a, mass_bus=11350, mass_pass=70*20, rho_air=1.225, Sd=8, cd=1.17,
                  Pv=0.703, Pre=0.20, cr=0.01, theta=0.0, T_amb=20.0, p_cnt=20, dt=1.0, E_ref=300000.0,     # 归一化参考能量：~30kJ/步（见下）
                  E_cap=300000.0):
        EPS = 1e-3
        M = mass_bus + mass_pass     # kg
        v_avg = 0.5*(v_pre + v_now)
        dS = v_avg * dt                                 # Eq.(11) 的离散近似

        # 四力（Eq.6~10，坡度近似为0则 sin/cos 退化）
        Facc = M * 1.05 * a                             # f≈1.05 的旋转质量系数
        Froll = M * 9.81 * cr
        Fair  = 0.5 * rho_air * Sd * cd * (v_avg**2)
        Fclimb= M * 9.81 * 0.0                          # theta≈0

        if v_now > v_pre:   # 加速（Eq.12）
            Erun = (Facc + Froll + Fair + Fclimb) * dS / max(Pv, EPS)
        elif v_now < v_pre: # 减速（Eq.13）动能回收
            Ekin = 0.5 * M * (v_now**2 - v_pre**2) * Pre
            Erun = (Froll + Fair + Fclimb) * dS / max(Pv, EPS) + Ekin
        else:               # 匀速（Eq.14）
            Erun = (Froll + Fair + Fclimb) * dS / max(Pv, EPS)

        # HVAC（Eq.16~18），冷却负载线性叠加；简单离散化积分
        PHVAC_T = 0.0264*(T_amb**2) - 1.046*T_amb + 11.13
        PHVAC_p = 0.064*max(p_cnt, 0) + 2.49
        EHVAC = (PHVAC_T + PHVAC_p) * dt

        E_bus = Erun + EHVAC
        return E_bus
    
    def reward_stop_penalty(self, v_now, v_lead, d_lead, is_at_stop, tls_state, d_to_isec):
    # 在非站点位置速度近似为0时的惩罚，拥堵，等红灯的情况除外，v_lead > 0.5 or <0 表示未拥堵
        if is_at_stop == 1: #在站判定
            return 0.0
        if v_lead > 0 and v_lead < 0.5 and d_lead < 10.0:   # 拥堵判定
            return 0.0
        if tls_state == 'R' and (d_to_isec >= 0.0) and (d_to_isec < 20):  # 等红灯判定
            return 0.0
        if v_now < 0.5:
            return -5.0
        return 0.0
    
    def reward_safety_brake(self, a_now):
        # 过激制动的小惩罚（可选）
        return -0.5 if a_now < -2.0 else 0.0
    
    def v_ref_docking(self, d_to_stop: float,
                    v_dock_max: float = 10.0,
                    a_comf: float = 2.0,
                    d_margin: float = 0.0) -> float:
        """
        根据到站距离给一个“合理/安全”的进站目标速度。
        - 远处（d 很大）：允许接近 v_dock_max
        - 近处（d 接近 0）：目标速度逐渐降到 0
        """
        if d_to_stop <= 0:
            return 0.0

        # 理论上在 a_comf 下能从 v_ref 减速到 0 所需距离：v_ref^2 / (2 a_comf) ≈ d_to_stop - d_margin
        d_eff = max(0.0, d_to_stop - d_margin)
        v_safe = (2.0 * a_comf * d_eff) ** 0.5  # sqrt(2 a d)

        # 不能比全局的“进站上限”更大
        return min(v_safe, v_dock_max)
    # 进入公交站过程
    def reward_stage4(self, v_i, a_action, d_to_stop, duration, dL, is_at_stop):
        V_SAFE  = 2.6
        V_HARD = 4.0      # 认为“过快”的阈值
        EPS = 1e-3
        # 停站成功的一次性奖励
        if d_to_stop == 0 and duration == 15.0:
            return 3.0
        v_target = self.v_ref_docking(d_to_stop)
        overspeed = max(0.0, v_i - v_target)
        slowspeed = max(0.0, 0.8*v_target - v_i) # punish the speed lower than the threshold
        return -0.1 * min(max(overspeed**2, slowspeed**2), 30)
        # Debs = min(d_to_stop, dL) if (dL is not None) else d_to_stop  # Eq.(26)
        # disc = v_i**2 + 2*a_action*Debs
        # if disc < 0:
        #     return -0.5
        # v_ebs = (disc)**0.5                                          # Eq.(27)
        # if 0.0 <= v_ebs <= V_SAFE:
        #     # 从 0 速到 V_SAFE，reward 从 0 → +2
        #     return 2.0 * (v_ebs / max(V_SAFE, EPS))
        # if v_ebs <= V_HARD:
        # # 线性从 0 → -1
        #     return -1.0 * ((v_ebs - V_SAFE) / (V_HARD - V_SAFE + EPS))
        # return -1.0
    # 自由行驶过程
    def reward_stage3_TTC(self, ttc, T_safe=3.5, T_hard=1.5, w=1.0, w_hard=3.0, p=2):
        """
        只在 TTC 低于阈值时惩罚；p=1线性，p=2平方。
        返回 ≤0
        """
        if ttc is None or ttc <= 0 or ttc >= 10:
            return 0
        r = 0.0
        if ttc < T_safe:
            r -= w * ((T_safe - ttc) / max(T_safe, 1e-6))**p
        if ttc < T_hard:
            r -= w_hard * ((T_hard - ttc) / max(T_hard, 1e-6))**p  # 更近再加重罚
        return float(r)
    
    def reward_stage3_progress(self, v_now: float,
                            w_prog: float = 1.0,
                            s_ref: float = 10.0,
                            v_min_prog: float = 0.5) -> float:
        """
        仅用于 Stage 3（cruise）的小幅前进奖励：
        - s_now, s_prev: 上下两步沿线路坐标（单位 m）
        - v_now: 当前速度
        - w_prog: 最大奖励系数
        - s_ref: 归一化尺度, 比如 10m 对应 w_prog
        - v_min_prog: 低于该速度不认为是在“有效行驶”，不给奖励
        """
        if v_now < v_min_prog:
            return 0.0

        prog_norm = min(1.0, v_now / max(s_ref, 1e-3))
        return w_prog * prog_norm
    
    
    def green_extension_prob_from_pressure(self, pressure_info, t_rem: float,
                                            k: float = 6.0, w_min: float = 0.15) -> float:
        """
        pressure_info = [p_cur, p_next, gap]，其中 gap = p_cur - p_next
        返回 p_ext ∈ [0,1]，表示“下一拍仍维持当前相位”的概率近似
        """
        p_cur, p_next, gap = pressure_info
        den = abs(p_cur) + abs(p_next) + 1e-2
        adv = gap / den  # 归一化优势 ∈(-1,1)
        w = 1.0 - max(0.0, min(1.0, t_rem / self.T0))
        w = max(w_min, w)
        p_pressure = 1.0 / (1.0 + math.exp(-k * adv))
        return (1.0 - w) * 0.5 + w * p_pressure
    
    def expected_extra_green(self, p1: float,
                         ext_cap: float,
                         rho: float = 0.7,
                         max_steps: int = 2) -> float:
        """
        期望额外绿窗（多拍）：ext_cap * sum_{i=1..H} (rho^{i-1} * p1^i)
        = ext_cap * (p1 * (1 - (rho*p1)**H)) / (1 - rho*p1)
        - p1: 下一拍续绿的概率（用上面的 time-aware 函数得到）
        - rho: 往后每拍的“相关性/确信度”衰减（0.5~0.8 常用）
        - max_steps: 最多考虑的续绿拍数（2~4 足够）
        """
        EPS = 1e-9
        p = max(0.0, min(1.0, p1))
        x = rho * p
        if max_steps <= 0 or ext_cap <= 0:
            return 0.0
        if abs(1 - x) < EPS:
            return ext_cap * p * max_steps          # 罕见极端，近似线性
        return ext_cap * (p * (1.0 - (x ** max_steps))) / (1.0 - x)

    def in_auto_stop_zone(self, d_stop: float, v_now: float) -> bool:
        """
        粗略判定：是否已经进入“自动进站控制区域”
        d_stop: 距离下一个公交站的距离（>0 表示还没到站，=0 在站内）
        v_now: 当前速度
        """
        if d_stop is None:
            # 以后都没有站，或者信息缺失
            return False

        # 一个基础半径，比如 20m 以内我们都认为是在 SUMO 的进站区附近
        BASE_RADIUS = 10.0

        # 根据当前速度算一个“舒适刹车距离”
        DECEL_COMF = 2.0  # 你可以用你车道/车型的实际 decel，这里只是示意
        v = max(v_now, 0.1)
        d_brake = v**2 / (2.0 * DECEL_COMF)

        # 最终认为“自动停站区”的阈值：
        trigger_dist = max(BASE_RADIUS, 0)  # 再加一点裕度

        return d_stop <= trigger_dist

    
    def reward_stage2_events(self, passed_on_green: bool, w_pass: float = 5.0) -> float:
        r = 0.0  # 只奖励通过绿灯，不惩罚等待红灯；停车惩罚已经施加过了
        if passed_on_green:
            r += w_pass
        return r
    
    def can_pass_green_simple(self, d, v, t_rem, v_max, a_max, margin=1.0):
        """
        判定在当前绿灯剩余时间 t_rem 内，是否“物理上有可能”通过路口。
        - d: 距离停止线
        - v: 当前速度
        - t_rem: 当前相位剩余时间（可叠加期望续绿 ext_cap）
        - v_max: 该路段限速
        - a_max: 车辆最大加速度
        - margin: 预留的安全余量（比如 1s），避免踩到刚好边界
        """
        # 预留一点余量，防止刚好踩线
        t_eff = max(0.0, t_rem - margin)
        if t_eff <= 0.0:
            return False

        # 需要多长时间才能从 v 加到 v_max
        if a_max <= 0:
            # 没法加速，只能看当前速度能不能在时间内跑完
            return v * t_eff >= d

        t_to_vmax = max(0.0, (v_max - v) / a_max)

        if t_eff <= t_to_vmax:
            # 还没到限速，一直加速
            d_max = v * t_eff + 0.5 * a_max * t_eff**2
        else:
            # 先加速到 v_max，再匀速
            d_acc = v * t_to_vmax + 0.5 * a_max * t_to_vmax**2
            d_cruise = v_max * (t_eff - t_to_vmax)
            d_max = d_acc + d_cruise

        return d_max >= d

    def reward_stage2_guided(self, v: float, d_tls: float,
                         tls_state: str, t_rem: float, pressure_info, d_lead: float,
                         margin: float = 1.0, r_pos: float = 3.0,
                         r_neg: float = -3.0, slack_tau=None) -> float:
        """
        仅依赖“下一次切换剩余时间 t_rem + 压力推断的续绿概率”，不需要未来相位表。
        - 绿灯：用 期望可用绿窗 G_exp = max(t_rem - margin,0) + p_ext * ext_cap
        与 ETA 比较，赶上→奖，赶不上→罚
        - 红/黄：鼓励 ETA 落在切换之后
        """
        EPS = 1e-3
        if d_tls is None or d_tls == float('inf') or d_tls < 0:
            return 0.0
        v_max = self.vmax
        a_max = self.amax
        ETA = d_tls / max(v, EPS)  # expected pass time
        # p_ext = self.green_extension_prob_from_pressure(pressure_info, k=6.0)
        p_ext = self.green_extension_prob_from_pressure(pressure_info, t_rem=t_rem)
        ext_cap = self.expected_extra_green(p_ext, ext_cap=self.T0, rho=0.5, max_steps=2)

        if tls_state.upper() == 'G':
            G_exp = max(t_rem - margin, 0.0) + ext_cap
            d = d_tls
            if d_lead > 0:
                d = min(d_lead, d_tls)
            can_pass = self.can_pass_green_simple(
                d=d, v=v, t_rem=G_exp, v_max=v_max, a_max=a_max, margin=1.0
            )

            if not can_pass:
                # 2) 判定为“这次绿灯物理上已经赶不上”
                #    → 不再鼓励冲灯，给一个固定惩罚或 0，然后交给红灯逻辑去处理
                return 0  
            
            slack = G_exp - ETA  # positive means that the expected green window is reachable
            # The default time scale is derived from the old reward's common
            # two-sided slope at slack=0: |dR/dslack| = |r|/(G_exp+margin).
            # Thus tau=G_exp+margin preserves the old local sensitivity while
            # removing its value discontinuity and normalization switch.
            tau = (
                G_exp + margin + EPS
                if slack_tau is None
                else max(float(slack_tau), EPS)
            )
            return continuous_slack_reward(
                slack, tau=tau, r_pos=r_pos, r_neg=r_neg
            )
        else:
            t_end = t_rem + self.T0 if tls_state.upper() == 'Y' else t_rem + self.T0*(1-p_ext)
            gap = ETA - (t_end + margin) # 红灯时，ETA要大于剩余的红灯时间，但并不是越大越好
            if gap >= 0:
                # 不早闯是好事，但 gap 越大并不表示越好：用一个指数衰减，让 gap 很大时奖励 → 0，而不是无穷增大
                TAU = t_end + margin + EPS  # 时间尺度
                val = math.exp(-gap / TAU)  # gap=0 时=1，gap 越大越接近0
                return r_pos * float(val)
            # 过早（ETA 太小）→ 惩罚，越早罚越重，上限 |r_neg|
            over_early = -gap  # 越大越早
            return r_neg * min(1.0, over_early / (margin + t_end + EPS))
        
    def compute_stage2_reward(self, obs: dict) -> float:
        """
        需要的 obs 字段：
        v, d_to_isec, tls_state('G'/'Y'/'R'), tls_t_rem,
        pressure_info=[p_cur, p_next, gap]
        """
        v   = obs["v"]
        d   = obs["d_to_isec"]
        st  = obs["tls_state"]
        tR  = max(obs.get("tls_t_rem", 0.0), 0.0)
        pinf = obs["pressure_info"]
        d_lead = obs["distance_to_lead_vehicle"]

        # 引导项（期望绿窗 vs ETA）
        Reff = self.reward_stage2_guided(v=v, d_tls=d, tls_state=st,
                                    t_rem=tR, pressure_info=pinf, d_lead=d_lead)

        # 风险抑制项
        # v_next = max(0.0, v + action*dt)
        # p_ext = self.green_extension_prob_from_pressure(pinf, k=p.get("k_prob", 6.0))
        # Rrisk = self.penalty_red_risk(v_next=v_next, d_tls=d, tls_state=st, p_ext=p_ext,
        #                         t_rem=tR, k_sqrt=p.get("k_sqrt", 0.6),
        #                         w=p.get("w_risk", 0.5), small=p.get("small", 1.5))

        # The green-passage event is added once in the outer reward. Keeping
        # it outside also captures the crossing step after the primary stage
        # may already have changed.
        return Reff
    
    def tls_prepare_threshold(self, v: float,
                          t_window: float = 10.0,   # 时间窗(6~8s 常用)
                          tau: float = 2.0,        # 反应时(s)
                          a_comf: float = 2.0,     # 舒适减速度的绝对值(m/s^2)
                          s0: float = 5.0,         # 静态安全距(m)
                          buffer: float = 5.0,    # 额外裕度(m)
                          d_min: float = 30.0,     # 下限夹取(m)
                          d_max: float = 120.0     # 上限夹取(m)
                          ) -> float:
        """
        自适应“准备通过信号”的距离阈值：
        thr = clip( max(v*t_window, v*tau + v^2/(2*a_comf) + s0 + buffer), [d_min, d_max] )
        """
        v = max(v, 0.0)
        d_time = v * t_window
        d_stop = v * tau + (v * v) / max(2.0 * a_comf, 1e-2) + s0 + buffer
        thr = max(d_time, d_stop)
        return min(max(thr, d_min), d_max)

    def is_stop_intersection_overlap(self, d_to_isec, d_to_stop,
                                     v: float | None,
                                     duration: float = 15.0,
                                     is_at_stop: float = 0.0,
                                     stop_window: float = 50.0,
                                     fallback_tls_thr: float = 80.0) -> bool:
        """Return whether both reward influence regions are active.

        The result affects reward composition only; it never projects or
        replaces the action selected by the agent.
        """
        stop_active = (
            d_to_stop is not None
            and 0.0 <= float(d_to_stop) <= float(stop_window)
            and float(duration) > 14.0
            and float(is_at_stop) < 0.5
        )
        if v is None:
            tls_threshold = float(fallback_tls_thr)
        else:
            tls_threshold = self.tls_prepare_threshold(float(v))
        intersection_active = (
            d_to_isec is not None
            and 0.0 <= float(d_to_isec) <= tls_threshold
        )
        return bool(stop_active and intersection_active)
    
    # ===== 阶段识别 =====
    def classify_stage(self, d_to_isec, d_to_stop,
                    has_stop_ahead: bool,
                    v: float | None = None,
                    duration: float = 15,
                    is_at_stop: float = 0,
                    fallback_tls_thr: float = 80.0,
                    **thr_kwargs) -> int:
        """
        返回主奖励阶段编号。阶段用于选择密集奖励；若车站与交叉口
        影响区重叠，交叉口主阶段会额外保留进站速度引导奖励。
        stage: 2=准备通过信号, 3=路口下游驶往下一站/无目标, 4=进站
        说明:
        - 若车站是路线上的下一控制点且进入窗口 → 阶段4
        - 若信号是下一控制点且进入自适应阈值 → 阶段2
        - 距离相等时阶段4作为确定性回退；重叠 fallback 同样生效
        - 其余 → 阶段3
        额外参数:
        v: 当前车速(m/s)。若为 None，则使用 fallback_tls_thr 作为固定阈值
        fallback_tls_thr: 无法提供速度时的保守固定阈值
        thr_kwargs: 透传给 tls_prepare_threshold（可调 t_window/tau/a_comf 等）
        """
        info = {"thr_tls": None, "why": ""}
        STOP_WINDOW = 50
        # --- 阶段4：进站优先 --- 但是完成停站之后就不属于进站阶段了，不包含在站等待信号灯的阶段
        if d_to_stop is not None and (d_to_stop <= STOP_WINDOW) and\
                        ((d_to_isec is None) or (d_to_stop <= d_to_isec)) and duration > 14.0:
            info["why"] = "near_stop"
            return 4

        # --- 阶段2：准备通过信号（信号在前且更近，且进入准备区） ---
        if d_to_isec is not None:
            # 判断“信号是否更近/是否有站”
            stop_farther = (not has_stop_ahead) or (d_to_stop is None) or (d_to_isec < d_to_stop)
            if stop_farther:
                thr_tls = self.tls_prepare_threshold(v, **thr_kwargs) if (v is not None) else float(fallback_tls_thr)
                info["thr_tls"] = thr_tls
                if d_to_isec <= thr_tls:
                    info["why"] = "approach_tls"
                    return 2
        # --- 阶段1：停在站内，并且未完成dwell
        if is_at_stop == 1 and duration < 15.0 and duration > 0:
            return 1
        # --- 阶段3：其余情况 ---
        info["why"] = "cruise_to_next_stop_or_none"
        return 3
    

    def eval_step(self, signal_policy, bus_policy):
        """
        Evaluation-mode step:
        - 不再计算 RL reward / episode，只推进仿真一步；
        - 调用外部给定的信号控制策略 + 公交控制策略；
        - 返回这一秒的网络层 + 公交层指标。
        """
        if not traci.isLoaded():
            raise RuntimeError(
                "eval_step called but TraCI is not connected. "
                "Did you call env.reset(...) before the eval loop?"
            )

        # 1) 可选清理历史记录（和训练 step 保持一致）
        if self.t0 % self.clean_time_window == 0:
            self._cleanup_old_records()
        
        if not hasattr(self, "_eval_cv_info"):
            self._eval_cv_info = None
            # self._eval_bus_info = []
            # self._eval_bus_ids = []
        CV_info = self._eval_cv_info
        # bus_info = self._eval_bus_info
        # Bus_ids = self._eval_bus_ids

        # 2) 读取当前 CV / 公交信息（和训练里的套路一致）
        desired_type = ["bus_bus", "tram_tram", "veh_CV"]
        # 只需要在初始处更新一次，其余时刻在仿真后马上更新，避免重复
        if CV_info is None:
            vehicle_ids = traci.vehicle.getIDList()
            CV_ids = vehicle_ids
            # CV_ids = [veh for veh in vehicle_ids
            #         if traci.vehicle.getTypeID(veh) in desired_type]

            CV_info = extract_CV_information(
                CV_ids,
                self.link_start_edge,
                self.edge_length_speed,
                self.bus_stop_position,
                self.vehicle_edges,
                self.time_entering_edge,
                self.t0,
                self.T0,
                considerdwell=1,
            )

            # 更新当前所有 corridor 公交 + 对应的 CV 行缓存
            self.Bus_ids = self._get_all_buses(vehicle_ids)  
            cv_row = {row[0]: row for row in CV_info}
            self._bus_info = [cv_row[bid] for bid in self.Bus_ids if bid in cv_row]
            self._eval_cv_info = CV_info

        # 3) 信号控制：交给 signal_policy；若没传策略就按SUMO自带的方式进行控制
        if signal_policy is not None:
            # 只在PMP控制中进行压力更新，SAC也只会在PMP时启用，否则不需要压力值
            weightZZ, weightIM, weightWL = compute_pressure_weights(CV_info, self.in_out, self.sourced_link)
            # self.signal_phase = self.signal_controller.update_signal_phase(
            #     self.t0, self.startopt, self.T0, self.signal_phase, weightZZ, weightIM, weightWL,
            #     pressure=True, signal=False
            # ) # 更新一次压力值，用于后续状态的获取以及信号灯的控制
            if self.t0 > self.startopt:
                self.signal_controller.update_signal_phase(
                    self.t0,
                    self.startopt,
                    self.T0,
                    self.signal_phase,
                    weightZZ,
                    weightIM,
                    weightWL,
                    pressure=True,
                    signal=True,
                )
        
        # 4) 公交控制：交给 bus_policy；未指定则默认不覆盖任何车（SUMO 自己开）
        actions = {}
        if bus_policy is not None: ######## bus_policy 定义在另一个脚本里，这里可以直接调用吗，不需要import吗
            # 约定：bus_policy.step(env, t, Bus_ids, bus_info) -> dict{bus_id: acc} TODO：除了SAC控制，GLOSA与GLODTA同样需要编码
            # 可能需要重新定义一个类——已定义
            alive = set(self.Bus_ids)
            self._update_prev_cache(alive, prune=True, commit=False)  # 只 prune + init，保证新车有 prev
            maybe_actions = bus_policy.step(self, self.t0, self.Bus_ids, self._bus_info)
            if maybe_actions is not None:
                actions = dict(maybe_actions)
            self._update_prev_cache(alive, prune=False, commit=True)  # 把当前写成 prev，供下一步
            for bid, acc in actions.items():
                try:
                    traci.vehicle.setAcceleration(bid, float(acc), 1.0)
                    # traci.vehicle.setAcceleration(bid, 1.0, 1.0)
                except traci.TraCIException:
                    # 车可能已经离网，忽略
                    continue
            
        # 5) 推进一步仿真（唯一的 simulationStep）
        traci.simulationStep()
        self.t0 += 1

        # 6) 再拉一次状态，用于指标统计（“下一秒”的状态）
        vehicle_ids = traci.vehicle.getIDList()
        CV_ids = vehicle_ids
        # CV_ids = [veh for veh in vehicle_ids
        #            if traci.vehicle.getTypeID(veh) in desired_type]
        self.Bus_ids = self._get_all_buses(vehicle_ids) # 更新bus信息，后面的采集数据可能需要

        if signal_policy is not None:
            CV_info = extract_CV_information(
                CV_ids,
                self.link_start_edge,
                self.edge_length_speed,
                self.bus_stop_position,
                self.vehicle_edges,
                self.time_entering_edge,
                self.t0,
                self.T0,
                considerdwell=1,
            )
            self._eval_cv_info = CV_info
            cv_row = {row[0]: row for row in CV_info}
            self._bus_info = [cv_row[bid] for bid in self.Bus_ids if bid in cv_row]

        # 7) 计算这一秒的评估指标（网络层 + 公交层）# 没有公交指标吗；为什么输入里面没有公交list--》在self里面
        metrics = self._collect_eval_metrics(CV_info)

        # 8) 判断仿真是否结束（3小时或 max_sim_time 到）
        world_done = (self.t0 >= self.max_sim_time)
        
        return metrics, world_done

    def _collect_eval_metrics(self, CV_info, dt=1.0, T_hard=2.0):
        """
        每步调用：更新网络与公交的统计量，并返回“网络快照metrics”（方便写timeseries.csv）。
        公交的逐车轨迹会 append 到 self._bus_traj_rows，episode 末再统一落盘/汇总。
        """

        # ---------- 0) cache edges/lanes once ----------
        if not hasattr(self, "_edges"):
            try:
                self._edges = traci.edge.getIDList()
            except traci.TraCIException:
                self._edges = []
        if not hasattr(self, "_lanes"):
            try:
                self._lanes = traci.lane.getIDList()
            except traci.TraCIException:
                self._lanes = []

        t = self.t0

        # ---------- 1) network-level ----------
        try:
            veh_cnt = sum(traci.edge.getLastStepVehicleNumber(e) for e in self._edges)
        except traci.TraCIException:
            veh_cnt = 0

        try:
            queue_cnt = sum(traci.lane.getLastStepHaltingNumber(l) for l in self._lanes)
        except traci.TraCIException:
            queue_cnt = 0

        try:
            pending = len(traci.simulation.getPendingVehicles())
        except traci.TraCIException:
            pending = 0

        # delay proxy: edge waiting time / veh_cnt
        try:
            total_wait = sum(traci.edge.getWaitingTime(e) for e in self._edges)
            avg_delay = float(total_wait) / max(float(veh_cnt), 1.0)
        except traci.TraCIException:
            avg_delay = 0.0

        # spillover proxy: pending
        spillover_cnt = int(pending)

        # update maxima / averages
        self._max_veh_cnt = max(getattr(self, "_max_veh_cnt", 0), int(veh_cnt))
        self._max_queue_cnt = max(getattr(self, "_max_queue_cnt", 0), int(queue_cnt))
        self._max_spillover_cnt = max(getattr(self, "_max_spillover_cnt", 0), spillover_cnt)

        self._delay_sum = getattr(self, "_delay_sum", 0.0) + avg_delay
        self._delay_steps = getattr(self, "_delay_steps", 0) + 1

        # teleport counts（你如果仍想记录）
        try:
            teleports = int(traci.simulation.getStartingTeleportNumber() + traci.simulation.getEndingTeleportNumber())
        except traci.TraCIException:
            teleports = 0

        # ---------- 2) bus-level ----------
        # collisions（建议：先查数量，减少开销）
        try:
            if traci.simulation.getCollidingVehiclesNumber() > 0:
                colliding_ids = set(traci.simulation.getCollidingVehiclesIDList())
            else:
                colliding_ids = set()
        except traci.TraCIException:
            colliding_ids = set()

        # 确保 dict 存在
        for name, default in [
            ("v_prev_by_bus", {}),
            ("a_prev_by_bus", {}),
            ("ttc_under_by_bus", {}),
            ("collision_by_bus", {}),
            ("energy_by_bus", {}),
            ("dist_by_bus", {}),
            ("time_by_bus", {}),
            ("abs_acc_sum_by_bus", {}),
            ("abs_jerk_sum_by_bus", {}),
        ]:
            if not hasattr(self, name) or getattr(self, name) is None:
                setattr(self, name, default.copy())

        bus_cnt = len(getattr(self, "Bus_ids", []))
        try:
            tele_start_ids = set(traci.simulation.getStartingTeleportIDList())
            tele_end_ids   = set(traci.simulation.getEndingTeleportIDList())
            teleported_ids = tele_start_ids.union(tele_end_ids)
        except Exception:
            teleported_ids = set()

        for bid in getattr(self, "Bus_ids", []):
            try:
                v = float(traci.vehicle.getSpeed(bid))
                edge = traci.vehicle.getRoadID(bid)
                lane = traci.vehicle.getLaneID(bid)
                lane_pos = float(traci.vehicle.getLanePosition(bid))

                # ✅ 真实累计距离（odometer）
                try:
                    odom_m = float(traci.vehicle.getDistance(bid))
                except Exception:
                    odom_m = np.nan

                # line id
                try:
                    line_id = traci.vehicle.getLine(bid)
                except Exception:
                    line_id = ""

                # ---- accel & jerk (from v_prev / a_prev) ----
                v_prev = float(self.v_prev_by_bus.get(bid, v))
                a = (v - v_prev) / dt

                a_prev = float(self.a_prev_by_bus.get(bid, a))
                jerk = (a - a_prev) / dt

                # ---- TTC ----
                ttc = float("inf")
                try:
                    leader = traci.vehicle.getLeader(bid, 200.0)
                    if leader is not None:
                        leader_id, gap = leader
                        v_lead = float(traci.vehicle.getSpeed(leader_id))
                        dv = v - v_lead
                        if dv > 1e-3:
                            ttc = float(gap) / dv
                except Exception:
                    pass

                # ---- collision ----
                collided = (bid in colliding_ids)
                if collided:
                    self.collision_by_bus[bid] = int(self.collision_by_bus.get(bid, 0)) + 1

                # ---- TTC under threshold duration ----
                if ttc < T_hard:
                    self.ttc_under_by_bus[bid] = float(self.ttc_under_by_bus.get(bid, 0.0)) + dt

                # ---- energy（你原逻辑保留）----
                e = float(self.energy_consumption(v_prev, v, a))
                self.energy_by_bus[bid] = float(self.energy_by_bus.get(bid, 0.0)) + e

                # ---- accumulate stats ----
                self.dist_by_bus[bid] = float(self.dist_by_bus.get(bid, 0.0)) + v * dt
                self.time_by_bus[bid] = float(self.time_by_bus.get(bid, 0.0)) + dt
                self.abs_acc_sum_by_bus[bid] = float(self.abs_acc_sum_by_bus.get(bid, 0.0)) + abs(a)
                self.abs_jerk_sum_by_bus[bid] = float(self.abs_jerk_sum_by_bus.get(bid, 0.0)) + abs(jerk)

                # ✅ ---- TLS snapshot（actuated 也适用：记录“真实发生的状态”）----
                tls_id = None
                tls_idx = None
                tls_dist_m = None
                tls_state = None
                tls_state_full = None
                tls_time_to_switch_s = None

                info = self._next_tls_info(bid)
                if info is not None:
                    tls_id, tls_idx, tls_dist_m, tls_state, _ = info
                    # lane 对应的全量状态串（更可靠）
                    try:
                        tls_state_full = traci.trafficlight.getRedYellowGreenState(tls_id)
                    except Exception:
                        tls_state_full = None
                    # 下一次切换时间（actuated 会动态变化，但每步都能取到当下值）
                    try:
                        nxt = float(traci.trafficlight.getNextSwitch(tls_id))
                        tls_time_to_switch_s = max(0.0, nxt - float(traci.simulation.getTime()))
                    except Exception:
                        tls_time_to_switch_s = None

                # ---- write trajectory row ----
                self._bus_traj_rows.append(dict(
                    t=t, bus_id=bid, line_id=line_id,
                    edge=edge, lane=lane, lane_pos=lane_pos,
                    speed=v, accel=a, jerk=jerk,
                    odom_m=odom_m,                       # ✅新增
                    ttc=ttc, collided=int(collided),
                    energy=e,
                    teleported=int(bid in teleported_ids),  # ✅新增（建议）
                    tls_id=tls_id, tls_idx=tls_idx,         # ✅新增
                    tls_dist_m=tls_dist_m,
                    tls_state=tls_state,
                    tls_state_full=tls_state_full,
                    tls_time_to_switch_s=tls_time_to_switch_s,
                ))

            except traci.TraCIException:
                continue

        # 这里返回“网络快照”（你现有 timeseries.csv 就写这个）
        metrics = dict(
            t=t,
            avg_delay=float(avg_delay),
            veh_cnt=int(veh_cnt),
            queue_cnt=int(queue_cnt),
            spillover_cnt=int(spillover_cnt),
            teleports=int(teleports),
            bus_cnt=int(bus_cnt),
        )
        return metrics

    def debug_log_stop_reward(self, next_state, focus_id, action):
        """
        仅用于部署期调试：对“异常停车”步打印reward分量。
        """
        distance_to_lead_vehicle = next_state[8]*10
        lead_vehicle_speed = next_state[9]*10
        current_speed = next_state[1]*10
        current_acceleration = next_state[2]*2
        distance_to_intersection = next_state[0]*500
        distance_to_next_bus_stop = next_state[11]*500
        pressure_info = [next_state[5]*10, next_state[6]*10, next_state[7]*10]
        dec_remain = next_state[4]*10
        duration = next_state[13]*10
        is_at_stop = next_state[12]
        # 信号灯状态判定
        light_state = next_state[3]
        if light_state == 0:
            tls_state = 'G'
        elif light_state == 1:
            tls_state = 'Y'
        elif light_state == 2:
            tls_state = 'R'
        # 碰撞风险判定
        dv = current_speed - lead_vehicle_speed
        if distance_to_lead_vehicle < 0:
            TTC = float('inf')
        else:
            d_eff = max(distance_to_lead_vehicle - 1.0, 0.0)
            TTC = d_eff / max(dv, 0.1) if dv > 0 else float('inf')
        TTC = min(TTC, 10.0)
        # 距离
        d_to_isec = None if distance_to_intersection < 0  else distance_to_intersection
        if distance_to_next_bus_stop < 0:
            d_to_stop = None
            has_stop_ahead = False
        else:
            d_to_stop = distance_to_next_bus_stop
            has_stop_ahead = True
        # 绿灯过路口判定
        passed_on_green = False # 默认为否 # 这个判定有问题，通过路口后当前路口会被移除
        current_lane = traci.vehicle.getLaneID(focus_id)
        if (current_lane.startswith(":")) and (self.dtls_prev >= 0 and self.dtls_prev < 20) and (self.tls_prev in (0,1))\
              and (current_speed > 2.0):
            passed_on_green = True
        # 是否在站内停车判定
        if is_at_stop:
            self.isatstop = True
        elif self.isatstop and current_speed == 0:
            self.isatstop = True
        else:
            self.isatstop = False
        # 判定公交的阶段
        current_stage = self.classify_stage(d_to_isec, d_to_stop, has_stop_ahead, current_speed, duration, self.isatstop)
        stop_intersection_overlap = self.is_stop_intersection_overlap(
            d_to_isec, d_to_stop, current_speed, duration, self.isatstop
        )
        # 碰撞/平滑/停车/安全/能耗/效率
        colliding_ids = set(traci.simulation.getCollidingVehiclesIDList())
        collided = (focus_id in colliding_ids)
        R_sat = self.reward_speed_saturation(current_speed, self.vmax, action)
        Rcol = self.reward_collision(collided, w_collision=12.0)
        try:
            a_prev = self.a_prev_by_bus[focus_id]
        except traci.TraCIException:
            a_prev = 0
        try:
            v_prev = self.v_prev_by_bus[focus_id]
        except traci.TraCIException:
            v_prev = 0
        Rj  = self.reward_jerk(current_acceleration, a_prev)
        Rsf = self.reward_safety_brake(current_acceleration)
        Rst = self.reward_stop_penalty(current_speed, lead_vehicle_speed, distance_to_lead_vehicle, 
                                       self.isatstop, tls_state, d_to_isec)
        Reng = self.energy_reward(
            v_pre=v_prev, v_now=max(0.0, current_speed), a=current_acceleration
        )
        Rgreen = self.reward_stage2_events(passed_on_green)
        # Reff = 0.0
        R_stage_2 = 0
        R_stage_3 = 0
        R_stage_4 = 0
        if current_stage == 2: # 过路口
            self.update_bus_speed_mode(focus_id, d_to_isec)
            obs = dict(
                v=current_speed,
                d_to_isec=d_to_isec,
                tls_state=tls_state,
                tls_t_rem=dec_remain,  # 在stage2则是正常数字
                pressure_info=pressure_info,
                distance_to_lead_vehicle=distance_to_lead_vehicle
            )
            R_stage_2 = self.compute_stage2_reward(obs)
            if stop_intersection_overlap and not self.in_auto_stop_zone(d_to_stop, current_speed):
                R_stage_4 = self.reward_stage4(
                    current_speed, current_acceleration, d_to_stop, duration,
                    distance_to_lead_vehicle, is_at_stop
                )
        elif current_stage == 3:  # 自由行驶
            r_TTC = self.reward_stage3_TTC(TTC)
            r_prog = self.reward_stage3_progress(current_speed)
            R_stage_3 = r_TTC + r_prog
        elif current_stage == 4: # 进站
            auto_stop_zone = self.in_auto_stop_zone(d_to_stop, current_speed)
            if auto_stop_zone:
                R_stage_4 = 0
            else:
                R_stage_4 = self.reward_stage4(current_speed, current_acceleration, 
                                        d_to_stop, duration, distance_to_lead_vehicle, is_at_stop)
        else:  # stage 1
            R_stage_1 = 0
        Reff = R_stage_2 + R_stage_3 + R_stage_4
        r_time = -0.1
        reward = Reff + Reng + Rcol + Rj + Rsf + Rst + Rgreen + R_sat + r_time
        veh_id  = focus_id
        lane_id = current_lane
        t0      = self.t0
        self.a_prev = current_acceleration
        self.v_prev = current_speed
        self.dtls_prev = distance_to_intersection
        self.dtbstop_prev = d_to_stop
        self.tls_prev = light_state

        print(f"[DEBUG STOP] t={t0:.1f} veh={veh_id} lane={lane_id}")
        print(
            f"  v={current_speed:.2f}, a_rl={current_acceleration:.2f}, "
            f"a_proj={current_acceleration:.2f}, "
            f"d_stop={distance_to_next_bus_stop:.1f}, d_tls={distance_to_intersection:.1f}, "
            f"tls={tls_state}, t_rem={dec_remain:.1f}"
        )
        print(f"  Stage={current_stage}, Action={action:.2f}")
        print(
            f"  r_stop={Rst:.2f}, r_stage_2={R_stage_2:.2f}, r_stage_3={R_stage_3:.2f}, "
            f"r_stage_4={R_stage_4:.2f}, r_sat={R_sat:.2f} "
        )
        print(
            f"  r_energy={Reng:.2f}, r_brake={Rsf:.2f}, r_collision={Rcol:.2f}, r_jerk={Rj:.2f}, "
            f"r_pass_green={Rgreen:.2f}, "
        )
        print(
            f"  r_sat={R_sat:.2f}, r_time_penalty={r_time:.2f}, R_total={reward:.2f}"
        )
        print(f"  green_pass={passed_on_green}, duration={duration:.1f}")
        print("-" * 60)

        return None
    
    def stop_debug(self, next_state):
    # 在非站点位置速度近似为0时的惩罚，拥堵，等红灯的情况除外，v_lead > 0.5 or <0 表示未拥堵
        d_lead = next_state[8]*10
        v_lead = next_state[9]*10
        v_now = next_state[1]*10
        current_acceleration = next_state[2]*2
        d_to_isec = next_state[0]*500
        distance_to_next_bus_stop = next_state[11]*500
        pressure_info = [next_state[5]*10, next_state[6]*10, next_state[7]*10]
        dec_remain = next_state[4]*10
        duration = next_state[13]*10
        is_at_stop = next_state[12]
        # 信号灯状态判定
        light_state = next_state[3]
        if is_at_stop == 1: #在站判定
            return False
        if v_lead > 0 and v_lead < 0.5 and d_lead < 10.0:   # 拥堵判定
            return False
        if light_state == 2.0 and (d_to_isec >= 0.0) and (d_to_isec < 20):  # 等红灯判定
            return False
        if v_now < 0.5:
            return True
        return False
    
    def _update_prev_cache(self, alive: set, prune: bool = False, commit: bool = False, dt: float = 1.0):
        """
        alive: 当前仍在网内的 bus set
        commit=False: 只 prune + init（不覆盖已有 prev）
        commit=True : 覆盖写入当前 v/a，供下一步使用
        """
        if prune:
            # 1) prune
            self.v_prev_by_bus = {bid: v for bid, v in self.v_prev_by_bus.items() if bid in alive}
            self.a_prev_by_bus = {bid: a for bid, a in self.a_prev_by_bus.items() if bid in alive}

            # 2) init only（关键：只对新出现的车补齐，不覆盖老车）
            for bid in alive:
                if bid not in self.v_prev_by_bus:
                    try:
                        self.v_prev_by_bus[bid] = traci.vehicle.getSpeed(bid)
                    except traci.TraCIException:
                        self.v_prev_by_bus[bid] = 0.0
                if bid not in self.a_prev_by_bus:
                    self.a_prev_by_bus[bid] = 0.0

        if not commit:
            return

        # 3) commit：覆盖写入“本步的当前值”，留给下一步用
        for bid in alive:
            try:
                v_now = traci.vehicle.getSpeed(bid)
                v_prev = float(self.v_prev_by_bus.get(bid, v_now))
                a_now = (v_now - v_prev) / max(dt, 1e-6)
                self.v_prev_by_bus[bid] = float(v_now)
                self.a_prev_by_bus[bid] = float(a_now)
            except traci.TraCIException:
                pass

    def _next_tls_info(self, bid: str):
        """
        兼容 SUMO traci.vehicle.getNextTLS() 返回 4 或 5 个字段的差异
        return: (tls_id, tls_index, dist_tls, state_char, next_switch_time_or_nan)
        """
        try:
            tls_list = traci.vehicle.getNextTLS(bid)
        except traci.TraCIException:
            return None
        if not tls_list:
            return None

        item = tls_list[0]
        if len(item) == 4:
            tls_id, tls_index, dist_tls, state_char = item
            next_sw = float("nan")
        else:
            tls_id, tls_index, dist_tls, state_char, next_sw = item

        return str(tls_id), int(tls_index), float(dist_tls), str(state_char), float(next_sw)


    def export_bus_traj(self, out_csv: str):
        import pandas as pd
        if not hasattr(self, "_bus_traj_rows") or len(self._bus_traj_rows) == 0:
            pd.DataFrame([]).to_csv(out_csv, index=False)
            return
        pd.DataFrame(self._bus_traj_rows).to_csv(out_csv, index=False)

    def export_bus_summary(self, out_csv: str):
        import pandas as pd
        # 以出现过的 bus_id 为全集
        bus_ids = set()
        for d in [self.time_by_bus, self.dist_by_bus, self.energy_by_bus,
                self.abs_acc_sum_by_bus, self.abs_jerk_sum_by_bus,
                self.collision_by_bus, self.ttc_under_by_bus]:
            bus_ids |= set(d.keys())

        rows = []
        for bid in sorted(bus_ids):
            T = float(self.time_by_bus.get(bid, 0.0))
            D = float(self.dist_by_bus.get(bid, 0.0))
            E = float(self.energy_by_bus.get(bid, 0.0))
            acc_sum = float(self.abs_acc_sum_by_bus.get(bid, 0.0))
            jerk_sum = float(self.abs_jerk_sum_by_bus.get(bid, 0.0))
            coll = int(self.collision_by_bus.get(bid, 0))
            ttc_under = float(self.ttc_under_by_bus.get(bid, 0.0))

            rows.append(dict(
                bus_id=bid,
                time_s=T,
                dist_m=D,
                avg_speed_mps=(D / T) if T > 1e-6 else 0.0,
                avg_abs_acc=(acc_sum / T) if T > 1e-6 else 0.0,
                avg_abs_jerk=(jerk_sum / T) if T > 1e-6 else 0.0,
                energy=E,
                collisions=coll,
                ttc_under_s=ttc_under,
                ttc_under_ratio=(ttc_under / T) if T > 1e-6 else 0.0,
            ))
        pd.DataFrame(rows).to_csv(out_csv, index=False)
