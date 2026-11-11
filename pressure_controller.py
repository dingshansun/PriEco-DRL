import numpy as np
import traci


def compute_pressure_weights(CV_info, in_out, sourced_link):
    # Initialize weight matrices
    ##intersection ZZ (16 movements), IM (22 movements), WL (24 movements)
                    #incoming link state, outgoing link state (L, S, R), pressure: 
                        #PTT-based (0-4), TT-based (5-9), original (10-14), position-weighted (15-19)
                        #saturated flow rate 20
    weightZZ = np.zeros((16, 21))
    weightZZ[:, 20] = 0.5
    weightZZ[[4, 11, 12], 20] = 0.4

    weightIM = np.zeros((22, 21))
    weightIM[:, 20] = 0.5
    weightIM[[3, 4, 5, 10, 14, 15, 16, 18, 21], 20] = 0.4

    weightWL = np.zeros((24, 21))
    weightWL[:, 20] = 0.5
    weightWL[[3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23], 20] = 0.4

    # Update weight matrices with each vehicle info
    for vehinfo in CV_info:
        # vehinfo[4] = 1  # no priority weighting for now
        weightZZ, weightIM, weightWL = incoming_state(vehinfo, weightZZ, weightIM, weightWL, sourced_link)
        weightZZ, weightIM, weightWL = outgoing_state(vehinfo, in_out, weightZZ, weightIM, weightWL)

    # Compute stream pressures
    weightZZ = stream_pressure(weightZZ)
    weightIM = stream_pressure(weightIM)
    weightWL = stream_pressure(weightWL)

    return weightZZ, weightIM, weightWL

def get_phase_pressures(current_tls_id, signal_phase, max_num_phases=5):
    """
    提取当前交通灯控制器所有 phase 的压力值（如 PTT-based 压力 signal_phase[:, 4]）
    返回长度为 max_num_phases 的 list，不足的补 0
    """
    # 获取当前交通灯控制器的所有相位压力
    pressures = [
        float(row[4])
        for row in signal_phase
        if row[0] == current_tls_id.split('_')[1]
    ]
    # 补零对齐长度
    if len(pressures) < max_num_phases:
        pressures.extend([-10.0] * (max_num_phases - len(pressures)))
    return pressures[:max_num_phases]


def incoming_state(vehinfo, weightZZ, weightIM, weightWL, sourced_link):
    if vehinfo[1] in sourced_link:
        sourced_or_not = 0# the vehicle is on a sourced link
    else:
        sourced_or_not = 1# the vehicle is not on a sourced link
    if vehinfo[5] == 'TL_ZZ':
        weightZZ[vehinfo[6],0] = weightZZ[vehinfo[6],0]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
        weightZZ[vehinfo[6],5] = weightZZ[vehinfo[6],5]+vehinfo[3]/vehinfo[10]
        weightZZ[vehinfo[6],10] = weightZZ[vehinfo[6],10]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
        weightZZ[vehinfo[6],15] = weightZZ[vehinfo[6],15]+(1-sourced_or_not*vehinfo[7]/vehinfo[9])*vehinfo[4]*vehinfo[11]
    elif vehinfo[5] == 'TL_IM':
        weightIM[vehinfo[6],0] = weightIM[vehinfo[6],0]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
        weightIM[vehinfo[6],5] = weightIM[vehinfo[6],5]+vehinfo[3]/vehinfo[10]
        weightIM[vehinfo[6],10] = weightIM[vehinfo[6],10]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
        weightIM[vehinfo[6],15] = weightIM[vehinfo[6],15]+(1-sourced_or_not*vehinfo[7]/vehinfo[9])*vehinfo[4]*vehinfo[11]
    elif vehinfo[5] == 'TL_WL':
        weightWL[vehinfo[6],0] = weightWL[vehinfo[6],0]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
        weightWL[vehinfo[6],5] = weightWL[vehinfo[6],5]+vehinfo[3]/vehinfo[10]
        weightWL[vehinfo[6],10] = weightWL[vehinfo[6],10]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
        weightWL[vehinfo[6],15] = weightWL[vehinfo[6],15]+(1-sourced_or_not*vehinfo[7]/vehinfo[9])*vehinfo[4]*vehinfo[11]
    return weightZZ, weightIM, weightWL

def outgoing_state(vehinfo, in_out, weightZZ, weightIM, weightWL):
    # out_index = in_out[:,0]==vehinfo[1]
    incoming_edge = str(vehinfo[1])
    out_node, out_streams = in_out.get(incoming_edge, (None, ()))
    if out_node is not None:
        if out_node == 'ZZ':
            if vehinfo[6]<0:# the vehicle is on the edge leaving the network
                for stream in out_streams:
                    weightZZ[int(stream),1]=weightZZ[int(stream),1]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
                    weightZZ[int(stream),2]=weightZZ[int(stream),2]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
                    weightZZ[int(stream),3]=weightZZ[int(stream),3]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
                    weightZZ[int(stream),6]=weightZZ[int(stream),6]+vehinfo[3]/vehinfo[10]
                    weightZZ[int(stream),7]=weightZZ[int(stream),7]+vehinfo[3]/vehinfo[10]
                    weightZZ[int(stream),8]=weightZZ[int(stream),8]+vehinfo[3]/vehinfo[10]
                    weightZZ[int(stream),11]=weightZZ[int(stream),11]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
                    weightZZ[int(stream),12]=weightZZ[int(stream),12]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
                    weightZZ[int(stream),13]=weightZZ[int(stream),13]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
                    weightZZ[int(stream),16]=weightZZ[int(stream),16]+vehinfo[7]*vehinfo[4]/vehinfo[9]
                    weightZZ[int(stream),17]=weightZZ[int(stream),17]+vehinfo[7]*vehinfo[4]/vehinfo[9]
                    weightZZ[int(stream),18]=weightZZ[int(stream),18]+vehinfo[7]*vehinfo[4]/vehinfo[9]
            elif vehinfo[6] in [3,7,10,15]: #the vehicle is on a internel edge, left turn
                for stream in out_streams:
                    weightZZ[int(stream),1]=weightZZ[int(stream),1]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
                    weightZZ[int(stream),6]=weightZZ[int(stream),6]+vehinfo[3]/vehinfo[10]
                    weightZZ[int(stream),11]=weightZZ[int(stream),11]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
                    weightZZ[int(stream),16]=weightZZ[int(stream),16]+vehinfo[7]*vehinfo[4]/vehinfo[9]*vehinfo[11]
            elif vehinfo[6] in [0,5,8,13]: #the vehicle is on a internel edge, right turn
                for stream in out_streams:
                    weightZZ[int(stream),3]=weightZZ[int(stream),3]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
                    weightZZ[int(stream),8]=weightZZ[int(stream),8]+vehinfo[3]/vehinfo[10]
                    weightZZ[int(stream),13]=weightZZ[int(stream),13]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
                    weightZZ[int(stream),18]=weightZZ[int(stream),18]+vehinfo[7]*vehinfo[4]/vehinfo[9]*vehinfo[11]
            else: #the vehicle is on a internel edge, straight through
                for stream in out_streams:
                    weightZZ[int(stream),2]=weightZZ[int(stream),2]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
                    weightZZ[int(stream),7]=weightZZ[int(stream),7]+vehinfo[3]/vehinfo[10]
                    weightZZ[int(stream),12]=weightZZ[int(stream),12]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
                    weightZZ[int(stream),17]=weightZZ[int(stream),17]+vehinfo[7]*vehinfo[4]/vehinfo[9]*vehinfo[11]
        elif out_node == 'IM':
            if vehinfo[6]<0:# the vehicle is on the edge leaving the network
                for stream in out_streams:
                    weightIM[int(stream),1]=weightIM[int(stream),1]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
                    weightIM[int(stream),2]=weightIM[int(stream),2]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
                    weightIM[int(stream),3]=weightIM[int(stream),3]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
                    weightIM[int(stream),6]=weightIM[int(stream),6]+vehinfo[3]/vehinfo[10]
                    weightIM[int(stream),7]=weightIM[int(stream),7]+vehinfo[3]/vehinfo[10]
                    weightIM[int(stream),8]=weightIM[int(stream),8]+vehinfo[3]/vehinfo[10]
                    weightIM[int(stream),11]=weightIM[int(stream),11]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
                    weightIM[int(stream),12]=weightIM[int(stream),12]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
                    weightIM[int(stream),13]=weightIM[int(stream),13]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
                    weightIM[int(stream),16]=weightIM[int(stream),16]+vehinfo[7]*vehinfo[4]/vehinfo[9]
                    weightIM[int(stream),17]=weightIM[int(stream),17]+vehinfo[7]*vehinfo[4]/vehinfo[9]
                    weightIM[int(stream),18]=weightIM[int(stream),18]+vehinfo[7]*vehinfo[4]/vehinfo[9]
            elif vehinfo[6] in [2,6,11,12,15,18,19,23,24]: #the vehicle is on a internel edge, left turn
                for stream in out_streams:
                    weightIM[int(stream),1]=weightIM[int(stream),1]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
                    weightIM[int(stream),6]=weightIM[int(stream),6]+vehinfo[3]/vehinfo[10]
                    weightIM[int(stream),11]=weightIM[int(stream),11]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
                    weightIM[int(stream),16]=weightIM[int(stream),16]+vehinfo[7]*vehinfo[4]/vehinfo[9]*vehinfo[11]
            elif vehinfo[6] in [0,3,4,7,8,13,16,20,21,25]: #the vehicle is on a internel edge, right turn
                for stream in out_streams:
                    weightIM[int(stream),3]=weightIM[int(stream),3]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
                    weightIM[int(stream),8]=weightIM[int(stream),8]+vehinfo[3]/vehinfo[10]
                    weightIM[int(stream),13]=weightIM[int(stream),13]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
                    weightIM[int(stream),18]=weightIM[int(stream),18]+vehinfo[7]*vehinfo[4]/vehinfo[9]*vehinfo[11]
            else: #the vehicle is on a internel edge, straight through
                for stream in out_streams:
                    weightIM[int(stream),2]=weightIM[int(stream),2]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
                    weightIM[int(stream),7]=weightIM[int(stream),7]+vehinfo[3]/vehinfo[10]
                    weightIM[int(stream),12]=weightIM[int(stream),12]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
                    weightIM[int(stream),17]=weightIM[int(stream),17]+vehinfo[7]*vehinfo[4]/vehinfo[9]*vehinfo[11]
        elif out_node == 'WL':
            if vehinfo[6]<0:# the vehicle is on the edge leaving the network
                for stream in out_streams:
                    weightWL[int(stream),1]=weightWL[int(stream),1]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
                    weightWL[int(stream),2]=weightWL[int(stream),2]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
                    weightWL[int(stream),3]=weightWL[int(stream),3]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
                    weightWL[int(stream),6]=weightWL[int(stream),6]+vehinfo[3]/vehinfo[10]
                    weightWL[int(stream),7]=weightWL[int(stream),7]+vehinfo[3]/vehinfo[10]
                    weightWL[int(stream),8]=weightWL[int(stream),8]+vehinfo[3]/vehinfo[10]
                    weightWL[int(stream),11]=weightWL[int(stream),11]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
                    weightWL[int(stream),12]=weightWL[int(stream),12]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
                    weightWL[int(stream),13]=weightWL[int(stream),13]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
                    weightWL[int(stream),16]=weightWL[int(stream),16]+vehinfo[7]*vehinfo[4]/vehinfo[9]
                    weightWL[int(stream),17]=weightWL[int(stream),17]+vehinfo[7]*vehinfo[4]/vehinfo[9]
                    weightWL[int(stream),18]=weightWL[int(stream),18]+vehinfo[7]*vehinfo[4]/vehinfo[9]
            elif vehinfo[6] in [2,6,9,12,15,18,21,24]: #the vehicle is on a internel edge, left turn
                for stream in out_streams:
                    weightWL[int(stream),1]=weightWL[int(stream),1]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
                    weightWL[int(stream),6]=weightWL[int(stream),6]+vehinfo[3]/vehinfo[10]
                    weightWL[int(stream),11]=weightWL[int(stream),11]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
                    weightWL[int(stream),16]=weightWL[int(stream),16]+vehinfo[7]*vehinfo[4]/vehinfo[9]*vehinfo[11]
            elif vehinfo[6] in [0,3,7,10,13,16,19,22]: #the vehicle is on a internel edge, right turn
                for stream in out_streams:
                    weightWL[int(stream),3]=weightWL[int(stream),3]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
                    weightWL[int(stream),8]=weightWL[int(stream),8]+vehinfo[3]/vehinfo[10]
                    weightWL[int(stream),13]=weightWL[int(stream),13]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
                    weightWL[int(stream),18]=weightWL[int(stream),18]+vehinfo[7]*vehinfo[4]/vehinfo[9]*vehinfo[11]
            else: #the vehicle is on a internel edge, straight through
                for stream in out_streams:
                    weightWL[int(stream),2]=weightWL[int(stream),2]+vehinfo[3]*vehinfo[4]/vehinfo[10]*vehinfo[11]
                    weightWL[int(stream),7]=weightWL[int(stream),7]+vehinfo[3]/vehinfo[10]
                    weightWL[int(stream),12]=weightWL[int(stream),12]+1*vehinfo[4]/np.sqrt(vehinfo[9])*vehinfo[11]
                    weightWL[int(stream),17]=weightWL[int(stream),17]+vehinfo[7]*vehinfo[4]/vehinfo[9]*vehinfo[11]
    return weightZZ, weightIM, weightWL

def stream_pressure(weight):
    total = weight[:,11] + weight[:,12] + weight[:,13]
    for i in range(len(weight[:,0])):
        if total[i]>0:
            left = weight[i,11]/total[i]
            straight = weight[i,12]/total[i]
            right = weight[i,13]/total[i]
            weight[i,1] = weight[i,1]*left
            weight[i,2] = weight[i,2]*straight
            weight[i,3] = weight[i,3]*right
            weight[i,6] = weight[i,6]*left
            weight[i,7] = weight[i,7]*straight
            weight[i,8] = weight[i,8]*right
            weight[i,11] = weight[i,11]*left
            weight[i,12] = weight[i,12]*straight
            weight[i,13] = weight[i,13]*right
            weight[i,16] = weight[i,16]*left
            weight[i,17] = weight[i,17]*straight
            weight[i,18] = weight[i,18]*right
        else:
            continue
    weight[:,4] = (weight[:,0] - (weight[:,1]+weight[:,2]+weight[:,3]))*weight[:,20]
    weight[:,9] = (weight[:,5] - (weight[:,6]+weight[:,7]+weight[:,8]))*weight[:,20]
    weight[:,14] = (weight[:,10] - (weight[:,11]+weight[:,12]+weight[:,13]))*weight[:,20]
    weight[:,19] = abs(weight[:,15] - (weight[:,16]+weight[:,17]+weight[:,18]))*weight[:,20]
    return weight
            

def phase_pressure(signal_phase, weightZZ, weightIM, weightWL):
    '''
    <tlLogic id="TL_IM" type="actuated" programID="0" offset="0">
        <phase duration="16" state="gggGGGrrrrrrrrrrrrrrrr" minDur="5" maxDur="50"/>
        <phase duration="3"  state="yyyyyyrrrrrrrrrrrrrrrr"/>
        <phase duration="16" state="rrrrrrGGGGGrrrrrrrrrrr" minDur="5" maxDur="50"/>
        <phase duration="3"  state="rrrrrryyyyyrrrrrrrrrrr"/>
        <phase duration="16" state="rrrrrrrrrrrgggGGGrrrrr" minDur="5" maxDur="50"/>
        <phase duration="3"  state="rrrrrrrrrrryyyyyyrrrrr"/>
        <phase duration="16" state="rrrrrrrrrrrrrrrrrGGGGG" minDur="5" maxDur="50"/>
        <phase duration="3"  state="rrrrrrrrrrrrrrrrryyyyy"/>
    </tlLogic>
    <tlLogic id="TL_WL" type="actuated" programID="0" offset="0">
        <phase duration="12" state="gggGGGrrrrrrrrrrrrrrrrrr" minDur="5" maxDur="50"/>
        <phase duration="3"  state="yyyyyyrrrrrrrrrrrrrrrrrr"/>
        <phase duration="14" state="rrrrrrgggGGGrrrrrrrrrrrr" minDur="5" maxDur="50"/>
        <phase duration="3"  state="rrrrrryyyyyyrrrrrrrrrrrr"/>
        <phase duration="14" state="rrrrrrrrrrrrgggGGGrrrrrr" minDur="5" maxDur="50"/>
        <phase duration="3"  state="rrrrrrrrrrrryyyyyyrrrrrr"/>
        <phase duration="14" state="rrrrrrrrrrrrrrrrrrgggGGG" minDur="5" maxDur="50"/>
        <phase duration="3"  state="rrrrrrrrrrrrrrrrrryyyyyy"/>
    </tlLogic>
    <tlLogic id="TL_ZZ" type="actuated" programID="0" offset="0">
        <phase duration="15" state="GGGGrrrrrrrrrrrr" minDur="5" maxDur="50"/>
        <phase duration="3"  state="yyyyrrrrrrrrrrrr" next="2 6 8"/>
        <phase duration="6"  state="rrrrrrrrGGGrrrrr" minDur="5" maxDur="50"/>
        <phase duration="3"  state="rrrrrrrryyyrrrrr" next="4 6 8 0"/>
        <phase duration="15" state="rrrrGrrrrrrGgrrr" minDur="5" maxDur="50"/>
        <phase duration="3"  state="rrrryrrrrrryyrrr"/>
        <phase duration="15" state="rrrrrGGGrrrrrrrr" minDur="5" maxDur="50"/>
        <phase duration="3"  state="rrrrryyyrrrrrrrr" next="8 0 2"/>
        <phase duration="6"  state="rrrrrrrrrrrrrGGG" minDur="5" maxDur="50"/>
        <phase duration="3"  state="rrrrrrrrrrrrryyy" next="0 2 6"/>
    </tlLogic>
    '''
    for i in range(len(signal_phase[:,0])):
        node = signal_phase[i,0]
        phase = signal_phase[i,1]
        #sr = signal_phase[i,3]
        indices = [i for i, char in enumerate(phase) if char != 'r']
        if node == 'ZZ':
            for stream in indices:
                signal_phase[i,4] = signal_phase[i,4] + weightZZ[stream,4]
                signal_phase[i,5] = signal_phase[i,5] + weightZZ[stream,9]
                signal_phase[i,6] = signal_phase[i,6] + weightZZ[stream,14]
                signal_phase[i,7] = signal_phase[i,7] + weightZZ[stream,19]
        elif node == 'IM':
            for stream in indices:
                signal_phase[i,4] = signal_phase[i,4] + weightIM[stream,4]
                signal_phase[i,5] = signal_phase[i,5] + weightIM[stream,9]
                signal_phase[i,6] = signal_phase[i,6] + weightIM[stream,14]
                signal_phase[i,7] = signal_phase[i,7] + weightIM[stream,19]
        elif node == 'WL':
            for stream in indices:
                signal_phase[i,4] = signal_phase[i,4] + weightWL[stream,4]
                signal_phase[i,5] = signal_phase[i,5] + weightWL[stream,9]
                signal_phase[i,6] = signal_phase[i,6] + weightWL[stream,14]
                signal_phase[i,7] = signal_phase[i,7] + weightWL[stream,19]
    
    return signal_phase

class SignalController:
    def __init__(self, losstime: int):
        # 你的初值，原样沿用
        self.last_phase     = {'TL_ZZ': -1,  'TL_IM': -1,  'TL_WL': -1}
        self.yellow_flag    = {'TL_ZZ': False,'TL_IM': False,'TL_WL': False}
        self.yellow_counter = {'TL_ZZ': 0,    'TL_IM': 0,    'TL_WL': 0}
        self.pending_phase  = {'TL_ZZ': -1,   'TL_IM': -1,   'TL_WL': -1}
        self.pending_state  = {'TL_ZZ': None, 'TL_IM': None, 'TL_WL': None}
        self.yellow_duration = losstime
        # 由键集得到 tl 列表，后续遍历就用它
        self.tl_ids = list(self.last_phase.keys())

    def _get_yellow_state(self, last_phase_idx, signal_phase):
        if last_phase_idx < 0:
            return None
        return signal_phase[last_phase_idx, 1].replace('G', 'y').replace('g', 'y')

    def _scale_pressures(self, signal_phase, node_token, last_phase_idx, losstime, T0):
        """node_token 用 'ZZ'/'IM'/'WL'，而不是 'TL_ZZ'。"""
        scale = (T0 - losstime) / T0
        # 所有该路口候选相位按可用绿时间缩放
        for i in range(signal_phase.shape[0]):
            if signal_phase[i, 0] == node_token:
                signal_phase[i, 4] *= scale
        # 上一相位反缩放（上一周期已经消耗黄灯时间）
        if last_phase_idx >= 0:
            # 只反缩放该路口的上一相位
            if signal_phase[last_phase_idx, 0] == node_token:
                signal_phase[last_phase_idx, 4] /= scale
        return signal_phase
    

    def update_signal_phase(self, t0, startopt, T0, signal_phase, weightZZ, weightIM, weightWL, pressure, signal):
        # 1) 先把压力写入 signal_phase（每秒一次）
        if pressure:
            signal_phase[:, 4:] = 0
            signal_phase = phase_pressure(signal_phase, weightZZ, weightIM, weightWL)
            # 压力缩放：用短绿折减、上一相位反折
            for tl_id in ['TL_ZZ', 'TL_IM', 'TL_WL']:
                node_token = tl_id.split('_')[1]  # 'ZZ'/'IM'/'WL'
                last_idx = self.last_phase[tl_id]
                signal_phase = self._scale_pressures(signal_phase, node_token, last_idx, self.yellow_duration, T0)

        # 做一次相位决策（不落地执行，交给后面的“每步控制”去处理黄灯/绿灯）
        if (t0 >= startopt) and ((t0 - startopt) % T0 == 0):
            for tl_id in ['TL_ZZ', 'TL_IM', 'TL_WL']:
                node_token = tl_id.split('_')[1]
                # 取该路口所有相位 [index, pressure, state_string]
                candidates = [(i, float(signal_phase[i, 4]), signal_phase[i, 1])
                                for i in range(signal_phase.shape[0])
                                if signal_phase[i, 0] == node_token]
                if not candidates:
                    continue
                best_idx, _, best_state = max(candidates, key=lambda x: x[1])
                # 若需要换相位：启动黄灯，并记住“黄后切到谁”
                if best_idx != self.last_phase[tl_id]:
                    self.yellow_flag[tl_id] = True
                    self.yellow_counter[tl_id] = 0
                    self.pending_phase[tl_id] = best_idx
                    self.pending_state[tl_id] = best_state
                else:
                    # 不需要换相位，直接维持当前绿灯状态
                    traci.trafficlight.setRedYellowGreenState(tl_id, best_state)

        # 2) 每一步处理黄灯过渡
        if signal:
            for tl_id in ['TL_ZZ', 'TL_IM', 'TL_WL']:
                if self.yellow_flag[tl_id]:
                    # 上一相位的黄灯显示
                    last_idx = self.last_phase[tl_id]
                    yellow_state = self._get_yellow_state(last_idx, signal_phase)
                    if yellow_state:
                        traci.trafficlight.setRedYellowGreenState(tl_id, yellow_state)
                    self.yellow_counter[tl_id] += 1
                    # 黄灯结束：切到 pending 绿灯，并更新 last_phase
                    if self.yellow_counter[tl_id] >= self.yellow_duration:
                        self.yellow_flag[tl_id] = False
                        self.yellow_counter[tl_id] = 0
                        if self.pending_state[tl_id] is not None:
                            traci.trafficlight.setRedYellowGreenState(tl_id, self.pending_state[tl_id])
                        if self.pending_phase[tl_id] >= 0:
                            self.last_phase[tl_id] = self.pending_phase[tl_id]
                        # 清理 pending
                        self.pending_phase[tl_id] = -1
                        self.pending_state[tl_id] = None

        return signal_phase
    
    # def phase_pressure_update(self, T0, signal_phase, weightZZ, weightIM, weightWL):
    #     """到 T0 边界才进入：重算压力、缩放、为每个路口选 best 相位；
    #        若需换相位则置黄灯标志并写 pending_*；否则仅更新 pending_* 以维持当前绿。"""

    #     # 1) 重算压力
    #     signal_phase[:, 4:] = 0
    #     signal_phase = phase_pressure(signal_phase, weightZZ, weightIM, weightWL)

    #     # 2) 压力缩放（你的规则）
    #     for tl_id in self.tl_ids:
    #         node_token = tl_id.split('_')[1]
    #         last_idx = self.last_phase[tl_id]
    #         signal_phase = self._scale_pressures(signal_phase, node_token, last_idx, self.yellow_duration, T0)

    #     # 3) 相位决策（只记录，不执行）
    #     for tl_id in self.tl_ids:
    #         node_token = tl_id.split('_')[1]
    #         # [row_idx, pressure, state_string]
    #         candidates = [(i, float(signal_phase[i, 4]), signal_phase[i, 1])
    #                       for i in range(signal_phase.shape[0])
    #                       if signal_phase[i, 0] == node_token]
    #         if not candidates:
    #             # 清空 pending，避免误用旧状态
    #             self.pending_phase[tl_id] = -1
    #             self.pending_state[tl_id] = None
    #             continue

    #         best_idx, _, best_state = max(candidates, key=lambda x: x[1])

    #         # 首次还没有 last_phase：直接“定绿”，不走黄灯
    #         if self.last_phase[tl_id] == -1:
    #             self.pending_phase[tl_id] = best_idx
    #             self.pending_state[tl_id] = best_state
    #             self.yellow_flag[tl_id] = False
    #             self.yellow_counter[tl_id] = 0
    #             continue

    #         if best_idx != self.last_phase[tl_id]:
    #             # 需要换相位：开启黄灯并记录黄后要切到哪里
    #             self.yellow_flag[tl_id] = True
    #             self.yellow_counter[tl_id] = 0
    #             self.pending_phase[tl_id] = best_idx
    #             self.pending_state[tl_id] = best_state
    #         else:
    #             # 不换相：把“要维持的绿灯串”放到 pending_state，方便后续维持
    #             self.pending_phase[tl_id] = best_idx
    #             self.pending_state[tl_id] = best_state

    #     return signal_phase

    # def signal_update(self, signal_phase, t0, startopt, T0):
    #     # if (t0 < startopt) or ((t0 - startopt) % T0 != 0):
    #     #     return
    #     """每个仿真步都调用：黄灯显示与计数；黄灯结束后切绿并更新 last_phase；
    #        非黄灯期间就维持 pending_state（当前绿）。"""
    #     for tl_id in self.tl_ids:
    #         if self.yellow_flag[tl_id]:
    #             last_idx = self.last_phase[tl_id]
    #             # 若 last_idx 无效（-1），无法计算黄灯，直接切到 pending 绿
    #             if last_idx >= 0:
    #                 y = self._get_yellow_state(last_idx, signal_phase)
    #                 if y:
    #                     traci.trafficlight.setRedYellowGreenState(tl_id, y)
    #                 self.yellow_counter[tl_id] += 1
    #                 if self.yellow_counter[tl_id] >= self.yellow_duration:
    #                     self.yellow_flag[tl_id] = False
    #                     self.yellow_counter[tl_id] = 0
    #                     if self.pending_state[tl_id] is not None:
    #                         traci.trafficlight.setRedYellowGreenState(tl_id, self.pending_state[tl_id])
    #                     if self.pending_phase[tl_id] >= 0:
    #                         self.last_phase[tl_id] = self.pending_phase[tl_id]
    #             else:
    #                 # 兜底：没有上一相位，直接上 pending 绿
    #                 if self.pending_state[tl_id] is not None:
    #                     traci.trafficlight.setRedYellowGreenState(tl_id, self.pending_state[tl_id])
    #                 if self.pending_phase[tl_id] >= 0:
    #                     self.last_phase[tl_id] = self.pending_phase[tl_id]
    #                 self.yellow_flag[tl_id] = False
    #                 self.yellow_counter[tl_id] = 0
    #         else:
    #             # 非黄灯过渡：维持当前绿
    #             if self.pending_state[tl_id] is not None:
    #                 traci.trafficlight.setRedYellowGreenState(tl_id, self.pending_state[tl_id])