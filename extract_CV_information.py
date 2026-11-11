import traci
import math
import numpy as np
from pressure_controller import get_phase_pressures
# 重点检查状态提取
def extract_EB_information(bus_info, link_start_edge, edge_length_speed, bus_stop_position, corridor_edges, 
                           signal_phase, t0, T0, startopt, v_prev):
    all_states = []
    for row in bus_info:
        # 当前道路
        vehicleID = row[0]
        depart_index = row[11]
        current_edge = row[1]
        current_lane  = traci.vehicle.getLaneID(vehicleID)
        current_position = row[2]
        current_speed = traci.vehicle.getSpeed(vehicleID)
        # current_acceleration = traci.vehicle.getAcceleration(vehicleID)
        current_acceleration = current_speed - v_prev

        # 距离下一个交叉口
        # 当前信号灯信息
        next_tls = traci.vehicle.getNextTLS(vehicleID)
        if next_tls:
            tls_id = next_tls[0][0]
            distance_to_intersection = next_tls[0][2]/500 # 归一化
            state_ch = next_tls[0][3]
            color = state_ch.lower()
            light_state = 0 if color == 'g' else 1 if color == 'y' else 2  # 当前车道红绿状态：0=G, 1=y, 2=r；直接一个函数搞定
            # 当前相位编号（保留）：这个没有必要，只需要知道目前的红绿灯情况即可；另外，相位切换也不是按照顺序来的；之前的考虑似乎是为了与压力对应
            # current_phase = traci.trafficlight.getPhase(tls_id) 
            signal_index = next_tls[0][1]
            node_key = tls_id.split('_')[1]
            rows = [(i, row) for i, row in enumerate(signal_phase) if row[0] == node_key]
            pairs = []                              # [(row_idx, pressure, row_obj), ...]
            current_phase_idx = None
            current_phase_p   = -float("inf")
            for i, row in rows:
                state_str = row[1]          # 每个相位的 RYG 字符串
                p = float(row[4])           # 相位压力
                pairs.append((i, p))
                if signal_index < len(state_str) and state_str[signal_index] in ('G', 'g'):
                    current_phase_idx  = i
                    current_phase_p    = p
            if current_phase_idx is not None:
                others = [p for idx, p in pairs if idx != current_phase_idx]
            else:
                others = [p for _, p in pairs] 
            max_other = max(others) if others else 0.0
            gap = current_phase_p - max_other
            pressure_info = [current_phase_p, max_other, gap]
            # 当前相位已持续时间-->换成目前决策阶段剩余时间
            # current_phase_duration = traci.trafficlight.getSpentDuration(tls_id) # TLS以及持续时间能够正确获得
            deci_rem = (startopt - t0 if t0 < startopt else (T0 - ((t0 - startopt) % T0)) % T0)/10  # 归一化
            # if t0 < startopt:
            #     deci_rem = startopt - t0
            # else:
            #     deci_rem = (T0 - ((t0 - startopt) % T0)) % T0  # give the remaining time until the next decision step
            # 获取相位压力信息
            # phase_pressures = get_phase_pressures(tls_id, signal_phase)
        else:
            distance_to_intersection = -1
            # current_phase = -1
            # current_phase_duration = -1
            deci_rem = -1
            light_state = 0  # 缺省为绿灯；无交叉口时
            pressure_info = [0.0] * 3
        pressure_info_norm = [x / 10.0 for x in pressure_info]
        # 前车信息
        leader_info = traci.vehicle.getLeader(vehicleID, dist=50.0)
        if leader_info:
            distance_to_lead_vehicle = leader_info[1]/10 # 归一化
            lead_vehicle_speed = traci.vehicle.getSpeed(leader_info[0])/10 # 归一化
            lead_vehicle_acceleration = traci.vehicle.getAcceleration(leader_info[0])
        else:
            distance_to_lead_vehicle = -1
            lead_vehicle_speed = -1  # -1表示没有
            lead_vehicle_acceleration = 0

    stops = traci.vehicle.getStops(vehicleID)
    distance_to_next_bus_stop = -1.0  # -1 表示后续没有公交站
    bus_stop_isdwell = 0
    duration = 15.0

    if stops:
        stop = stops[0]
        stop_lane   = stop.lane
        stop_edge   = stop_lane.rsplit('_', 1)[0]
        stop_endPos = float(stop.endPos)
        duration    = float(stop.duration)

        bus_stop_isdwell = bus_in_dwell(vehicleID)

        EPS = 0.1  # 容忍一点浮点误差 & 位置范围

        if bus_stop_isdwell:
            # 明确在站台内：距离直接记为 0
            distance_to_next_bus_stop = 0.0

        elif (current_lane == stop_lane) and (current_position <= stop_endPos + EPS):
            # 同一条 lane 上，还没到站：用几何差
            distance_to_next_bus_stop = max(0.0, stop_endPos - current_position)

        else:
            # 需要走一段路程才能到下一站，用 driving distance
            raw_dist = traci.vehicle.getDrivingDistance(vehicleID, stop_edge, stop_endPos)
            if raw_dist < 0:
                # -5.3e11 之类的哨兵值都归为“没有可达的下一站”
                distance_to_next_bus_stop = -1.0
            else:
                distance_to_next_bus_stop = float(raw_dist)
 # 未归一化
        ## 在站点附近一定范围内则视为未离站
        # for edge, pos in bus_stop_position:
        #     if current_edge != str(edge):
        #         continue
        #     if (pos - 0.0) <= current_position <= (pos + 0.0):
        #         distance_to_next_bus_stop = 0.0
        #         break
    if distance_to_next_bus_stop > 0:
        distance_to_next_bus_stop /= 500
    state = [
        distance_to_intersection,
        current_speed/10,
        current_acceleration/2,
        light_state,       # 红绿灯状态, 0,1,2代表绿，黄红
        deci_rem,       # 当前决策周期剩余时间
        *pressure_info_norm,
        distance_to_lead_vehicle,
        lead_vehicle_speed,
        lead_vehicle_acceleration/2,
        distance_to_next_bus_stop,
        bus_stop_isdwell, # 公交是否处于停站状态：0是未停站，1是停站，默认未停站
        duration/10,
        depart_index # 只是包含了是否被算入压力；并未：包含了是否在站信息；是否需要包含停站时间信息呢？我觉得不需要
    ]  # 状态可能需要进行归一化；视情况而定
    # 状态需要补充，比如当前是否在站，在站多久


    all_states.append(state)

    return np.array(all_states, dtype=np.float32)

# for the purpose of calculating pressure; get the information of all the CVs in the network
# 重点检查debug部分
def extract_CV_information(CV_ids, link_start_edge, edge_length_speed, bus_stop_position, vehicle_edges, time_entering_edge, t0, T0, considerdwell):
    current_edge_position = []
    for vehicleID in CV_ids:
        current_edge = traci.vehicle.getRoadID(vehicleID)
        
        # Identify the logical link start used by the PTT/LTT pressure term.
        # If that mapped entry has not been observed for this vehicle, fall
        # back to the current SUMO edge instead of forcing LTT to remain zero.
        mapped_start_edge = link_start_edge.get(str(current_edge), str(current_edge))
        if (vehicleID, mapped_start_edge) in time_entering_edge:
            start_edge = mapped_start_edge
        else:
            start_edge = str(current_edge)
        #extract necessary information, type, position, occupancy, link travel time, route
        vtype = traci.vehicle.getTypeID(vehicleID)
        current_position = traci.vehicle.getLanePosition(vehicleID)
        current_occupancy = traci.vehicle.getPersonNumber(vehicleID)
        if current_occupancy<20 and vtype in ['tram_tram', 'bus_bus']:
            current_occupancy = 20
        if vtype == 'veh_CV':
            current_occupancy =2
        #vehicleRoute = traci.vehicle.getRoute(vehicleID)
        if len(traci.vehicle.getNextTLS(vehicleID))>0:
            nextTLS = traci.vehicle.getNextTLS(vehicleID)[0]#tls, movement index, distance to tls, signal state
        else:#the vehicle is on the leaving edge
            nextTLS = [-1,-1,-1,-1]
        #print(vehicleID, nextTLS)
        # Check if the vehicle has entered a new edge, record the time entering the edge
        # has_record = any((vid == vehicleID and e == current_edge) for vid, e, _ in time_entering_edge) #避免漏网之鱼
        if (vehicleID not in vehicle_edges) or (vehicle_edges[vehicleID] != current_edge):# or (not has_record):
            vehicle_edges[vehicleID] = current_edge
            time_entering_edge[(vehicleID, current_edge)] = t0  
        
        # record the current link travel time (LTT) and occupancy of CVs
        #if t0==11341:
        #    print(vehicleID, start_edge)
        LTT = t0 - time_entering_edge.get((vehicleID, start_edge), t0)
        #depart_index = 1 #the vehicle is counted for pressure calculation
        depart_index1 = 1
        if considerdwell == 1:
            if "bus" in vehicleID or "tram" in vehicleID:# this is a transit vehicle
            #two ways to consider bus stop:
            #1. retrieve next stop, identify bus departure by flag and duration
                depart_index1 = 0
                next_stops = traci.vehicle.getStops(vehicleID) # (lane, endPos, stoppingPlaceID(公交站), stopFlags, duration, until)
                last_stops = traci.vehicle.getStops(vehicleID, limit = -1)
                if len(next_stops)>0:
                    next_stop = next_stops[0]  # 判断在距离交叉口最近的车站，以及是否即将离站 # [2]返回的是停靠的bus stop ID; [4]返回的是计划离站时间倒计时
                    duration = float(next_stop.duration)
                    stop_id = next_stop.stoppingPlaceID
                    condition1 = (stop_id in ['3299908041','434507140','434507138','2674521970','bs_0','4621136512','434524954','3303180663','3011895790'])\
                        & (duration <= 1 or duration > 10000)#the condition to count the bus in MP (3 s before planned departure)
                    # condition5 = (next_stop[2] in ['3299908041','434507140','434507138','2674521970','bs_0','4621136512','434524954','3303180663'])\
                    #     & (next_stop[4] <= 1 or next_stop[4] > 10000)#the bus is going to leave the above stops at the moment (the condition to determine GLODTA)
                    # condition4 = next_stop[2] in ['3299908041','434507140','434507138','2674521970','bs_0','3303180663']#the bus stop for bus holding
                else:
                    condition1 = False
                    # condition4 = False
                    # condition5 = False
                if len(last_stops)>0:
                    last_stop = last_stops[0]  # 判断是否已经离开了距离交叉口最近的车站
                    condition2 = (last_stop.stoppingPlaceID in ['3299908041','434507140','434507138','2674521970','bs_0','4621136512','434524954','3303180663','3011895790']) \
                        & (current_edge == last_stop.lane[:-2])#the bus left the stop and is on the same edge (not too far)
                else: 
                    condition2 = False
                #the bus is on the sourced links without any bus stations
                condition3 = current_edge in ['770446896#0','754704157','-770426572#0','770426567','770426566#0','325023862#0','770434440','7045269','-814779818','122385065#0']
                if condition1 or condition2 or condition3:
                    depart_index1 = 1
                # if current_edge in ['770446896#0', '754704157','-770426572#0','770426567','770426566#0','325023862#0','770434440','7045269','-814779818']:
                #    depart_index1 = 1
                '''                    
            #2. retrieve bus position, identify whether the bus position exceeds bus stop. 
                if current_edge in bus_stop_position[:,0]:
                    station_position = float(bus_stop_position[bus_stop_position[:,0]==current_edge,1])
                    if current_position < station_position:
                        depart_index = 0 #the transit vehicle is not counted
                    #print(vehicleID, current_position, station_position, depart_index)  
                #if depart_index != depart_index1:
                #    print(t0, vehicleID, depart_index, depart_index1,next_stop,last_stop)  
                '''
                
        #exclude those CV not for pressure calculation
        if current_edge not in edge_length_speed:
            #print(vehicleID, current_edge)  # ['7046455.81', '814779818', '7045925#0', '-7045691'] and intersections
            depart_index1 = 0
            edge_length = float('inf')
            edge_ETT = float('inf')
        else:
            # edge_length = float(edge_length_speed[edge_length_speed[:,0]==current_edge,1])
            # edge_speed = float(edge_length_speed[edge_length_speed[:,0]==current_edge,2])
            edge_length, edge_speed = edge_length_speed[current_edge]
            edge_ETT = edge_length/edge_speed
            #print(vehicleID, edge_ETT)
        current_edge_position.append([vehicleID, current_edge, current_position, LTT, current_occupancy, \
                                      nextTLS[0], nextTLS[1], nextTLS[2], nextTLS[3], edge_length, edge_ETT, depart_index1])
    return current_edge_position


def bus_in_dwell(vehicleID):
    bus_stop_isdwell = 0
    # next_stops = traci.vehicle.getNextStops(vehicleID)
    bus_stop_isdwell = traci.vehicle.isAtBusStop(vehicleID)
    bus_stop_isdwell = 1 if bus_stop_isdwell else 0
    return bus_stop_isdwell
