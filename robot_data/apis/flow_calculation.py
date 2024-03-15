import numpy as np

def flow_calculate_global(flow):
    u_sum = np.zeros(63)
    v_sum = np.zeros(63)
    u_addon = list(u_sum)
    v_addon = list(v_sum)
    x_iniref, y_iniref = [], []
    x2, y2, u, v, x1_paired, y1_paired, x2_paired, y2_paired  = [], [], [], [], [], [], [], []
    x1_return, y1_return, x2_return, y2_return, u_return, v_return = [],[],[],[],[],[]

    Ox, Oy, Cx, Cy, Occupied = flow
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            x2.append(Cx[i][j])
            y2.append(Cy[i][j])
            x_iniref.append(Ox[i][j])
            y_iniref.append(Oy[i][j])
    # for i in range(len(keypoints2)): 
    #     x2.append(keypoints2[i].pt[0]/self.scale)
    #     y2.append(keypoints2[i].pt[1]/self.scale)

    x2 = np.array(x2) 
    y2 = np.array(y2)
    x_initial = list(x_iniref)
    y_initial = list(y_iniref)
    u_ref = list(u_addon)
    v_ref = list(v_addon)
    

    for i in range(x2.shape[0]):
        distance = list(((np.array(x_initial) - x2[i])**2 + (np.array(y_initial) - y2[i])**2))
        min_index = distance.index(min(distance))  
        u_temp = x2[i] - x_initial[min_index] 
        v_temp = y2[i] - y_initial[min_index] 
        shift_length = np.sqrt(u_temp**2+v_temp**2)
        # print 'length',shift_length

        # print xy2.shape,min_index,len(distance)
        if shift_length < 7:
            x1_paired.append(x_initial[min_index]-u_ref[min_index])
            y1_paired.append(y_initial[min_index]-v_ref[min_index])
            x2_paired.append(x2[i])
            y2_paired.append(y2[i])
            u.append(u_temp + u_ref[min_index])
            v.append(v_temp + v_ref[min_index])
            # sign = self.ROI[y2[i].astype(np.uint16),x2[i].astype(np.uint16)]
            # x1_return.append((x_initial[min_index]-u_ref[min_index])*sign)
            # y1_return.append((y_initial[min_index]-v_ref[min_index])*sign)
            # x2_return.append((x2[i])*sign)
            # y2_return.append((y2[i])*sign)
            # u_return.append((u_temp + u_ref[min_index])*sign)
            # v_return.append((v_temp + v_ref[min_index])*sign)
            del x_initial[min_index], y_initial[min_index], u_ref[min_index], v_ref[min_index]   

        
    # print('len:',len(x_iniref), len(x2_paired))
    x_iniref = list(x2_paired) 
    y_iniref = list(y2_paired)
    u_addon = list(u)
    v_addon = list(v)
    refresh = False 
    # print(np.array(y2_paired).astype(np.uint16),np.array(x2_paired).astype(np.uint16),np.array(range(len(x2_paired))))
    # inbound_check = img[np.array(y2_paired).astype(np.uint16),np.array(x2_paired).astype(np.uint16)]*np.array(range(len(x2_paired)))

    # final_list = list(set(inbound_check)- set([0]))
    # final_list = list(set(x2_paired)- set([0]))
    # x1_return = np.array(x1_paired)[final_list]
    # y1_return = np.array(y1_paired)[final_list]
    # x2_return = np.array(x2_paired)[final_list]
    # y2_return = np.array(y2_paired)[final_list]
    # u_return = np.array(u)[final_list]
    # v_return = np.array(v)[final_list]

    x1_return = np.array(x1_paired)
    y1_return = np.array(y1_paired)
    x2_return = np.array(x2_paired)
    y2_return = np.array(y2_paired)
    u_return = np.array(u)
    v_return = np.array(v)

    return x1_return, y1_return, x2_return, y2_return, u_return, v_return

def estimate_uv(tran_matrix, x1, y1, u_sum, v_sum, x2, y2):
    theta = np.arcsin(tran_matrix[1,0])
    x1_select = np.array(x1)
    y1_select = np.array(y1)
    u_select = u_sum
    v_select = v_sum

    u_mean = np.mean(u_select)
    v_mean = np.mean(v_select)
    x_mean = np.mean(x1_select)
    y_mean = np.mean(y1_select)

    u_estimate = u_mean + theta*(y_mean - np.array(y2))
    v_estimate = v_mean + theta*(np.array(x2)-x_mean)

    return u_estimate, v_estimate