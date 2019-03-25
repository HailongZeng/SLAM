import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
import copy
from numpy.random import randn, uniform
import cv2 as cv
from PIL import Image

def mapCorrelation(im, x_im, y_im, vp, xs, ys):
    '''
    INPUT
    im              the map
    x_im,y_im       physical x,y positions of the grid map cells
    vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)
    xs,ys           physical x,y,positions you want to evaluate "correlation"

    OUTPUT
    c               sum of the cell values of all the positions hit by range sensor
    '''
    nx = im.shape[0]
    ny = im.shape[1]
    xmin = x_im[0]
    xmax = x_im[-1]
    xresolution = (xmax - xmin) / (nx - 1)
    ymin = y_im[0]
    ymax = y_im[-1]
    yresolution = (ymax - ymin) / (ny - 1)
    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((nxs, nys))
    for jy in range(0, nys):
        y1 = vp[1, :] + ys[jy]  # 1 x 1076
        iy = np.int16(np.round((y1 - ymin) / yresolution))
        for jx in range(0, nxs):
            x1 = vp[0, :] + xs[jx]  # 1 x 1076
            ix = np.int16(np.round((x1 - xmin) / xresolution))
            valid = np.logical_and(np.logical_and((iy >= 0), (iy < ny)), \
                                   np.logical_and((ix >= 0), (ix < nx)))
            cpr[jx, jy] = np.sum(im[ix[valid], iy[valid]])
    return cpr


def bresenham2D(sx, sy, ex, ey):
    '''
    Bresenham's ray tracing algorithm in 2D.
    Inputs:
        (sx, sy)	start point of ray
        (ex, ey)	end point of ray
    '''
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex - sx)
    dy = abs(ey - sy)
    steep = abs(dy) > abs(dx)
    if steep:
        dx, dy = dy, dx  # swap

    if dy == 0:
        q = np.zeros((dx + 1, 1))
    else:
        q = np.append(0, np.greater_equal(
            np.diff(np.mod(np.arange(np.floor(dx / 2), -dy * dx + np.floor(dx / 2) - 1, -dy), dx)), 0))
    if steep:
        if sy <= ey:
            y = np.arange(sy, ey + 1)
        else:
            y = np.arange(sy, ey - 1, -1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx, ex + 1)
        else:
            x = np.arange(sx, ex - 1, -1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)
    return np.vstack((x, y))

def create_particles(N):
    '''
    :param N: # of particles
    :return: particles   [x, y, theta]
    '''
    particles = np.empty((3, N))
    particles[0, :] = 0  # x
    particles[1, :] = 0  # y
    particles[2, :] = 0  # theta
    return particles

def parameter_extract(encoder_counts, encoder_stamps, imu_stamps, imu_angular_velocity):
    '''
    :param encoder_counts:
    :param encoder_stamps:
    :param imu_stamps:
    :param imu_angular_velocity:
    :return: w_new --- angular velocity corresponding to v     v --- linear velocity  tau --- time difference
    '''
    w = imu_angular_velocity[2, :]  # yaw angular velocity
    FR = encoder_counts[0, :]
    FL = encoder_counts[1, :]
    RR = encoder_counts[2, :]
    RL = encoder_counts[3, :]
    d_l = (FR + RR) / 2 * 0.0022
    d_r = (FL + RL) / 2 * 0.0022
    d = (d_r + d_l) / 2  # 1*4956
    tau = encoder_stamps[1:] - encoder_stamps[0:-1]  # 1*4955
    v = d[0:-1] / tau  # 1*4955
    w_new = np.zeros((1, d.shape[0]-1))[0]  # corresponding to v
    if len(np.where(encoder_stamps <= np.min(imu_stamps))[0]) > 0:
        first_index = np.max(np.where(encoder_stamps <= np.min(imu_stamps))) + 1
    else:
        first_index = 0
    if len(np.where(encoder_stamps <= np.min(imu_stamps))[0]) > 0:
        last_index = np.min(np.where(encoder_stamps >= np.max(imu_stamps)))
    else:
        last_index = encoder_stamps.shape[0] - 1

    for i in range(first_index, last_index):
        index = np.where(
            np.logical_and(imu_stamps >= encoder_stamps[i], imu_stamps <= encoder_stamps[i + 1]))  # return tuple
        if len(list(index)[0]) == 0:
            w_new[i] = w_new[i - 1]
        else:
            w_new[i] = np.sum(w[index]) / len(list(index)[0])  # average the angular velocity between two time_stamps

    return tau, v, w_new

#### prediction
def prediction(mu_t, tau_t, v_t, w_t):
    '''
    :param mu_t: the position of robot at time t, 3*N, [x, y, theta]
    :param tau_t: time difference at time t
    :param v_t: linear velocity at time t
    :param w_t: angular velocity at time t
    :return: mu_pre  predicted position
    '''
    N = mu_t.shape[1]
    mu_pre = np.zeros((mu_t.shape[0], mu_t.shape[1]))

    for j in range(N):
        x_t = mu_t[0, j]
        y_t = mu_t[1, j]
        theta_t = mu_t[2, j]

        v_t_noise = copy.deepcopy(v_t) + 0.1 * randn(1)
        w_t_noise = copy.deepcopy(w_t) + 0.01 * randn(1)
        x_pre = x_t + tau_t * (v_t_noise * np.sinc(w_t_noise * tau_t / 2) * np.cos(theta_t + w_t_noise * tau_t / 2))
        y_pre = y_t + tau_t * (v_t_noise * np.sinc(w_t_noise * tau_t / 2) * np.sin(theta_t + w_t_noise * tau_t / 2))
        theta_pre = theta_t + w_t_noise * tau_t

        mu_pre[0, j] = x_pre
        mu_pre[1, j] = y_pre
        mu_pre[2, j] = theta_pre

    return mu_pre

##### mapping
def mapping(m_tm1, z_t, x_t):
    '''
    :param m_tm1: log_odds map m_(t-1)
    :param z_t: laser scan z_t
    :param x_t: the position information of the robot at time t, [x, y, orientation]
    :return: log_odds map m_t
    '''
    # position of robot
    x = x_t[0]
    y = x_t[1]
    theta = x_t[2]

    # log-odd parameters
    lo_occu = np.log(4)
    lo_free = np.log(1 / 4)

    angles = np.arange(-135, 135.25, 0.25) * np.pi / 180.0

    # take valid indices
    indValid = np.logical_and((z_t > 0.1), (z_t < 30))
    z_t = z_t[indValid]
    angles = angles[indValid]
    N = z_t.shape[0]

    res = 0.05;
    min = -30
    max = 30
    size = int(np.ceil((max - min) / res + 1))  #cells

    # # laser positon in the world frame
    xs = x + 0.13233
    ys = y

    # convert from meters to cells
    xis = np.ceil((xs - min) / res).astype(np.int16) - 1
    yis = np.ceil((ys - min)/ res).astype(np.int16) - 1

    # xy position in the world frame, the end point of the ray
    xe = z_t * np.cos(angles+theta) + x + 0.13233
    ye = z_t * np.sin(angles+theta) + y

    # convert from meters to cells, the end point of laser ray, occupied point
    xie = np.ceil((xe - min) / res).astype(np.int16) - 1
    yie = np.ceil((ye - min) / res).astype(np.int16) - 1

    m_t = copy.deepcopy(m_tm1)
    for i in range(N):
        free = bresenham2D(xis, yis, xie[i], yie[i])
        freex = free[0][:-1].astype(np.int16)
        freey = free[1][:-1].astype(np.int16)
        index_fx = np.logical_and((freex > 1), (freex < size))
        index_fy = np.logical_and((freey > 1), (freey < size))
        m_t[freex[index_fx], freey[index_fy]] += lo_free
       
    # update the map value when occupied
    indGood = np.logical_and(np.logical_and(np.logical_and((xie > 1), (yie > 1)), (xie < size)),
                             (yie < size))
    m_t[xie[indGood[0]], yie[indGood[0]]] += lo_occu

    # set constraint
    m_min = -10
    m_max = 5
    m_t[(m_t < m_min)] = m_min
    m_t[(m_t > m_max)] = m_max

    return m_t

##### update
def update(mu, alpha, z_tp1, m_t):
    '''
    :param mu: current particle position  [x, y, theta]  at time t  3*N， N means # of particles
    :param alpha: current weights at time t, 1*N, N means # of particles
    :param z_tp1: laser scan at time t+1     1081*1
    :param m_t: current map at time t
    :return: alpha_new---updated particle weights    mu_best --- position of the best particle
    '''
    Corr = np.zeros((1, mu.shape[1]))[0]
    angles = np.arange(-135, 135.25, 0.25) * np.pi / 180.0
    mu_best = np.zeros((3, 1))

    for i in range(mu.shape[1]):
        x = copy.deepcopy(mu[0, i])
        y = copy.deepcopy(mu[1, i])
        theta = copy.deepcopy(mu[2, i])

        # take valid indices
        indValid = np.logical_and((z_tp1 > 0.1), (z_tp1 < 30))
        z_tp1 = z_tp1[indValid]
        angles = angles[indValid]
        N = z_tp1.shape[0]

        # xy position in the world frame, the end point of the ray
        xs = z_tp1 * np.cos(angles + theta) + x + 0.13233
        ys = z_tp1 * np.sin(angles + theta) + y

        # convert position in the map frame here
        Y = np.stack((xs,ys))

        x_im = np.arange(-30, 30+0.05, 0.05)  # x-positions of each pixel of the map
        y_im = np.arange(-30, 30+0.05, 0.05)  # y-positions of each pixel of the map

        x_range = np.arange(-0.2, 0.2 + 0.05, 0.05)
        y_range = np.arange(-0.2, 0.2 + 0.05, 0.05)

        m_t_new = np.zeros((m_t.shape[0], m_t.shape[1]))
        # convert the map to 0-1
        index_f = (m_t < 0)
        index_o = (m_t > 0)
        m_t_new[index_f] = 0
        m_t_new[index_o] = 1

        c = mapCorrelation(m_t_new, x_im, y_im, Y, x_range, y_range)  # call function mapCorrelation
        corr = np.max(c)   # find tha maximal one to be used in updating particle weights
        Corr[i] = corr

    mu_best = copy.deepcopy(mu[:, np.argmax(Corr)])
    alpha_new = alpha*np.exp(Corr-np.max(Corr))
    alpha_new /= np.sum(alpha_new)  #normalization

    return mu_best, alpha_new

def Neff(alpha):
    '''
    :param alpha: particle weights
    :return: effective number of particles
    '''
    return 1./np.sum(np.square(alpha))

def resampling(mu, alpha):
    '''
    :param mu: current particle position  [x, y, theta]  at time t  3*N， N means # of particles
    :param alpha: current weights at time t, 1*N, N means # of particles
    :return: mu_new   update particle position    alpha_new   update particle position
    '''
    N = mu.shape[1]
    j = 0
    c = alpha[j]
    mu_new = np.zeros((3, N))
    alpha_new = np.zeros(N)
    for i in range(N):
        u = uniform(0, 1/N, 1)
        beta = u + i/N
        while beta > c:
            j = j + 1
            c = c + alpha[j]
        mu_new[:, i] = copy.deepcopy(mu[:, j])
        alpha_new[i] = 1/N
    return mu_new, alpha_new

##### texture_mapping
def calibration(x11, x12, x13, x22, x23, x33):
    '''
    :param x11: The entry value at row 1, column 1
    :param x12: The entry value at row 1, column 2
    :param x13: The entry value at row 1, column 3
    :param x22: The entry value at row 2, column 2
    :param x23: The entry value at row 2, column 3
    :param x33: The entry value at row 3, column 3
    :return: calibration matrix
    '''
    K = np.zeros((3, 3))
    K[0, 0] = x11
    K[0, 1] = x12
    K[0, 2] = x13
    K[1, 1] = x22
    K[1, 2] = x23
    K[2, 2] = x33

    return K

def Trans_b2w(mu_t):
    '''
    :param mu_t: robot position(current best particle)  3*1 [x_t, y_t, theta_t]
    :return: T_w2b---Transformation from world frame to body frame
    '''
    x_t = mu_t[0]
    y_t = mu_t[1]
    theta_t = mu_t[2]
    R_b2w = np.array([[np.cos(theta_t), -np.sin(theta_t), 0], [np.sin(theta_t), np.cos(theta_t), 0], [0, 0, 1]])
    # R_w2b = np.linalg.inv(R_b2w)
    h = np.array([[x_t], [y_t], [0]])
    v = np.array([0, 0, 0, 1])
    T_b2w = np.vstack((np.hstack((R_b2w, h)), v))

    return T_b2w

def Trans_ir2b(p, roll, pitch, yaw):
    '''
    :param p: The position camera in body frame
    :param φ: roll, A rotation φ about the original x-axis
    :param θ: pitch, A rotation θ about the intermediate y-axis
    :param ψ: yaw, A rotation ψ about the transformed z-axis
    :return: T_b2ir---from body frame to infrared camera
    '''
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]);
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R = np.dot(np.dot(R_z, R_y), R_x)
    v = np.array([0, 0, 0, 1])
    T_ir2b = np.vstack((np.hstack((R, p)), v))

    return T_ir2b

def texture_mapping(I_t, D_t, mu_t, texture_m_t, h_threshold):
    '''
    :param I_t: Current RGB image
    :param D_t: Current depth image
    :param mu_t: Robot pose(current best particle)  3*1
    :param texture_m_t: Current texture map
    :return: texture_m_tp1  Updated texture map
    '''
    min = -30
    res = 0.05
    ### Projection and Intrinsics
    x11 = 585.05108211
    x12 = 0
    x13 = 242.94140713
    x22 = 585.05108211
    x23 = 315.83800193
    x33 = 1
    K = calibration(x11, x12, x13, x22, x23, x33)  # calibrate matrix
    Pi_0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # canonical projection

    ### Entrinsics
    # transformation from camera frame to optical frame
    T_o2c = np.linalg.inv(np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]))

    # transformation from body frame to rgb camera frame
    p = np.array([[0.18], [0.005], [0.36]])
    roll = 0
    pitch = 0.36
    yaw = 0.021
    T_ir2b = Trans_ir2b(p, roll, pitch, yaw)

    # tranformation from world frame to body frame
    T_b2w = Trans_b2w(mu_t)

    sizei = D_t.shape[0]
    sizej = D_t.shape[1]
    i = np.arange(sizei).reshape(sizei, -1)  # i  480*1
    j = np.arange(sizej).reshape(-1, sizej)  # j  1*640
    ii = np.repeat(i, sizej, axis=1)    # 480*640
    jj = np.repeat(j, sizei, axis=0)    # 480*640
    iii = ii.reshape(-1)      # 1*307200
    jjj = jj.reshape(-1)      # 1*30720
    dd = -0.00304 * D_t + 3.31              # dd, 480*640
    depth = 1.03 / dd                       # depth, 480*640

    UV = np.stack((jjj, iii))  # 2*307200
    II = np.ones(sizei*sizej)
    UVI = np.row_stack((UV, II))      # 3*307200

    depth_re = depth.reshape(-1)  # 1*307200
    Po = np.dot(np.linalg.inv(np.dot(K, Pi_0)), UVI) * depth_re  # 3*307200, optical frame

    PoI = np.row_stack((Po, II))   # 4*307200
    Pw = np.dot(np.dot(np.dot(T_b2w, T_ir2b), T_o2c), PoI)  # 4*307200, world frame
    Xw = Pw[0, :].reshape(sizei, sizej)   # 480*640
    Yw = Pw[1, :].reshape(sizei, sizej)   # 480*640
    Zw = Pw[2, :].reshape(sizei, sizej)   # 480*640

    # h_threshold = -0.05
    index_grou = np.array(np.where(Zw < h_threshold))   # 2*N
    texture_m_tp1 = copy.deepcopy(texture_m_t)   # 1201*1201*3

    rgbi = np.ceil((index_grou[0] * 526.37 + dd[index_grou[0],index_grou[1]] * (-4.5 * 1750.46) + 19276.0) / 585.051).astype(np.int16) - 1  # N
    rgbj = np.ceil((index_grou[1] * 526.37 + 16662.0) / 585.051).astype(np.int16) - 1  # 480*640

    index_x = np.ceil((Xw[index_grou[0], index_grou[1]] - min) / res).astype(np.int16) - 1 # N*1
    index_y = np.ceil((Yw[index_grou[0], index_grou[1]] - min) / res).astype(np.int16) - 1 # N*1

    indGood = np.logical_and(np.logical_and(np.logical_and(rgbi < 480, rgbj < 640), index_x < 1201), index_y < 1201)
    texture_m_tp1[index_x[indGood], index_y[indGood], :] = I_t[rgbi[indGood], rgbj[indGood], :]

    return texture_m_tp1

if __name__ == '__main__':
    start = time.time()
    dataset = 21
    with np.load("Encoders%d.npz" % dataset) as data:
        encoder_counts = data["counts"]  # 4 x n encoder counts
        encoder_stamps = data["time_stamps"]  # encoder time stamps

    with np.load("Hokuyo%d.npz" % dataset) as data:
        lidar_angle_min = data["angle_min"]  # start angle of the scan [rad]
        lidar_angle_max = data["angle_max"]  # end angle of the scan [rad]
        lidar_angle_increment = data["angle_increment"]  # angular distance between measurements [rad]
        lidar_range_min = data["range_min"]  # minimum range value [m]
        lidar_range_max = data["range_max"]  # maximum range value [m]
        lidar_ranges = data["ranges"]  # range data [m] (Note: values < range_min or > range_max should be discarded)
        lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans

    with np.load("Imu%d.npz" % dataset) as data:
        imu_angular_velocity = data["angular_velocity"]  # angular velocity in rad/sec
        imu_linear_acceleration = data["linear_acceleration"]  # Accelerations in gs (gravity acceleration scaling)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

    with np.load("Kinect%d.npz" % dataset) as data:
        disp_stamps = data["disparity_time_stamps"]  # acquisition times of the disparity images
        rgb_stamps = data["rgb_time_stamps"]  # acquisition times of the rgb images

    tau, v, w_new = parameter_extract(encoder_counts, encoder_stamps, imu_stamps, imu_angular_velocity)

    N = 10  # number of particles

    Nthreshold = 0.8 * N
    mu_i = create_particles(N)  # particles
    alpha_i = 1 / N * np.ones(N)  # initial weights

    m_im1 = np.zeros((1201, 1201), dtype=np.float32)  # initial mapping
    z_i = lidar_ranges[:, 0]
    m_i = mapping(m_im1, z_i, mu_i[:, np.argmax(alpha_i)])  # first scan

    robot_pose = np.zeros((3, encoder_stamps.shape[0]))
    count = np.min((encoder_stamps.shape[0], lidar_stamps.shape[0]))
    for i in range(count - 1):  # range(encoder_stamps.shape[0] - 1):
        # mapping
        z_i = lidar_ranges[:, i]
        m_i = mapping(m_im1, z_i, robot_pose[:, i])  # mu_i[:, np.argmax(alpha_i)])

        # prediction
        mu_ip1 = np.zeros((3, N))  # 3*N
        mu_ip1 = prediction(mu_i, tau[i], v[i], w_new[i])

        # update
        if i + 1 < lidar_stamps.shape[0]:
            z_ip1 = lidar_ranges[:, i + 1]
            robot_pose[:, i + 1], alpha_update = update(mu_ip1, alpha_i, z_ip1,
                                                        m_i)  # update alpha, and get the best position
            # resample
            if Neff(alpha_update) <= Nthreshold:
                [mu_i, alpha_i] = resampling(mu_ip1, alpha_update)  # resample, update mu and alpha
            else:
                mu_i = copy.deepcopy(mu_ip1)
        else:
            robot_pose[:, i + 1] = copy.deepcopy(mu_ip1[:, np.argmax(alpha_update)])
            z_ip1 = copy.deepcopy(z_i)
        m_im1 = copy.deepcopy(m_i)

    m_i = mapping(m_im1, z_ip1, robot_pose[:, -1])
    m_i_pdf = 1 - 1 / (1 + np.exp(m_i))

    ##### texture_mapping
    texture_m_t = np.zeros((1201, 1201, 3))

    # robot_pose = np.loadtxt('robot_pose_20.txt')
    if rgb_stamps.shape[0] < disp_stamps.shape[0]:
        count = rgb_stamps.shape[0]
    else:
        count = disp_stamps.shape[0]

    texture_m_tp1 = copy.deepcopy(texture_m_t)
    h_threshold = 0.01
    for i in range(count):
        j = np.argmin(abs(encoder_stamps - disp_stamps[i]))
        mu_t = copy.deepcopy(robot_pose[:, j])
        img = Image.open('/Users/zenghailong/Library/Mobile Documents/com~apple~CloudDocs/Desktop/2019Winter/ECE276A/ECE276A_HW2/data/dataRGBD/Disparity{}/disparity{}_{}.png'.format(dataset, dataset, i+1))
        D_t = np.array(img.getdata(),  np.uint16).reshape(img.size[1], img.size[0])
        I_t = plt.imread('/Users/zenghailong/Library/Mobile Documents/com~apple~CloudDocs/Desktop/2019Winter/ECE276A/ECE276A_HW2/data/dataRGBD/RGB{}/rgb{}_{}.png'. format(dataset, dataset, i+1))
        texture_m_tp1 = texture_mapping(I_t, D_t, mu_t, texture_m_t, h_threshold)  #update texture_map
        texture_m_t = copy.deepcopy(texture_m_tp1)

    ##### Show figures
    m_i_pdf[(m_i_pdf < 0.5)] = 1
    m_i_pdf[(m_i_pdf >= 0.5)] = 0
    fig = plt.figure()
    plt.imshow(m_i_pdf)
    plt.imshow(texture_m_t)
    pose_x = robot_pose[0, :]
    pose_y = robot_pose[1, :]
    index_x = np.ceil((pose_x - (-30)) / 0.05).astype(np.int16) - 1
    index_y = np.ceil((pose_y - (-30)) / 0.05).astype(np.int16) - 1
    plt.plot(index_y, index_x)  # note, should be y---x since the horizontal axis is y axis
    plt.show()
    end = time.time()

    print(end - start)





