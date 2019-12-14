
import numpy as np
import cv2 


def overlay_kimono(src_image, dst_image, src_point, dst_point):
    _t_points, _o_points = [], []
    for t, o in zip(src_point, dst_point):
        if t[-1] == -1 or o[-1] == -1:
            continue
        _t_points.append(t[:2])
        _o_points.append(o[:2])

    scaled_src_point = np.copy(_t_points)
    scaled_dst_point = np.copy(_o_points)
    scaled_src_point[:, 0] *= src_image.shape[1] 
    scaled_src_point[:, 1] *= src_image.shape[0]
    scaled_dst_point[:, 0] *= dst_image.shape[1] 
    scaled_dst_point[:, 1] *= dst_image.shape[0]

    src_max = np.int32(np.max(scaled_src_point, 0))
    dst_max = np.int32(np.max(scaled_dst_point, 0))
    src_min = np.int32(np.min(scaled_src_point, 0)) 
    dst_min = np.int32(np.min(scaled_dst_point, 0))

    ratio = (dst_max - dst_min) / (src_max - src_min)

    kimono_margin_x = 1.4
    kimono_margin_y = 1.01
    dx, dy = dst_min
    nx, ny = (src_max - src_min) * ratio

    w = np.where(src_image[:, :, 3] != 0)
    xmax, xmin = np.max(w[1]), np.min(w[1])
    ymax = np.max(w[0])

    new_size = (int(nx*kimono_margin_x), int(ny+(ymax - src_max[1]) * ratio[1]))

    t_points = np.copy(_t_points)
    o_points = np.copy(_o_points)
    t_points -= np.mean(t_points, 0)
    o_points -= np.mean(o_points, 0)

    U, S, Vt = np.linalg.svd(np.dot(t_points.T, o_points))
    R = (U * Vt).T
    R_ = [[R[0, 0], R[0, 1], 0],
          [R[1, 0], R[1, 1], 0],
          [      0,       0, 1]]

    theta = np.math.acos(R[0, 0])
    
    reg = cv2.resize(src_image[src_min[1]:ymax, xmin:xmax], new_size)
    
    dx, dy = dst_min
    dx = max(int(dx+(1-kimono_margin_x)*nx/2), 0)
    dy = min(int(dy-(1-kimono_margin_y)*ny/2), np.max(w[0])-1)

    tmpl = np.copy(dst_image)
    y, x = np.where(reg[:, :, 3] != 0)
    yind = np.clip(dy+y, 0, tmpl.shape[0]-1)
    xind = np.clip(dx+x, 0, tmpl.shape[1]-1)
    tmpl[yind, xind] = reg[y, x, :3]
    
    return tmpl


def overlay_kimono_svd_refine(src_image, dst_image, src_point, dst_point):
    _t_points, _o_points = [], []
    for t, o in zip(src_point, dst_point):
        if t[-1] == -1 or o[-1] == -1:
            continue
        _t_points.append(t[:2])
        _o_points.append(o[:2])

    scaled_src_point = np.copy(_t_points)
    scaled_dst_point = np.copy(_o_points)
    scaled_src_point[:, 0] *= src_image.shape[1] 
    scaled_src_point[:, 1] *= src_image.shape[0]
    scaled_dst_point[:, 0] *= dst_image.shape[1] 
    scaled_dst_point[:, 1] *= dst_image.shape[0]

    src_max = np.int32(np.max(scaled_src_point, 0))
    dst_max = np.int32(np.max(scaled_dst_point, 0))
    src_min = np.int32(np.min(scaled_src_point, 0)) 
    dst_min = np.int32(np.min(scaled_dst_point, 0))
    src_mean = np.int32(np.mean(scaled_src_point, 0))
    dst_mean = np.int32(np.mean(scaled_dst_point, 0))

    ratio = (dst_max - dst_min) / (src_max - src_min)

    kimono_margin_x = 1.4
    kimono_margin_y = 1.01
    dx, dy = dst_min
    nx, ny = (src_max - src_min) * ratio
    new_size = (int(nx*kimono_margin_x), int(ny*kimono_margin_y))

    t_points = np.copy(_t_points)
    o_points = np.copy(_o_points)
    t_points -= np.mean(t_points)
    o_points -= np.mean(o_points)

    U, S, Vt = np.linalg.svd(np.dot(t_points.T, o_points))
    R = (U * Vt).T
    R_ = [[R[0, 0], R[0, 1], 0],
          [R[1, 0], R[1, 1], 0],
          [      0,       0, 1]]

    theta = np.math.acos(R[0, 0])
    w = np.where(src_image[:, :, 3] != 0)
    xmax, xmin = np.max(w[1]), np.min(w[1])

    reg = cv2.resize(src_image[src_min[1]:src_max[1], xmin:xmax], new_size)

    dx, dy = dst_min
    dx = max(int(dx+(1-kimono_margin_x)*nx/2), 0)
    dy = min(int(dy-(1-kimono_margin_y)*ny/2), np.max(w[0])-1)
    tmpl = np.copy(dst_image)
    y, x = np.where(reg[:, :, 3] != 0)
    tmpl[dy+y, dx+x] = reg[y, x, :3]
    
    return tmpl


