
import numpy as np
import cv2 

def overlay_kimono(src_image, dst_image, src_point, dst_point):
    t_point = src_point
    o_point = dst_point
    _t_points, _o_points = [], []
    for t, o in zip(src_point, dst_point):
        if t[-1] == -1 or o[-1] == -1:
            continue
        _t_points.append(t[:2])
        _o_points.append(o[:2])

    base_image = dst_image.copy()
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

    dx, dy = dst_min
    nx, ny = (src_max - src_min) * ratio
    new_size = (int(nx), int(ny))
    reg = cv2.resize(src_image[src_min[1]:src_max[1], src_min[0]:src_max[0]], new_size)
    tmpl = np.copy(dst_image)
    y, x = np.where(reg[:, :, 3] != 0)
    tmpl[dy+y, dx+x] = reg[y, x, :3]
    
    return tmpl
