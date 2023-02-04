import cupy as cp
import numpy as np
import rasterio
from skimage.morphology import square, dilation


def read_img(img_path):
    with rasterio.open(img_path) as dst:
        count = dst.count
        if count == 1:
            band = cp.asarray(dst.read(1), dtype=cp.float16)
            nodata = dst.nodata
            con = (band != nodata)
            return band, con
        else:
            band_list = []
            con = None
            nodata = dst.nodata
            for idx in range(count):
                band = cp.asarray(dst.read(idx+1), dtype=cp.float16)
                band_list.append(band)
                if idx == 0:
                    con = (band != nodata)
                else:
                    con &= (band != nodata)
            band = cp.stack(band_list, axis=2)
            return band, con


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def normal(matrix):
    _max = np.max(matrix, axis=2)
    _min = np.min(matrix, axis=2)
    _max = np.expand_dims(_max, axis=2)
    _min = np.expand_dims(_min, axis=2)
    temp = (matrix-_min)/(_max-_min)
    return temp


def decay(decay_core, x):
    a = decay_core[0]
    b = decay_core[1]
    c = decay_core[2]
    if c == 0:
        c = 0.0001
    mat = a*cp.exp(-pow((x-b), 2)/(2*pow(c, 2)))
    return mat


def euci_dist(input, moor_radius, value):

    input = cp.asnumpy(input)
    temp = np.zeros(input.shape)
    for i in range(3, 13, 2):
        temp += dilation(input.astype(int), square(i))
    out = (6-temp) % 6
    return out


def count_neighbour_vectorize(img, moor_radius, weight_decay):

    land_use_count = 6
    dead_span = moor_radius // 2
    new_m = img.shape[0] + (dead_span * 2)
    new_n = img.shape[1] + (dead_span * 2)
    neighbour_all = []
    for core in range(land_use_count):
        core_type = core+1
        # 欧几里得距离地图
        euci_input = (img == core_type)
        fuc_temp = euci_dist(euci_input, moor_radius, core_type)
        fuc_temp = cp.asarray(fuc_temp)
        # 边界扩充
        fuc_dis = cp.zeros((new_m, new_n), dtype=cp.int8)
        fuc_dis[dead_span:-dead_span, dead_span:-dead_span] = fuc_temp
        decay_core = weight_decay[core]
        # 读取距离衰减参数
        neighbour_img_list = []

        for index in range(land_use_count):
            land_type = index + 1
            landuse_img = cp.zeros((new_m, new_n), dtype=cp.int8)
            landuse_img[dead_span:-dead_span,
                        dead_span:-dead_span] = (img == land_type)
            # 获取所有core邻域的用地
            fuc_input = fuc_dis*landuse_img

            decay_func = decay(decay_core[index], fuc_input)*(fuc_input != 0)
            neighbour_img = cp.zeros(img.shape, dtype=cp.float16)
            for row in range(0, moor_radius):
                for col in range(0, moor_radius):

                    if row == dead_span and row == dead_span:
                        continue
                    row_end = new_m - (moor_radius - 1 - row)
                    col_end = new_n - (moor_radius - 1 - col)
                    neighbour_img += decay_func[row:row_end, col:col_end]

            neighbour_img_list.append(neighbour_img*euci_input)
        neighbour_core = cp.stack(neighbour_img_list, axis=2)
        neighbour_all.append(neighbour_core)
    neighbour_output = sum(neighbour_all)

    del neighbour_img_list, neighbour_all, neighbour_core
    return normal(neighbour_output)
