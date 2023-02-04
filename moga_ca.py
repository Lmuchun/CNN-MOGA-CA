import rasterio
import rasterio.shutil
import cupy as cp
from euci_neighbor import count_neighbour_vectorize
import pylandstats as pls
import numpy as np
import random


def read_img(img_path):
    with rasterio.open(img_path) as dst:
        count = dst.count
        if count == 1:
            band = cp.asarray(dst.read(1), dtype=cp.uint8)
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


def count_land_use(img, landuse_count=6):
    land_use_count_list = []
    for idx in range(landuse_count):
        landuse = idx + 1
        land_use_count_list.append(int(cp.count_nonzero(img == landuse)))
    return cp.array(land_use_count_list)


CHANGE_COST = cp.array([
    [0, 1, 0, 1, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0]])


def write_img(img, dst_filename, result):

    out_filename = '{:.5f}_{:.5f}.tif'.format(result[0], result[1])
    # 参考地图
    with rasterio.open(dst_filename) as dst:
        profile = dst.profile
    if rasterio.shutil.exists(out_filename):
        rasterio.shutil.delete(out_filename)

    with rasterio.open(out_filename, 'w', **profile) as dst:
        dst.write(cp.asnumpy(img), 1)
        return 0


def landscape_LPI(img_begin):
    img_begin = cp.asnumpy(img_begin)
    ls = pls.Landscape(img_begin, res=(100, 100))
    class_metrics_df = ls.compute_landscape_metrics_df(
        metrics=['largest_patch_index'])
    LPI = abs(29.6761-class_metrics_df.iat[0, 0])
    return LPI


def landscape_ED(img_begin):
    img_begin = cp.asnumpy(img_begin)
    ls = pls.Landscape(img_begin, res=(100, 100))
    class_metrics_df = ls.compute_landscape_metrics_df(
        metrics=['edge_density'])
    ED = abs(34.8468-class_metrics_df.iat[0, 0])
    return ED


def Kappa(img_simu, img_end, valid_con, p=0.2):

    # 随机采样矩阵
    sample = cp.random.choice(a=[True, False], size=(
        879, 879), p=[p, 1-p]) * valid_con
    sample_num = cp.count_nonzero(sample)
    img_simu = img_simu * sample
    img_end = img_end * sample
    po = cp.sum((img_simu == img_end) * sample)/sample_num
    num = 0
    for i in range(1, 7):
        temp = cp.count_nonzero(img_simu == i)*cp.count_nonzero(img_end == i)
        num += temp
    pe = num/sample_num/sample_num
    kappa = (po-pe)/(1-pe)
    return kappa


def FoM(img_simu, img_end, img_begin, valid_con):
    a = cp.sum((img_simu != img_begin) & (img_begin == img_end) & valid_con)
    b = cp.sum((img_simu != img_begin) & (
        img_end != img_begin) & (img_simu == img_end) & valid_con)
    c = cp.sum((img_simu != img_begin) & (
        img_end != img_begin) & (img_simu != img_end) & valid_con)
    d = cp.sum((img_simu == img_begin) & (img_begin != img_end) & valid_con)

    Fom = b/(a+b+c+d)
    return Fom


def target(img_simu, img_end, img_begin, valid_con):

    kapa = Kappa(img_simu, img_end, valid_con)
    fom = FoM(img_simu, img_end, img_begin, valid_con)
    # ED = landscape_ED(img_simu)

    kapa = cp.float_(kapa)
    fom = cp.float_(fom)
    # ED = cp.float_(ED)

    result = [kapa, fom]
    return result


def ca(neighbour_weight):

    neighbour_weight = neighbour_weight.reshape((6, 6, 3))
    begin_img_path = 'lu_05.tif'
    end_img_path = 'lu_15.tif'
    p_path = 'probability.tif'
    dispart_path = 'dispart.tif'
    number_of_iter = 35
    moor_radius = 11
    part, part_con = read_img(dispart_path)
    img_begin, con0 = read_img(begin_img_path)
    img_end, con1 = read_img(end_img_path)
    img_possibility_of_occurence, con2 = read_img(p_path)

    img_init = img_begin.copy()  # 初始地图，无改变
    valid_con = con1 & con0 & con2
    del con1, con2, con0

    goal_list = count_land_use(img_end)
    land_use_count_begin = count_land_use(img_begin)
    land_use_count_cur = land_use_count_begin.copy()
    land_use_count_goal = goal_list
    # print(land_use_count_goal)
    bol = cp.full((879, 879), True) & valid_con
    for iter_idx in range(number_of_iter):

        land_use_count_cur = count_land_use(img_begin)  # 在迭代中更改
        cur_diff_list = abs(land_use_count_goal-land_use_count_cur)  # t-1 diff
        # print(land_use_count_cur)
        if iter_idx > 5:
            if (sum(cur_diff_list) < 50):
                result = target(img_begin, img_end, img_init,
                                valid_con & (part == 11))
                if result[0] > 0.8 and result[1] > 0.2:
                    write_img(img_begin, begin_img_path, result)
                return result

        # 计算邻域效应
        img_neighbour = count_neighbour_vectorize(
            img_begin, moor_radius, neighbour_weight)
        possibility_of_change = img_neighbour * img_possibility_of_occurence
        possibility_of_change_mtx = possibility_of_change / \
            cp.sum(possibility_of_change, axis=2, keepdims=True)
        roulette_r = (cp.cumsum(possibility_of_change_mtx, axis=2) > cp.random.uniform(size=(
            possibility_of_change_mtx.shape[0], possibility_of_change_mtx.shape[1], 1), dtype=cp.float32)).astype(cp.int8)  # 获得轮盘结果
        # 轮盘赌
        roulette_r = (6 - cp.sum(roulette_r, axis=2)) + 1

        land_type_list_1 = cp.arange(1, 7)
        land_type_list_2 = cp.array([6, 5, 4, 3, 2, 1])

        for old_land_type in land_type_list_1:
            for new_land_type in land_type_list_2:
                if old_land_type == new_land_type:
                    continue
                if CHANGE_COST[old_land_type-1, new_land_type-1] == 0:
                    continue
                old_begin_count = land_use_count_begin[old_land_type-1]
                old_goal_count = land_use_count_goal[old_land_type - 1]
                old_cur_count = land_use_count_cur[old_land_type-1]

                new_begin_count = land_use_count_begin[new_land_type-1]
                new_goal_count = land_use_count_goal[new_land_type - 1]
                new_cur_count = land_use_count_cur[new_land_type-1]

                if iter_idx > 4:
                    if old_cur_count <= old_goal_count:
                        continue
                    if new_cur_count >= new_goal_count:
                        continue
                # 轮盘赌的获胜type
                roul_con = roulette_r == new_land_type
                # 从oldtype改变
                chan_con = img_begin == old_land_type
                con_all = valid_con & roul_con & chan_con & bol
                del roul_con, chan_con

                count_of_change = cp.count_nonzero(con_all)
                if count_of_change == 0:
                    continue
                if count_of_change > 0:
                    # 限制一次转移不过量
                    limit_cnt = int(land_use_count_begin[old_land_type-1]/3)
                    if count_of_change > limit_cnt:

                        new_p_list = img_possibility_of_occurence[:,
                                                                  :, new_land_type-1]*con_all
                        i = cp.argpartition(
                            new_p_list.ravel(), -limit_cnt)[-limit_cnt:]
                        i2d = cp.unravel_index(i, new_p_list.shape)
                        con_lim = cp.zeros(con_all.shape, cp.bool_())
                        con_lim[i2d] = True
                        con_all = con_all & con_lim

                        count_of_change = cp.count_nonzero(con_all)

                    old_tmp_count = old_cur_count - count_of_change
                    new_tmp_count = new_cur_count + count_of_change

                    if ((old_begin_count >= old_goal_count) and (old_tmp_count < old_goal_count)) or \
                            ((new_begin_count <= new_goal_count) and (new_tmp_count > new_goal_count)):
                        diff_old = old_cur_count - old_goal_count
                        diff_new = new_goal_count - new_cur_count

                        count_of_change_adj = min(diff_old, diff_new)
                        diff_count_of_change = count_of_change - count_of_change_adj
                        row_list, col_list = cp.where(con_all)
                        randomize = cp.arange(len(row_list))
                        cp.random.shuffle(randomize)
                        randomize = randomize[:diff_count_of_change]
                        row_list_adj = row_list[randomize]
                        col_list_adj = col_list[randomize]
                        con_all[row_list_adj, col_list_adj] = False

                    count_of_change = cp.count_nonzero(con_all)
                    land_use_count_cur[old_land_type-1] -= count_of_change
                    land_use_count_cur[new_land_type-1] += count_of_change
                    img_begin[con_all] = new_land_type
                    bol[con_all] = False

        del roulette_r, img_neighbour, possibility_of_change_mtx
    return [0.01, 0.01]


''''
if __name__ == '__main__':
    global type_weight
    weight_path = 'Phen.csv'
    data = np.loadtxt(open(weight_path, "rb"), delimiter=",", dtype='str')
    weight = data[0, :]
    type_weight = weight.astype('float64')
    c = ca(type_weight)
'''''
