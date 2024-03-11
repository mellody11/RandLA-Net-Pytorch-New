import pickle, yaml, os, sys
import numpy as np
from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_tool import DataProcessing as DP

data_config = os.path.join(BASE_DIR, 'semantic-kitti.yaml')
DATA = yaml.safe_load(open(data_config, 'r'))
remap_dict = DATA["learning_map"]       # 这个learning map应该是对原始的标签进行一个映射(将原始标签中的某些类别归为一类),因为原始的标签里面类别太多,做语义分割不太好做?
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)       # +100没看懂
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())              # remap_lut向量中key的位置(索引)赋值为value

grid_size = 0.06
dataset_path = '/data/dataset/SemanticKitti/dataset/sequences'
output_path = '/data/liuxuexun/mycode/sequence' + '_' + str(grid_size)
seq_list = np.sort(os.listdir(dataset_path))        # 返回的是00~21的文件夹名字

for seq_id in seq_list:
    print('sequence' + seq_id + ' start')
    seq_path = join(dataset_path, seq_id)           # 某个sequence的路径
    seq_path_out = join(output_path, seq_id)
    pc_path = join(seq_path, 'velodyne')
    pc_path_out = join(seq_path_out, 'velodyne')
    KDTree_path_out = join(seq_path_out, 'KDTree')
    os.makedirs(seq_path_out) if not exists(seq_path_out) else None
    os.makedirs(pc_path_out) if not exists(pc_path_out) else None
    os.makedirs(KDTree_path_out) if not exists(KDTree_path_out) else None           # 以上都是获取路径和创建文件夹目录

    if int(seq_id) < 11:                        # 小于11是训练集
        label_path = join(seq_path, 'labels')
        label_path_out = join(seq_path_out, 'labels')
        os.makedirs(label_path_out) if not exists(label_path_out) else None
        scan_list = np.sort(os.listdir(pc_path))        # 获得当前sequence下的所有点云文件名称
        for scan_id in scan_list:
            print(scan_id)
            points = DP.load_pc_kitti(join(pc_path, scan_id))       # 得到所有点的xyz坐标 该变量的维度是n*3
            labels = DP.load_label_kitti(join(label_path, str(scan_id[:-4]) + '.label'), remap_lut)
            sub_points, sub_labels = DP.grid_sub_sampling(points, labels=labels, grid_size=grid_size)       # 0.06的网格下采样
            search_tree = KDTree(sub_points)                                    # 放进KDTree中进行存储
            KDTree_save = join(KDTree_path_out, str(scan_id[:-4]) + '.pkl')
            np.save(join(pc_path_out, scan_id)[:-4], sub_points)                # 保存点和label
            np.save(join(label_path_out, scan_id)[:-4], sub_labels)
            with open(KDTree_save, 'wb') as f:                          # 保存pkl文件(KDTree文件)
                pickle.dump(search_tree, f)
            if seq_id == '08':                                          # sequence08作为验证集,需要生成下采样点云对原始点云映射的project文件
                proj_path = join(seq_path_out, 'proj')
                os.makedirs(proj_path) if not exists(proj_path) else None
                proj_inds = np.squeeze(search_tree.query(points, return_distance=False))        # 在子云的KDTree中取一个最近邻(没有指定K默认取1个近邻)
                proj_inds = proj_inds.astype(np.int32)
                proj_save = join(proj_path, str(scan_id[:-4]) + '_proj.pkl')
                with open(proj_save, 'wb') as f:
                    pickle.dump([proj_inds], f)
    else:
        proj_path = join(seq_path_out, 'proj')      # 测试集所有sequence都要有proj文件
        os.makedirs(proj_path) if not exists(proj_path) else None
        scan_list = np.sort(os.listdir(pc_path))    # 当前sequence下点的文件名称
        for scan_id in scan_list:       # 下面的代码和上面一样
            print(scan_id)
            points = DP.load_pc_kitti(join(pc_path, scan_id))
            sub_points = DP.grid_sub_sampling(points, grid_size=0.06)
            search_tree = KDTree(sub_points)
            proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            KDTree_save = join(KDTree_path_out, str(scan_id[:-4]) + '.pkl')
            proj_save = join(proj_path, str(scan_id[:-4]) + '_proj.pkl')
            np.save(join(pc_path_out, scan_id)[:-4], sub_points)
            with open(KDTree_save, 'wb') as f:
                pickle.dump(search_tree, f)
            with open(proj_save, 'wb') as f:
                pickle.dump([proj_inds], f)
