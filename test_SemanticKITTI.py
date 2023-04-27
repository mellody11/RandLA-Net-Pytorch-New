import warnings
from helper_tool import ConfigSemanticKITTI as cfg
from helper_tool import DataProcessing as DP
from RandLANet import Network, compute_loss, compute_acc, IoUCalculator  
from semantic_kitti_dataset import SemanticKITTI
import numpy as np
import os, argparse
from os.path import exists, join, isfile, dirname, abspath

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
import time
import yaml
import pickle
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint_path', default='output/checkpoint.tar', help='Model checkpoint path [default: None]')
parser.add_argument('--checkpoint_path', default='/data/liuxuexun/mycode/RandLA-Net-Pytorch-New2/train_output/2023-02-27_07-20-21/checkpoint.tar', help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='test_output', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--gpu', type=int, default=1, help='which gpu do you want to use [default: 2], -1 for cpu')
parser.add_argument('--gen_pseudo', default=False, action='store_true', help='generate pseudo labels or not')       
parser.add_argument('--retrain', default=False, action='store_true', help='Re-training with pseudo labels or not')      
parser.add_argument('--test_area', type=str, default='08', help='options: 11,12,13,14,15,16,17,18,19,20,21')


FLAGS = parser.parse_args()

#################################################   log   #################################################
LOG_DIR = FLAGS.log_dir
LOG_DIR = os.path.join(LOG_DIR, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))      # 返回的是英国时间
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)                # 创建多级目录
log_file_name = f'log_test_kitti.txt'
LOG_FOUT = open(os.path.join(LOG_DIR, log_file_name), 'a')      # 追加写入模式


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

test_dataset = SemanticKITTI('test', FLAGS.test_area)
test_dataloader = DataLoader(test_dataset, batch_size=cfg.val_batch_size, shuffle=True, collate_fn=test_dataset.collate_fn)

print(len(test_dataset), len(test_dataloader))

if FLAGS.gpu >= 0:
    if torch.cuda.is_available():
        FLAGS.gpu = torch.device(f'cuda:{FLAGS.gpu:d}')
    else:
        warnings.warn('CUDA is not available on your machine. Running the algorithm on CPU.')
        FLAGS.gpu = torch.device('cpu')
else:
    FLAGS.gpu = torch.device('cpu')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False       




device = FLAGS.gpu

net = Network(cfg)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)

CHECKPOINT_PATH = FLAGS.checkpoint_path
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    log_string("Breakpoint reconnection")

else:
    raise ValueError("No checkpoint: testing code must need a checkpoint.")


BASE_DIR = dirname(abspath(__file__))

data_config = join(BASE_DIR, 'utils', 'semantic-kitti.yaml')
DATA = yaml.safe_load(open(data_config, 'r'))
remap_dict = DATA["learning_map_inv"]

# make lookup table for mapping
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

remap_dict_val = DATA["learning_map"]
max_key = max(remap_dict_val.keys())
remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())



#################################################   test function   ###########################################
def test(dataset, gen_pseudo=False):
    
    # 变量初始化

    idx = 0
    test_probs = [np.zeros(shape=[len(l), cfg.num_classes], dtype=np.float16)
                        for l in dataset.possibility]                                   # 初始化voting的容器
    
    
    test_path = join('test', 'sequences')
    os.makedirs(test_path) if not exists(test_path) else None
    save_path = join(test_path, FLAGS.test_area, 'predictions')
    os.makedirs(save_path) if not exists(save_path) else None
    
    test_smooth = 0.98
    epoch_ind = 0
    
    while True:
        
        net.eval()
        
        for batch_idx, batch_data in enumerate(test_dataloader):
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(len(batch_data[key])):
                        batch_data[key][i] = batch_data[key][i].to(device)
                else:
                    batch_data[key] = batch_data[key].to(device)
        
            with torch.no_grad():
                end_points = net(batch_data)
                
            idx += 1

    
            stacked_probs = end_points['logits'].transpose(1, 2).reshape(-1, cfg.num_classes)                            # logit值，还未经过归一化
            stacked_labels = end_points['labels']
            point_idx = end_points['input_inds'].cpu().numpy()
            cloud_idx = end_points['cloud_inds'].cpu().numpy()
    
            
            stacked_probs = torch.reshape(stacked_probs, [cfg.val_batch_size, cfg.num_points,               # 应该是这个reshape出问题了，不应该reshape，而是transpose？
                                        cfg.num_classes])
            stacked_probs = F.softmax(stacked_probs, dim=2).cpu().numpy()
            stacked_labels = stacked_labels.cpu().numpy()
    
            for j in range(np.shape(stacked_probs)[0]):                     # 循环每个batch统计数据
                probs = stacked_probs[j, :, :]
                inds = point_idx[j, :]
                c_i = cloud_idx[j][0]
                test_probs[c_i][inds] = test_smooth * test_probs[c_i][inds] + (1 - test_smooth) * probs
    

            # if gen_pseudo:
            #     stacked_probs = np.reshape(stacked_probs, [-1, model.config.num_classes])
            #     pred = np.argmax(stacked_probs, axis=-1)
            #     invalid_idx = np.where(labels == 0)[0]
            #     labels_valid = np.delete(labels, invalid_idx)
            #     pred_valid = np.delete(pred, invalid_idx)
            #     labels_valid = labels_valid - 1
            #     correct = np.sum(pred_valid == labels_valid)
            #     acc = correct / float(len(labels_valid))
            #     if self.idx % 10 == 0:
            #         print('step' + str(self.idx) + ' acc:' + str(acc))
                
        new_min = np.min(dataset.min_possibility)
        log_string('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_ind, new_min))
        if np.min(dataset.min_possibility) > 0.5:  # 0.5                                    # 一直进行voting，直到最低的概率高于0.5，停止测试，开始生成最终的label
            log_string(' Min possibility = {:.1f}'.format(np.min(dataset.min_possibility)))

            # if gen_pseudo:
            #     for j in range(len(self.test_probs)):
            #         test_file_name = dataset.test_list[j]
            #         frame = test_file_name.split('/')[-1][:-4]
            #         probs = self.test_probs[j]
            #         for l_ind, label_value in enumerate(dataset.label_values):
            #             if label_value in dataset.ignored_labels:
            #                 probs = np.insert(probs, l_ind, 0, axis=1)

            #         preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
            #         seq_id = test_file_name.split('/')[-3]
            #         label_path = join(dataset.dataset_path, seq_id, 'labels', frame + '.npy')
            #         labels = np.squeeze(np.load(label_path))

            #         # ==================================================== #
            #         #          Generate pseudo labels for subclouds        #
            #         # ==================================================== #
            #         random_ratio = 0.05
            #         trust_ratio = 0.01 / random_ratio
            #         num_pts = len(preds)
                    
            #         trust_preds = np.zeros_like(preds, dtype=np.int32)
            #         random_num = max(int(num_pts * random_ratio), 1)
            #         random_idx = np.random.choice(num_pts, random_num, replace=False)
                    
            #         preds_random_selected = preds[random_idx]
            #         probs_random_selected = probs[random_idx]
            #         probs_random_selected_max_val = np.max(probs_random_selected, axis=1)
            #         trust_idx_all = []
            #         for i in range(dataset.num_classes):
            #             ind_per_class = np.where(preds_random_selected == i)[0]  # idx belongs to class
            #             num_per_class = len(ind_per_class)
            #             if num_per_class > 0:
            #                 trust_num = max(int(num_per_class * trust_ratio), 1)
            #                 probs_max_val_per_class = probs_random_selected_max_val[ind_per_class]
            #                 trust_pts_idx_per_class = probs_max_val_per_class.argsort()[-trust_num:][::-1]
            #                 trust_idx_per_class = ind_per_class[trust_pts_idx_per_class]
            #                 trust_idx_per_class = random_idx[trust_idx_per_class]
            #                 trust_idx_all.append(trust_idx_per_class)
            #         trust_idx_all = np.concatenate(trust_idx_all, axis=0)
            #         trust_preds[trust_idx_all] = preds[trust_idx_all]

            #         print(np.sum(preds[trust_idx_all] == labels[trust_idx_all]) / len(trust_idx_all))
            #         save_name = join(save_path, frame + '.npy')
            #         np.save(save_name, trust_preds)
            #     if gen_pseudo:
            #         return

            log_string('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))

            # For validation set
            num_classes = 19
            gt_classes = [0 for _ in range(num_classes)]
            positive_classes = [0 for _ in range(num_classes)]
            true_positive_classes = [0 for _ in range(num_classes)]
            val_total_correct = 0
            val_total_seen = 0

            for j in range(len(test_probs)):
                test_file_name = dataset.test_list[j]
                frame = test_file_name.split('/')[-1][:-4]
                proj_path = join(dataset.dataset_path, dataset.test_scan_number, 'proj')
                proj_file = join(proj_path, str(frame) + '_proj.pkl')
                if isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds = pickle.load(f)                  # 获取pro文件,用于推理出原来点的label
                probs = test_probs[j][proj_inds[0], :]              # 用于推理所有点的label
                pred = np.argmax(probs, 1)                          # 获得所有点的预测值(从这里往上推，应该是有问题的，预测值非常的零散，并没有连续的统一) [8, 8, 14, 8, 7, 0, 13, 9, 3, 9, 8, 1
                if dataset.test_scan_number == '08':
                    label_path = join(dirname(dataset.dataset_path), 'sequences', dataset.test_scan_number,
                                        'labels')
                    label_file = join(label_path, str(frame) + '.label')
                    labels = DP.load_label_kitti(label_file, remap_lut_val)
                    invalid_idx = np.where(labels == 0)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    pred_valid = np.delete(pred, invalid_idx)
                    labels_valid = labels_valid - 1
                    correct = np.sum(pred_valid == labels_valid)
                    val_total_correct += correct
                    val_total_seen += len(labels_valid)
                    conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, num_classes, 1))
                    gt_classes += np.sum(conf_matrix, axis=1)
                    positive_classes += np.sum(conf_matrix, axis=0)
                    true_positive_classes += np.diagonal(conf_matrix)
                else:       
                    store_path = join(test_path, dataset.test_scan_number, 'predictions',
                                        str(frame) + '.label')
                    pred = pred + 1
                    pred = pred.astype(np.uint32)
                    upper_half = pred >> 16  # get upper half for instances
                    lower_half = pred & 0xFFFF  # get lower half for semantics
                    lower_half = remap_lut[lower_half]  # do the remapping of semantics
                    pred = (upper_half << 16) + lower_half  # reconstruct full label
                    pred = pred.astype(np.uint32)
                    pred.tofile(store_path)
            log_string(str(dataset.test_scan_number) + ' finished')
            if dataset.test_scan_number == '08':
                iou_list = []
                for n in range(0, num_classes, 1):
                    iou = true_positive_classes[n] / float(
                        gt_classes[n] + positive_classes[n] - true_positive_classes[n])
                    iou_list.append(iou)
                mean_iou = sum(iou_list) / float(num_classes)

                log_string('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)))
                log_string('mean IOU:{}'.format(mean_iou))

                mean_iou = 100 * mean_iou
                print('Mean IoU = {:.1f}%'.format(mean_iou))
                s = '{:5.2f} | '.format(mean_iou)
                for IoU in iou_list:
                    s += '{:5.2f} '.format(100 * IoU)
                print('-' * len(s))
                print(s)
                print('-' * len(s) + '\n')
            return    
        epoch_ind += 1
        continue        
                
                
        
    
    


if __name__ == '__main__':
    
    test(test_dataset, FLAGS.gen_pseudo)
    
    
    
    
    
    
    # for data in test_dataloader:
    #     print(len(data['xyz']))
    #     print(data['xyz'][0].shape)
    #     print(data['features'].shape)
    #     print(data['labels'].shape)
    #     print(data['input_inds'].shape)
    #     print(data['cloud_inds'].shape)
    #     print(data['batch_xyz_anno'].shape)
    #     print(data['batch_label_anno'].shape)
    #     break
    

