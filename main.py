import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat

from model import MViT

from sklearn.metrics import confusion_matrix

import numpy as np
from thop import profile
import time
import os
import random

# import visdom
# import numpy as np
# np.array()
rngsd = 1

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--Dataset', choices=['Houston, MUUFL, Augsburg, Trento'], default='Houston', help='dataset to use')
parser.add_argument('--Flag_test', choices=['train', 'test'], default='test', help='testing mark')
parser.add_argument('--Mode', choices=['MViT'], default='MViT', help='mode choice')
parser.add_argument('--Gpu_id', default='0', help='gpu id')
parser.add_argument('--Seed', type=int, default=1, help='number of seed')
parser.add_argument('--Batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--Test_freq', type=int, default=10, help='number of evaluation')
parser.add_argument('--Patches', type=int, default=13, help='number of patches')
parser.add_argument('--Epoches', type=int, default=500, help='epoch number')
parser.add_argument('--Learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--Gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--Weight_decay', type=float, default=0, help='weight_decay')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.Gpu_id)

#-------------------------------------------------------------------------------
def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))
#-------------------------------------------------------------------------------
print("=================================== Parameters ===================================")
print_args(vars(args))
#-------------------------------------------------------------------------------
def select_points(mask, num_classes, select_type, ratio=None, rngsd1=None):
    
    select_size = []
    select_pos = {}
    
    if select_type == 'normal':
        
        for i in range(num_classes):
            each_class = []
            each_class = np.argwhere(mask==(i+1))
            select_size.append(each_class.shape[0])
            select_pos[i] = each_class
    
        total_select_pos = select_pos[0]
        for i in range(1, num_classes):
            total_select_pos = np.r_[total_select_pos, select_pos[i]] #(695,2)
        total_select_pos = total_select_pos.astype(int)
        
    elif select_type == 'random':
                
        for i in range(num_classes):
            each_class = []
            each_class = np.argwhere(mask==(i+1))
            lengthi = each_class.shape[0]
            num = range(1, lengthi)
            
            random.seed(rngsd1)
            nums = random.sample(num, int(lengthi*ratio))
            select_size.append(len(nums))
            select_pos[i] = each_class[nums, :]

    total_select_pos = select_pos[0]
    for i in range(1, num_classes):
        total_select_pos = np.r_[total_select_pos, select_pos[i]] #(695,2)
    total_select_pos = total_select_pos.astype(int)

    return total_select_pos, select_size
#-------------------------------------------------------------------------------
def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域左边镜像
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("Patch size: {}".format(patch))
    print("Padded image shape: [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    return mirror_hsi
#-------------------------------------------------------------------------------
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image
#-------------------------------------------------------------------------------
def prepare_data(mirror_image, label, band, select_point, patch):
    x_select = np.zeros((select_point.shape[0], patch, patch, band), dtype=float)
    y_select = np.zeros(select_point.shape[0], dtype=float)
    for i in range(select_point.shape[0]):
        x_select[i,:,:,:] = gain_neighborhood_pixel(mirror_image, select_point, i, patch)
        y_select[i] = label[select_point[i][0], select_point[i][1]]-1
    return x_select, y_select
#-------------------------------------------------------------------------------
def train_and_test_label(number_train, number_test, num_classes):
    y_train = []
    y_test = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print("y_train: shape = {}, type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {}, type = {}".format(y_test.shape,y_test.dtype))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    return y_train, y_test
#-------------------------------------------------------------------------------
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()
#-------------------------------------------------------------------------------
def train_epoch_MM(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        optimizer.zero_grad()
                
        batch_pred = model(batch_data[:,0:band1,:,:], batch_data[:,band1:,:,:])
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()       

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre
#-------------------------------------------------------------------------------
def valid_epoch_MM(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   
        
        batch_pred = model(batch_data[:,0:band1,:,:], batch_data[:,band1:,:,:])

        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
        
    return top1.avg, objs.avg, tar, pre

def test_epoch(model, test_loader, criterion, optimizer):
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre
#-------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
def mynorm(data, norm_type):
    data_norm = np.zeros(data.shape)    
    if norm_type == 'bandwise':
        for i in range(data.shape[2]):
            data_max = np.max(data[:,:,i])
            data_min = np.min(data[:,:,i])
            data_norm[:,:,i] = (data[:,:,i]-data_min)/(data_max-data_min)
    elif norm_type == 'pixelwise':
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data_max = np.max(data[i,j,:])
                data_min = np.min(data[i,j,:])
                data_norm[i,j,:] = (data[i,j,:]-data_min)/(data_max-data_min)
    return data_norm
#-------------------------------------------------------------------------------
        
# Parameter Setting
args.Seed = rngsd

np.random.seed(args.Seed)
random.seed(args.Seed)
torch.manual_seed(args.Seed)
torch.cuda.manual_seed(args.Seed)
cudnn.deterministic = True
cudnn.benchmark = False

# normalize data by band norm        
norm_type = 'bandwise'#'pixelwise', 'bandwise'
        
# prepare data
if args.Dataset == 'Houston':
    
    folder_data = './dataset/Houston2013/'
    data_HS = loadmat(folder_data + 'Houston_HS.mat')
    data_DSM1 = loadmat(folder_data + 'Houston_LIDAR.mat')
    label_TR = loadmat(folder_data + 'Houston_train.mat')
    label_TE = loadmat(folder_data + 'Houston_test.mat')

    input_HS = mynorm(data_HS['input'], norm_type) #(349, 1905, 144)
    input_DSM1 = np.expand_dims(data_DSM1['input'], axis=-1) #(349, 1905, 1)
    
    height, width, band1 = input_HS.shape    
    _, _, band2 = input_DSM1.shape   
    
    input_MultiModal = np.concatenate((input_HS, input_DSM1), axis=2)
    band_MultiModal = [band1, band2]

    label_TR = label_TR['TR']
    label_TE = label_TE['TE']
    label_Map = np.where(label_TE==0, 2, label_TE)
    num_classes = np.max(label_TR)

    folder_log = './log/HS-LiDAR Houston/' + str(args.Patches) + '/'


elif args.Dataset == 'Trento':
    folder_data = './dataset/Trento/'
    data_HS = loadmat(folder_data + 'HSI.mat')
    data_DSM1 = loadmat(folder_data + 'LiDAR.mat')
    label_TR = loadmat(folder_data + 'TRLabel.mat')
    label_TE = loadmat(folder_data + 'TSLabel.mat')

    input_HS = mynorm(data_HS['HSI'], norm_type)  # (349, 1905, 144)
    input_DSM1 = np.expand_dims(data_DSM1['LiDAR'], axis=-1)  # (349, 1905, 1)

    height, width, band1 = input_HS.shape
    _, _, band2 = input_DSM1.shape

    input_MultiModal = np.concatenate((input_HS, input_DSM1), axis=2)
    band_MultiModal = [band1, band2]

    label_TR = label_TR['TRLabel']
    label_TE = label_TE['TSLabel']

    num_classes = np.max(label_TR)

    folder_log = './log/HS-LiDAR Trento/' + str(args.Patches) + '/'


elif args.Dataset == 'Augsburg':
    folder_data = './dataset/Augsburg/'
    data_HS = loadmat(folder_data + 'data_HS_LR.mat')
    data_DSM1 = loadmat(folder_data + 'data_DSM.mat')
    label_TR = loadmat(folder_data + 'TrainImage.mat')
    label_TE = loadmat(folder_data + 'TestImage.mat')

    input_HS = mynorm(data_HS['data_HS_LR'], norm_type)  # (349, 1905, 144)
    input_DSM1 = np.expand_dims(data_DSM1['data_DSM'], axis=-1)  # (349, 1905, 1)

    height, width, band1 = input_HS.shape
    _, _, band2 = input_DSM1.shape

    input_MultiModal = np.concatenate((input_HS, input_DSM1), axis=2)
    band_MultiModal = [band1, band2]

    label_TR = label_TR['TrainImage']
    label_TE = label_TE['TestImage']

    num_classes = np.max(label_TR)

    folder_log = './log/HS-LiDAR Augsburg/' + str(args.Patches) + '/'

elif args.Dataset == 'MUUFL':
    folder_data = './dataset/MUUFL/'
    data_HS = loadmat(folder_data + 'MUUFL_HS.mat')
    data_DSM1 = loadmat(folder_data + 'MUUFL_LiDAR.mat')
    label_TR = loadmat(folder_data + 'MUUFL_TR.mat')
    label_TE = loadmat(folder_data + 'MUUFL_gt.mat')



    input_HS = mynorm(data_HS['MUUFL_HS'], norm_type)  # (349, 1905, 144)
    input_DSM1 = np.expand_dims(data_DSM1['MUUFL_Lidar'][:,:,1], axis=-1)  # (349, 1905, 1)

    height, width, band1 = input_HS.shape
    _, _, band2 = input_DSM1.shape

    input_MultiModal = np.concatenate((input_HS, input_DSM1), axis=2)
    band_MultiModal = [band1, band2]

    label_TR = label_TR['TR']
    label_TE = label_TE['MUUFL_gt']

    num_classes = np.max(label_TR)

    folder_log = './log/HS-LiDAR MUUFL/' + str(args.Patches) + '/'

else:
    raise ValueError("Unknown dataset")

if not os.path.exists(folder_log):
    os.makedirs(folder_log)

#-------------------------------------------------------------------------------
# obtain train positions
select_type = 'normal'
total_pos_TR, number_TR = select_points(label_TR, num_classes, select_type)
# obtain test positions
select_type = 'normal'
label_Map = np.where(label_TE==0, 1, label_TE)
each_class = []
each_class = np.argwhere(label_Map != 0)
total_pos_TE1, number_TE1 = select_points(label_Map, num_classes, select_type)
total_pos_TE1 = each_class


# total_pos_TE1 = np.array_split(total_pos_TE1,2,axis=0)
total_pos_TE, number_TE = select_points(label_TE, num_classes, select_type)


## test
mirror_image = mirror_hsi(height, width, np.sum(band_MultiModal), input_MultiModal, patch=args.Patches)
# obtain train data from train positions
x_TR_patch, y_TR = prepare_data(mirror_image, label_TR, np.sum(band_MultiModal), total_pos_TR, patch=args.Patches)
# obtain test data from test positions
# x_TE_patch1, y_TE1 = prepare_data(mirror_image, label_Map, np.sum(band_MultiModal), total_pos_TE1, patch=args.Patches)

x_TE_patch, y_TE = prepare_data(mirror_image, label_TE, np.sum(band_MultiModal), total_pos_TE, patch=args.Patches)

#-------------------------------------------------------------------------------
# load data
x_TR = torch.from_numpy(x_TR_patch.transpose(0,3,1,2)).type(torch.FloatTensor) #[#TR, band, patch*patch]
y_TR = torch.from_numpy(y_TR).type(torch.LongTensor)
Label_TR = Data.TensorDataset(x_TR, y_TR)

x_TE=torch.from_numpy(x_TE_patch.transpose(0,3,1,2)).type(torch.FloatTensor)
y_TE=torch.from_numpy(y_TE).type(torch.LongTensor)
Label_TE = Data.TensorDataset(x_TE, y_TE)
Label_TR_loader = Data.DataLoader(Label_TR, batch_size=args.Batch_size, shuffle=True)
Label_TE_loader = Data.DataLoader(Label_TE, batch_size=args.Batch_size, shuffle=True)

# x_TE1=torch.from_numpy(x_TE_patch1.transpose(0,3,1,2)).type(torch.FloatTensor)
# y_TE1=torch.from_numpy(y_TE1).type(torch.LongTensor)
# Label_TE1 = Data.TensorDataset(x_TE1, y_TE1)
# Label_TE_loader1 = Data.DataLoader(Label_TE1, batch_size=args.Batch_size, shuffle=False)

#-------------------------------------------------------------------------------
# create model
model = MViT(
    patch_size = args.Patches,
    num_patches = band_MultiModal,
    num_classes = num_classes,
    dim = 64,
    depth = 6,
    heads = 8,
    mlp_dim = 32,
    dropout = 0.1,
    emb_dropout = 0.1,
    mode = args.Mode
)
# print(model)
model = model.cuda()
# criterion
criterion = nn.CrossEntropyLoss().cuda()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.Learning_rate, weight_decay=args.Weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=args.Gamma)

#-------------------------------------------------------------------------------
if args.Flag_test == 'test':


    # input = torch.randn(64, 144, 13, 13).cuda()
    # input2 = torch.randn(64, 1, 13, 13).cuda()
    # Flops, params = profile(model, inputs=(input,input2))  #
    # print('Flops: % .4fM' % (Flops / 1000000))  #
    # print('params参数量: % .4fK' % (params / 1000))  #
    print("=================================== Testing ===================================")

    PATH = './log/HS-LiDAR MUUFL/15/MUUFL325.pt'

    model.load_state_dict(torch.load(PATH))

    model.eval()

    pre_total = []

    test_acc, test_obj, tar_v, pre_v = valid_epoch_MM(model, Label_TE_loader, criterion, optimizer)
    # pre_v1 = pre_v.reshape(349,381)
    np.savetxt('muufl.txt', pre_v, fmt="%d",delimiter=" ")  # 将结果保存到文本文件
    print("fengefu")
    OA_TE, AA_TE, Kappa_TE, CA_TE = output_metric(tar_v, pre_v)
    print(">>> Inference finished!")

    print("=================================== Results ===================================")
    print("OA: {:.2f} | AA: {:.2f} | Kappa: {:.4f}".format(OA_TE*100, AA_TE*100, Kappa_TE))
    np.set_printoptions(precision=2, suppress=True)
    print("CA: ", CA_TE*100)
    
elif args.Flag_test == 'train':
    
    # vis = visdom.Visdom(use_incoming_socket=False)
    # assert vis.check_connection()
    
    best_checkpoint = {"OA_TE": 0.50}
    
    print("=================================== Training ===================================")
    tic = time.time()
    for epoch in range(args.Epoches): 
        
        model.train()
        
        train_acc, train_obj, tar_t, pre_t = train_epoch_MM(model, Label_TR_loader, criterion, optimizer)

        scheduler.step()

        OA_TR, AA_TR, Kappa_TR, CA_TR = output_metric(tar_t, pre_t) 

        # vis.line(X=np.array([epoch]), Y=np.array([train_acc.data.cpu().numpy()]), win='train_acc', update='append', opts={'title':'Train Accuracy'})
        # vis.line(X=np.array([epoch]), Y=np.array([train_obj.data.cpu().numpy()]), win='train_obj', update='append', opts={'title':'Train Loss'})
        print("Epoch: {:03d} train_loss: {:.4f}, train_OA: {:.2f}".format(epoch + 1, train_obj, OA_TR * 100))
        if (epoch % args.Test_freq == 0) | (epoch == args.Epoches - 1):
            


            model.eval()
            test_acc, test_obj, tar_v, pre_v = valid_epoch_MM(model, Label_TE_loader, criterion, optimizer)
            OA_TE, AA_TE, Kappa_TE, CA_TE = output_metric(tar_v, pre_v)
            print("Epoch: {:03d} test_loss: {:.4f}, test_OA: {:.2f}, test_AA: {:.2f}, test_Kappa: {:.4f}".format(epoch+1, train_obj, OA_TE*100, AA_TE*100, Kappa_TE))

            # vis.line(X=np.array([epoch]), Y=np.array([test_acc.data.cpu().numpy()]), win='test_oa', update='append', opts={'title':'Test Overall Accuracy'})
            # vis.line(X=np.array([epoch]), Y=np.array([AA_TE*100]), win='test_aa', update='append', opts={'title':'Test Average Accuracy'})
            # vis.line(X=np.array([epoch]), Y=np.array([test_obj.data.cpu().numpy()]), win='test_obj', update='append', opts={'title':'Test Loss'})
            
            if OA_TE*100>best_checkpoint['OA_TE']:
                best_checkpoint = {'epoch': epoch, 'OA_TE': OA_TE*100, 'AA_TE': AA_TE*100, 'Kappa_TE': Kappa_TE, 'CA_TE': CA_TE*100}
                
            PATH = folder_log + args.Dataset + str(epoch) + '.pt'
            torch.save(model.state_dict(), PATH)

    toc = time.time()
    runtime = toc - tic
    print(">>> Training finished!")

    print(">>> Running time: {:.2f}".format(runtime))
    print("=================================== Results ===================================")

    print(">>> The peak performance in terms of OA is achieved at epoch", best_checkpoint['epoch'])
    print("OA: {:.2f} | AA: {:.2f} | Kappa: {:.4f}".format(best_checkpoint['OA_TE'], best_checkpoint['AA_TE'], best_checkpoint['Kappa_TE']))
    np.set_printoptions(precision=2, suppress=True)
    print("CA: ", best_checkpoint['CA_TE'])
    
    output_txt_path = os.path.join(folder_log, 'precision.txt')
    write_message = "Patch size {}, weight decay {}, learning rate {}, the best epoch {}, OA {}, AA {}, Kappa {}, run time {}".format(args.Patches, args.Weight_decay, args.Learning_rate, best_checkpoint['epoch'], round(best_checkpoint['OA_TE'],2), round(best_checkpoint['AA_TE'],2), round(best_checkpoint['Kappa_TE'],4), round(runtime,2))
    
    output_txt_file = open(output_txt_path, "a")
    now = time.strftime("%c")
    output_txt_file.write('=================================== Precision Log (%s) ===================================\n' % now)
    output_txt_file.write('%s\n' % write_message)
    output_txt_file.close()
    
    # vis.close('train_acc')
    # vis.close('train_obj')
    # vis.close('test_acc')
    # vis.close('test_obj')
