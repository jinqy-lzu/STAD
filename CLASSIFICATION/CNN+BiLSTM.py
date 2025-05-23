import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader,ConcatDataset, random_split
import pandas
import argparse
import os
import random
import tensorboard_logger as tb_logger
import torch.backends.cudnn as cudnn
# 定义模型
import torch.nn.functional as F

select_gene = [
"KDM5D",
"PIK3CG",
"IFIH1",
"SLC16A4",
"PDK4",
"CFLAR",
"TNFSF13B",
"UACA",
"Label"]


#CNN+BiLSTM
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(BiLSTMClassifier, self).__init__()
        #Organizing in progress........
    def forward(self, x):
       #Organizing in progress...........
        return out


class CsvDataset(Dataset):
    def __init__(self,filepath="",select_data = []):
        df = pandas.read_csv(
                filepath,
                skip_blank_lines = True,
                header=1
                )
        df = df[select_data]
        print(f'the shape of dataframe is {df.shape}')
        feat = df.iloc[:, :-1].astype(float).values
        label = df.iloc[:, -1].astype(float).values
        # iloc就像python里面的切片操作,此处得到的是numpy数组
        self.x = torch.from_numpy(feat)
        self.y = torch.from_numpy(label)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,index):
        return self.x[index],self.y[index]

from sklearn.model_selection import train_test_split
def creat_split_loader(batch_size = 64,file_path = '',select_data = [], test_size = 0.3):
    csv_dataset = CsvDataset(file_path,select_data=select_data)
    train_data, test_data = train_test_split(csv_dataset, test_size=test_size, random_state=42)
    train_dataloder = DataLoader(train_data,batch_size = batch_size , shuffle = True)
    test_dataloader =  DataLoader(test_data,batch_size = batch_size , shuffle = True)
    return train_dataloder,test_dataloader

def mix_loader(batch_size = 64,test_file_path = '',train_file_path = "", train_ratio = 0.7,select_data = []):
    dataset1 = CsvDataset(test_file_path, select_data=select_data)
    dataset2 = CsvDataset(train_file_path, select_data=select_data)
    combined_dataset = ConcatDataset([dataset1, dataset2])#拼接
    total_size = len(combined_dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def creat_loader(batch_size = 64,file_path = '',select_data = []):
    csv_dataset = CsvDataset(file_path,select_data=select_data)
    csv_dataloder = DataLoader(csv_dataset,batch_size = batch_size , shuffle = F)
    return csv_dataloder



def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in args.schedule:
        args.lr = args.lr * args.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

def accuracy(output, target, topk=1):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = topk
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # for k in topk:
        correct_k = correct[:topk].float().sum()
        return correct_k.mul_(1.0 / batch_size)
         
def unsqueeze_label(label, class_num,device):  
    batch_size = label.shape[0]
    label_tensor = torch.zeros(batch_size, class_num)
    for i in range(batch_size):
        index = label[i].item()
        label_tensor[int(i), int(index)] = 1
    return label_tensor.cuda(device)
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count      
def train(model, train_data, criterion, optimizer,device):
    top1 = AverageMeter()
    train_loss = AverageMeter()
    criterion = nn.CrossEntropyLoss()
    model.train()
    for idx,(batch_x,batch_y) in enumerate(train_data):
        # batch_x = np.log2(batch_x + 1)
        batch_x = batch_x.to(torch.float)
        batch_y = batch_y.to(torch.float)
        batch_x = batch_x.cuda(device)
        batch_y = batch_y.cuda(device)
        label = unsqueeze_label(batch_y, 3, device)
        
        optimizer.zero_grad()
        # batch_x = batch_x.reshape(batch_x.shape[0], 1, -1)
        outputs = model(batch_x)
        # outputs = F.softmax(outputs,dim=1)
        loss = criterion(outputs, label)
        acc1 = accuracy(outputs, batch_y, topk=(1))
        loss.backward()
        optimizer.step()
        top1.update(acc1, batch_y.size(0))
        train_loss.update(loss)
    return top1.avg ,train_loss.avg
def test(model, test_data,criterion, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    top1 = AverageMeter()
    test_loss = AverageMeter()
    label_list = []
    predict_list = []
    predict = []
    label_s = []
   
 
    with torch.no_grad():
        for idx,(batch_x,batch_y) in enumerate(test_data):
            label_list.append(batch_y)
            batch_x = batch_x.to(torch.float)
            batch_y = batch_y.to(torch.float)
            batch_x = batch_x.cuda(device)
            batch_y = batch_y.cuda(device)
            # batch_x = batch_x.reshape(batch_x.shape[0], 1, -1)
            outputs = model(batch_x)
            # outputs = F.softmax(outputs,dim=1)
            predict_list.append(outputs)
            #predict_list = [output.cpu().detach().numpy() for output in outputs]
            label = unsqueeze_label(batch_y, 3, device)
            loss = criterion(outputs, label)
            acc1= accuracy(outputs, batch_y, topk=1)
            top1.update(acc1, batch_y.size(0))
            test_loss.update(loss)      
    return top1.avg, test_loss.avg, predict_list, label_list
def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
def draw_auc(y_score, y_true, path):
    #将list中的内容进行拼接
    for i,item in enumerate(y_score):
        if i == 0:
            score = item
        else:  
            score  = torch.cat((score, item), dim=0)  
    y_score = np.array(score.cpu().detach())
    
    for i,item in enumerate(y_true):
        if i == 0:
            score = item
        else:  
            score  = torch.cat((score, item), dim=0)  
    y_true = np.array(score.cpu().detach())


    n_classes = y_score.shape[1]
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Subtypes {i+1} (area = {roc_auc[i]:.2f})')
        

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    print('AUC-predict_depth:',roc_auc)
    plt.legend()                      
    plt.xlabel(u'False Positive Rate')               
    plt.ylabel(u'True Positive Rate')           
    plt.savefig('{}/roc-auc.pdf'.format(path),format='pdf') 
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=500, type=int)#500
    parser.add_argument('--train_file_path', default='/home/chenkp/CODA/gastric/classification/data/STAD_train_exp_combat1.csv')
    parser.add_argument('--test_file_path', default='/home/chenkp/CODA/gastric/classification/data/STAD_test_exp_combat1.csv')
    parser.add_argument('--predict_label_txt', default='/home/chenkp/CODA/gastric/classification/result/CNN+BiLSTM/label_predict.txt')
    parser.add_argument('--best_model_path', default='/home/chenkp/CODA/gastric/classification/result/CNN+BiLSTM')
    parser.add_argument('--schedule', default=[200, 300, 400], type=int, nargs='+')#[200, 300, 400]
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.05, type=float)#0.05 
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--gpu_id', default='3', type=str)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)#1e-4
    parser.add_argument('--trail', default=5, type=int)
    args = parser.parse_args()
    device = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    gpu_id = int(args.gpu_id)
    lucky_number = 3407 
    seed_torch(lucky_number)

    model = BiLSTMClassifier(input_size= 8, hidden_size=128, num_classes=3,num_layers=3)
    train_list = nn.ModuleList([])
    train_list.append(model)
    optimizer = optim.SGD(train_list.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    logger = tb_logger.Logger(logdir='./CNN+BiLSTM/tensorboard_{}'.format(args.trail), flush_secs=2)

    model_list = nn.ModuleList([])
    model_list.append(model)
    model_list.cuda()
    L_CE_Loss = nn.CrossEntropyLoss()
    criterion = nn.ModuleList([])
    criterion.append(L_CE_Loss)
    criterion.cuda()
    cudnn.benchmark = True
    
    train_data, test_data = mix_loader(batch_size=args.batch_size, train_file_path=args.train_file_path, test_file_path=args.test_file_path,
                                       train_ratio=0.8, select_data=select_gene)
    best_acc = 0
    pred_best = {}
    label_save = {}
    f = open(args.predict_label_txt,'w') 
    for epoch in range(args.epoch):
        adjust_learning_rate(optimizer, epoch, args)
        train_acc1,train_loss = train(model, train_data, criterion, optimizer, device)
        test_acc1, test_loss, score_list, label_list = test(model, test_data, criterion, device)
        print('Epoch: {0:>3d} |GPU: {1:} |Train_Acc: {2:>2.4f} | Test_Acc: {3:>2.4f}'.format(epoch, gpu_id, train_acc1, test_acc1))
        logger.log_value('train_acc', train_acc1, epoch)
        logger.log_value('test_acc', test_acc1, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        if best_acc < test_acc1:
            best_acc = test_acc1
            best_model = model
            pred_best = score_list
            label_save = label_list
    
    draw_auc(score_list, label_list, args.best_model_path)
    # f.write('best_acc: %.2f\n %.4f\n %.4f\n %d\n %d\n'%(test_acc1,args.lr, args.lr_decay, args.epoch, args.schedule))
    print("The best acc1: {},train acc1: {},save best model".format(best_acc,train_acc1))
    input_data = torch.tensor([[0.278, 1.215, 0.29, 0.08, -0.35, -0.54, -0.58, -0.75]], dtype=torch.float32).unsqueeze(0)  # 生成一个新的输入数据
    # input_data = torch.rand(1,1,10)
    model = best_model.eval()
    with torch.no_grad():
        trace_model = torch.jit.script(model.cpu())
        modelfile = "{}/app_best_model_{}.pt".format(args.best_model_path, args.trail)
        torch.jit.save(trace_model, modelfile)
        jit_model = torch.jit.load(modelfile)
        input_data = input_data.to(torch.float)
        output = jit_model(input_data)
        print(output)
        torch.save(best_model.state_dict(), "{}/best_model_{}.pth".format(args.best_model_path, args.trail))
if __name__ == '__main__':
    main()