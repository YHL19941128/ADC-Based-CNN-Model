# -*- coding = utf-8 -*-

import  torch
from torchvision import transforms
from    torch import optim, nn
import visdom
import time
import torchvision
import numpy as np
import pandas as pd
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from build_dataset import Mydata
from net_me import Netme
from  my_utils import Flatten
from my_utils import initialize_weights
from PIL import Image

batchsz = 4
lr = 1e-3
epochs = 60
torch.manual_seed(3167)
visdom_switch = 'off'
image_path = r"C:\Users\..."
saveswitch = 'on'
savepath = r'C:\Python\...'
save_trainval ='on'


train_db = Mydata(image_path, 128, mode='train')
val_db = Mydata(image_path, 128, mode='val')
test_db = Mydata(image_path, 128, mode='test')

train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,num_workers=4)
val_loader = DataLoader(val_db, batch_size=batchsz, shuffle=False, num_workers=1)
test_loader = DataLoader(test_db, batch_size=batchsz, shuffle=False, num_workers=1)

model = Netme(2)


optimizer= optim.Adam(model.parameters(), lr=lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer)
criterion = nn.CrossEntropyLoss()


if visdom_switch == 'on':
    viz = visdom.Visdom()


def train(epoch):
    running_loss = 0.0
    correct = 0
    total = 0
    train_batchloss=[]

    for batch_idx, data in enumerate(train_loader, 0):
        imgtensor, labeltensor = data[0], data[1]

        '''forward + backward + update'''
        optimizer.zero_grad()
        outputs = model(imgtensor)
        loss = criterion(outputs, labeltensor)
        loss.backward()
        optimizer.step()
        probability, predicted = torch.max(outputs.data, dim=1)
        running_loss += loss.item()
        total += labeltensor.size(0)
        correct += (predicted == labeltensor).sum().item()
        train_batchloss.append(loss.item())
    train_epoch_loss=running_loss / (total/batchsz)
    train_accuracy =100 * correct / total
    print('[%d]epoch_loss: %.3f' % (epoch + 1, train_epoch_loss))
    print('Accuracy on train set: %d %%' % (train_accuracy))
    running_loss = 0.0
    return train_batchloss, train_epoch_loss, train_accuracy

def evalute(epoch,loader):
    running_loss = 0.0
    correct = 0
    total = 0
    batchloss, predictedlabel, probabilitylist, truelabel, imgpathlist= [], [], [], [], []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader, 0):
            imgtensor, labeltensor, imgpath = data
            outputs = model(imgtensor)
            loss = criterion(outputs, labeltensor)
            probability = outputs.data[:,1]
            _, predicted = torch.max(outputs.data, dim=1)
            total += labeltensor.size(0)
            correct += (predicted == labeltensor).sum().item()
            running_loss += loss.item()
            batchloss.append(loss.item())
            temp1 = np.concatenate([predictedlabel, predicted.numpy()], axis=0)
            predictedlabel = temp1
            temp2 = np.concatenate([truelabel, labeltensor.numpy()], axis=0)
            truelabel = temp2
            temp3 = np.concatenate([imgpathlist, imgpath], axis=0)
            imgpathlist = temp3
            temp4 = np.concatenate([probabilitylist, probability.numpy()], axis=0)
            probabilitylist = temp4

    epoch_loss = running_loss / (total/batchsz)
    accuracy = 100 * correct / total
    if loader == val_loader:
        print('--[%d]_epoch的loss: %.3f' % (epoch + 1, epoch_loss))
        print('Accuracy on validation set: %d %%' % (accuracy))
    elif loader == test_loader:
        print('Vloss: %.3f' % (epoch_loss))
        print('Accuracy on test set: %d %%' % (accuracy))
    else:
        print('loss: %.3f' % (epoch_loss))
        print('Accuracy on this set: %d %%' % (accuracy))

    running_loss = 0.0
    return batchloss, epoch_loss, accuracy, predictedlabel, probabilitylist, truelabel, imgpathlist


if __name__ == '__main__':

    if visdom_switch == 'on':
        viz.line(Y=[0], X=[-1], win='train_loss', opts=dict(title='train_loss', xlabel="Batchturn", ylabel="Loss"))
        viz.line([0], [-1], win='val_loss', opts=dict(title='val_loss'))
        viz.line([0], [-1], win='train_acc', opts=dict(title='train_acc'))
        viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))

    train_batch_loss_all, train_epoch_loss_all, train_accuracy_all, train_epoch_all= [], [], [], []
    val_batch_loss_all, val_epoch_loss_all, val_accuracy_all, val_epoch_all = [], [], [], []
    test_batch_loss, test_epoch_loss, test_accuracy, test_predictedlabel, test_probability = [], [], [], [], []
    best_accuracy, best_epoch = 0, 0
    global_train_batchix,global_val_batchix = 0, 0

    for epoch in range(epochs):


        train_batch_loss, train_epoch_loss, train_accuracy= train(epoch)

        if visdom_switch == 'on':
            for train_singlebatch_loss in train_batch_loss:
                global_train_batchix += 1
                viz.line([train_singlebatch_loss], [global_train_batchix], win='train_loss',update='append')
            viz.line([train_accuracy], [epoch], win='train_acc',update='append')
        train_batch_loss_all.extend(train_batch_loss)
        train_epoch_loss_all.append(train_epoch_loss)
        train_accuracy_all.append(train_accuracy)
        train_epoch_all.append(epoch)


        if epoch % 1 == 0:
            val_batch_loss, val_epoch_loss, val_accuracy, val_predictedlabel, val_probability, val_truelabel, val_imgpathlist= evalute(epoch,val_loader)
            if val_accuracy > best_accuracy:
                best_epoch = epoch
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), savepath + r'\bestmodel.mdl')

            if visdom_switch == 'on':
                for val_singlebatch_loss in val_batch_loss:
                    global_val_batchix += 1
                    viz.line([val_singlebatch_loss], [global_val_batchix], win='val_loss',update='append')
                viz.line([val_accuracy], [epoch], win='val_acc',update='append')

            val_batch_loss_all.extend(val_batch_loss)
            val_epoch_loss_all.append(val_epoch_loss)
            val_accuracy_all.append(val_accuracy)
            val_epoch_all.append(epoch)

    torch.save(model.state_dict(), savepath + r'\lastmodel.mdl')

    print('***********val_best acc:%d %%*************'%(best_accuracy) +'\n'+ '***********val_best epoch:%d*************'%(best_epoch+1))

    model.load_state_dict(torch.load(savepath + r'\bestmodel.mdl'))
    print('------Ok!--------')

    test_batch_loss, test_epoch_loss, test_accuracy, test_predictedlabel, test_probability, test_truelabel, test_imgpathlist = evalute(best_epoch, test_loader)
    print('***********test loss:%d :*************'%(test_epoch_loss))
    print('***********test acc:%d %%*************'%(test_accuracy))

    if saveswitch == 'on':
        train_alldata = pd.DataFrame([train_epoch_all, train_accuracy_all, train_epoch_loss_all],
                                     index=['train_epoch', 'train_accuracy', 'train_epoch_loss'])
        train_alldata.to_csv(savepath + r'\train_alldata.csv')

        val_alldata = pd.DataFrame([val_epoch_all, val_accuracy_all, val_epoch_loss_all],
                                   index=['val_epoch', 'val_accuracy', 'val_epoch_loss'])
        val_alldata.to_csv(savepath + r'\val_alldata.csv')

        trainandval_batch_loss_all=pd.DataFrame([train_batch_loss_all,val_batch_loss_all],
                                                index=['train_batch_loss_all', 'val_batch_loss_all'])
        trainandval_batch_loss_all.to_csv(savepath + r'\trainandval_batch_loss_all.csv')

        test_predictedlabel = pd.DataFrame([test_predictedlabel, test_probability, test_truelabel, test_imgpathlist],
                                     index=['test_predictedlabel', 'test_probability', 'test_truelabel', 'test_imgpathlist'])
        test_predictedlabel.to_csv(savepath + r'\test_predictedlabel.csv')

        with open(savepath + r'\test_infor.txt','w') as f:
            f.write('val_best acc:%d %%'%(best_accuracy) +'\t'+'val_best epoch:%d'%(best_epoch+1)+'\r'+'test acc:%d %%:'%(test_accuracy))

        print('save_final！')

    if visdom_switch == 'on':
        i = 0
        while i <= (test_imgpathlist.shape[0]-1):
            truelabel=test_truelabel[i]
            predictedlabel=test_predictedlabel[i]
            imgpath=test_imgpathlist[i]
            i=i+1
            imgdata_from_path = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.ToTensor(),
                ])
            imgtensor = imgdata_from_path(str(imgpath))
            viz.images(imgtensor, nrow=6, win='batch', opts=dict(title='batch'))
            viz.text(str(truelabel), win='truelabel', opts=dict(title='true'))
            viz.text(str(predictedlabel), win='predictedlabel', opts=dict(title='predicted'))
            time.sleep(3)

    if save_trainval == 'on':
        val_batch_loss, val_epoch_loss, val_accuracy, val_predictedlabel, val_probabilitylist, val_truelabel, val_imgpathlist = evalute(best_epoch, val_loader)
        train_batch_loss, train_epoch_loss, train_accuracy, train_predictedlabel, train_probabilitylist, train_truelabel, train_imgpathlist = evalute(best_epoch, train_loader)

    if save_trainval == 'on':
        val_predictedlabel = pd.DataFrame([val_predictedlabel, val_truelabel, val_imgpathlist],
                                           index=['val_predictedlabel', 'val_truelabel', 'val_imgpathlist'])
        val_predictedlabel.to_csv(savepath + r'\val_predictedlabel.csv')

        train_predictedlabel = pd.DataFrame([train_predictedlabel, train_truelabel, train_imgpathlist],
                                           index=['train_predictedlabel', 'train_truelabel', 'train_imgpathlist'])
        train_predictedlabel.to_csv(savepath + r'\train_predictedlabel.csv')

        print('V_and_T OK！')










