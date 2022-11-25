import torch 
import cv2
import numpy as np
import os 
import glob
from torch.utils.data import DataLoader,Dataset
from torch import optim,nn 
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt


np.random.seed(1)
torch.manual_seed(3)
class ShapeDataset(Dataset):
    def __init__(self,path):
        self.class_label = {"circle":0,"rectangle":1,"triangle":2,"square":3,"star":4,"bg":5}
        self.Image_path_list = []

        for data_path in glob.glob(path+"**/*.jpg"):
            self.Image_path_list.append(data_path)
       
        np.random.shuffle(self.Image_path_list)
        
    def get_label_box(self,csvFileName,  image_size ):
       
        df = pd.read_csv(csvFileName,
        usecols=['ClassName','X','Y','Width','Height'])
        df = np.array(df)[0]
        # print(df[1:].astype(np.float32))
        label ,boxes = self.class_label[df[0]],(df[1:].astype(np.float32)/image_size)
        boxes = torch.from_numpy(boxes)
        return label,boxes

    def __getitem__(self, index):
        img = cv2.imread(self.Image_path_list[index],0)
        image_size = 224
        img = cv2.resize(img,(image_size,  image_size ),interpolation=cv2.INTER_AREA).reshape(1,image_size,image_size)
        img = torch.from_numpy(img)/255.0
        one_hot_label = torch.zeros(6,dtype=torch.float32)
        label,boxes = self.get_label_box(self.Image_path_list[index][:-4]+".csv",  image_size )
        return img,boxes,label
    def __len__(self):
        return len(self.Image_path_list)


class ConvBn(nn.Module):
    def __init__(self,input_ch,out_ch,stride):
        super().__init__()
        self.Conv = nn.Sequential(

            nn.Conv2d(in_channels=input_ch,out_channels=out_ch,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.Conv(x)
        return x 

class DepthWiseSeparableConv(nn.Module):
    def __init__(self,input_ch,out_ch,stride=1):
        super().__init__()
        # Depthwise convolution
        self.ConvDw = nn.Sequential(
        nn.Conv2d(in_channels=input_ch,out_channels=input_ch,kernel_size=3,stride=stride,padding=1,groups=input_ch),
        nn.BatchNorm2d(num_features=input_ch),
        nn.ReLU()
        )
        # Pointwise convolution
        self.ConvPw = nn.Sequential(

            nn.Conv2d(in_channels=input_ch,out_channels=out_ch,kernel_size=1,stride=1),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.ConvDw(x)
        x = self.ConvPw(x)
        return x 

class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(ConvBn(1,32,1),# 224
                    DepthWiseSeparableConv(32,64,2),# 112
                    DepthWiseSeparableConv(64,128,2),# 56
                    DepthWiseSeparableConv(128,128,1),# 56
                    DepthWiseSeparableConv(128,256,2),# 28
                    DepthWiseSeparableConv(256,256,1), # 28
                    DepthWiseSeparableConv(256,512,2), # 14
                    DepthWiseSeparableConv(512,512,1),# 14
                    DepthWiseSeparableConv(512,256,2),# 7
                    nn.AvgPool2d(7))

       
        self.fc = nn.Linear(256,256)
        self.class_fc = nn.Linear(256,6)
        self.box_fc = nn.Linear(256,4)

    def forward(self,x):
        x = self.layers(x)
        x = x.view(-1,256)
        x = self.fc(x)
        class_x = self.class_fc(x)
        box_x = self.box_fc(x)
        box_x = F.relu(box_x)
        return class_x,box_x

def train(model,train_loader,test_dataset,class_loss_func,boxes_loss_func,epochs=100):

    for epoch in range(epochs):
        total_train_loss,total_test_loss,correct=0,0,0

        model.train()
        
        for x_train , b_train , y_train in train_loader :
            optimizer.zero_grad()
            z_train = model(x_train)
            # class loss
            train_class_loss = class_loss_func(z_train[0],y_train)
            # box loss 
            nan = torch.isnan(b_train)
            B = torch.where(nan, torch.tensor(0.0), b_train)
            BZ = torch.where(nan, torch.tensor(0.0), z_train[1])
            train_box_loss = boxes_loss_func(BZ,B)

            # total train loss
            train_loss = train_box_loss + train_class_loss
            # train_loss = torch.mean(torch.nansum(train_loss,axis=0))

            #backprop
            train_loss.backward()
            optimizer.step()
            
            total_train_loss +=train_loss.item()
            print(" Loss:{}".format(train_loss.item()),end="\n")
            
        model.eval()
        
        val_accuracy = 0
        total_train_loss = total_train_loss/total_batches_for_test
        print(print_format.format(epoch,total_train_loss,total_test_loss,val_accuracy),end="") 


if __name__ == "__main__":
    train_path = "ShapeData/Train"
    test_path = "ShapeData/Test"
    train_dataset = ShapeDataset(train_path)

    test_dataset = ShapeDataset(test_path)
    print("train :",len(train_dataset),"test:",len(test_dataset),"total:",len(train_dataset)+len(test_dataset))
    batch_size = 8
    drop_last = True
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=4, drop_last=drop_last)
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True,num_workers=4, drop_last=drop_last)
    class_loss_func = nn.CrossEntropyLoss()
    boxes_loss_func = nn.MSELoss()
    model = MobileNet()
    optimizer = optim.Adam(model.parameters(),lr = 0.0001)
    print_format = 'Epoch {:03}: | Train Loss: {:.3f} | Test Loss:{:.3f} | val_accuracy: {:.5f}'

    total_batches_for_train = len(train_dataset)/batch_size
    total_batches_for_train = np.floor(total_batches_for_train) if drop_last else np.ceil(total_batches_for_train)

    total_batches_for_test = len(test_dataset)/batch_size
    total_batches_for_test = np.floor(total_batches_for_test) if drop_last else np.ceil(total_batches_for_test)
    trained_model = train(model,train_loader,test_dataset,class_loss_func,boxes_loss_func)