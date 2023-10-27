import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms , datasets, models
from torch.utils.data import DataLoader

model = models.resnet18(weights='IMAGENET1K_V1' )
numf = model.fc.in_features
for parameters in model.parameters():
    parameters.requires_grad = False
model.fc = nn.Sequential(nn.Linear(512,5),nn.Softmax())
data_transformation =  transforms.Compose([transforms.Resize(size=(150,150)), transforms.ToTensor()])
data_directory = '/Users/chiddu/Documents/E-yrc/Input_data'
input_image_dataset = datasets.ImageFolder(data_directory, transform= data_transformation)
input_dataloader = DataLoader(input_image_dataset,batch_size=10, shuffle= True)

data_directory_output = '/Users/chiddu/Documents/E-yrc/Output_data'
output_image_dataset = datasets.ImageFolder(data_directory_output, transform= data_transformation)
output_dataloader = DataLoader(output_image_dataset,batch_size=1, shuffle= True)

lossfunction = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=0.01)
num_of_epochs = 5
#print(input_image_dataset.classes)
#print(input_image_dataset[0])



def training_the_model(model, input_dataloader,lossfunction , optimiser):
    for i in range(num_of_epochs) :
        for image, labels in input_dataloader:
            outputs= model(image)
            labell = torch.zeros(10,5)
            for i in range(10):
                labell[i][labels[i]]=1
            #print(outputs)
            #print(labell)
            #print(labels)
            #print(outputs)
            #outputs = torch.argmax(outputs,1) 
            loss= lossfunction(outputs,labell)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return model
def testing_the_model(model,output_dataloader ):
    model.eval()
    correct = 0
    total = 0
    for image,labels in output_dataloader:
        outputs = model(image)
        outputs = torch.argmax(outputs,1) #outputs will become tensor with only one value, which is the index of max value out of 5 values
        if outputs==labels :
            correct = correct+1
        total= total+1
    print((correct/total)*100)
    accuracy = (correct/total)*100
    return accuracy


if __name__ == "__main__":
    model= training_the_model(model,input_dataloader,lossfunction,optimiser)
    testing_the_model(model,output_dataloader)
    torch.save(model,'/Users/chiddu/Documents/E-yrc/task2B/trainedmodel.pth')



        
        
            
            





    




    
