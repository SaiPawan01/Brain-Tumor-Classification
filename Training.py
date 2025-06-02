import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms


import os
from tqdm import tqdm
import matplotlib.pyplot as plt


from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader



# device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# dataset path
root_dir = 'Dataset'
trian_dir = os.path.join(root_dir,'Training')
test_dir =  os.path.join(root_dir,'Testing')


# transform
transform = transforms.Compose([
    transforms.Resize(size=(128,128)),
    transforms.ToTensor()
])



# data loading
train_data = ImageFolder(root=trian_dir,transform=transform,target_transform=None)
test_data = ImageFolder(root=test_dir,transform=transform,target_transform=None)
print("Data Loaded from the directory/...")
classes = test_data.classes
classes2idx = test_data.class_to_idx
idx2classes = {idx : label for label,idx in classes2idx.items()}



# data Loader
train_dataLoader = DataLoader(dataset=train_data,batch_size=16,shuffle=True,num_workers=0)
test_dataLoader =DataLoader(dataset=test_data,batch_size=16,shuffle=True,num_workers=0)
print("Data divided into batch/...")


# model
# model = models.vgg16()

model = models.densenet121(pretrained=True)
num_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(4096, 1024),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(1024, 4),        
    nn.Softmax(dim=1) 
)




# load existing model weights
# model.load_state_dict(torch.load('models/denseNet121_model_1.pth',weights_only=True))
model.to(device)


# loading model existing model
# torch.serialization.add_safe_globals({'VGG': models.VGG})
# model = torch.load('models/vgg16_model_1.pth', weights_only=False)
# model.to(device)


# Training Loop
LOSS = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam(model.classifier.parameters(), lr=0.0001)
EPOCHS = 5


training_loss = []
training_accuracy = [] 


for epoch in range(EPOCHS):
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0



    print(f"Epoch {epoch+1}/{EPOCHS}")

    for images,labels in tqdm(train_dataLoader):

        images = images.to(device)
        labels = labels.to(device)

        # zero gradients
        OPTIMIZER.zero_grad()

        # forward pass
        outputs = model(images)

        # loss
        loss = LOSS(outputs,labels)

        # calculating graidents
        loss.backward()

        # updating gradients
        OPTIMIZER.step()

        # calculating total loss
        total_loss += loss.item()
        

        # accuracy calculation
        _,predicted = torch.max(outputs,1)
        total+=labels.size(0)
        correct += (predicted == labels).sum().item()
        
    

    train_loss = total_loss / len(train_dataLoader)
    accuracy = 100*correct/total

    training_loss.append(train_loss)
    training_accuracy.append(accuracy)


    print(f"Train Loss : {train_loss:.4f} || Train Accuracy : {accuracy:.2f}")


# save the model
torch.save(model.state_dict(), 'models/test_model.pth')


# plot accuracy and loss
plt.plot(training_loss)
plt.show()
plt.plot(training_accuracy)
plt.show()

