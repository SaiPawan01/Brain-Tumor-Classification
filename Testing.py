import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import confusion_matrix

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# device agnoistic code 
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# training loop
def testing_loop(model:nn.Module,test_dataLoader:DataLoader,loss_fn:nn.Module):
    
    # move model to the device
    model = model.to(device)
    model.eval()

    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    Actual_values = []
    Predicted_Values = []
    
    with torch.no_grad():

        batch = 0
        for images,labels in tqdm(test_dataLoader):

            # move to the device
            images = images.to(device)
            labels = labels.to(device)
            Actual_values.extend(labels.cpu().numpy())

            # make predictions
            predictions = model(images)

            # calculate loss
            loss = loss_fn(predictions,labels).item()
            total_loss += loss
            
            # calculate accuracy
            _,predicted = torch.max(predictions,1)
            Predicted_Values.extend(labels.cpu().numpy())

            
            total_predictions += labels.size(0)
            correct_batch_predictions = (predicted == labels).sum().item()
            correct_predictions += correct_batch_predictions

            # batch loss and accuracy
            # batch_accuracy = 100*correct_batch_predictions/16
            batch+=1
            # print(f"Batch : {batch} || Loss : {loss} || Accuracy : {batch_accuracy}")
        

        # total loss & accuracy
        total_loss = total_loss/len(test_dataLoader)
        total_accuracy = 100*correct_predictions/total_predictions

        print(f"Loss : {loss} || Accuracy : {total_accuracy}")



        return (Actual_values,Predicted_Values)



# data directory 
root_dir = 'Dataset'
test_dir = os.path.join(root_dir,'Testing')


# walk throught directories
print("===========Directory Structure=========")
for dirpath,dirnames,filename in os.walk(test_dir):
        print(f"there are {len(dirnames)} directories and {len(filename)} images in '{dirpath}")


# transform
transform = transforms.Compose([
    transforms.Resize(size=(128,128)),
    transforms.ToTensor()
])


# test data
test_data = ImageFolder(root=test_dir,transform=transform,target_transform=None)
classes = test_data.classes
classes2idx = test_data.class_to_idx
idx2classes = {idx:label for label,idx in classes2idx.items()}

# {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}


# test DataLoader 
test_dataLoader = DataLoader(dataset=test_data,batch_size=16,shuffle=True,num_workers=0)


# loss function
loss_fn = nn.CrossEntropyLoss()



# loading model
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

# torch.serialization.add_safe_globals({'VGG': models.VGG})
# model = torch.load('models/vgg16_model_1.pth', weights_only=False)

model.load_state_dict(torch.load('models/denseNet121_model_2.pth',weights_only=True,map_location=torch.device(device)))
model.to(device)

# function call
Actual_values,Predicted_Values = testing_loop(model,test_dataLoader,loss_fn)


# creating dataframe
Actual_Predicted_df = pd.DataFrame({"Actual" : Actual_values,
                                    "Predicted_Values" : Predicted_Values})


# confusion matrix
confusion_matrix = confusion_matrix(Actual_values,Predicted_Values)
sns.heatmap(confusion_matrix,annot=True,xticklabels=classes, yticklabels=classes)
plt.show()