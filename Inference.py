import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

import gradio as gr


device = 'cuda' if torch.cuda.is_available() else "cpu"


model = models.densenet121(pretrained=False)
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


model.load_state_dict(torch.load('models/denseNet121_model_2.pth',weights_only=True,map_location=torch.device(device)))
model.to(device)
model.eval()


# transform
transform = transforms.Compose([
    transforms.Resize(size=(128,128)),
    transforms.ToTensor()
])

# Prediction function
def predict_tumor(image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    class_names = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']
    index = predicted.item()
    if index == 2 :
        return "No Tumaor Detected"
    
    return f"Tumaor Detected : {class_names[predicted.item()]}"


# {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
examples =[['Dataset\Testing\glioma\Te-gl_0010.jpg'],
           ['Dataset\Testing\glioma\Te-gl_0020.jpg'],
           ['Dataset\Testing\glioma\Te-gl_0030.jpg'],
           ['Dataset\Testing\glioma\Te-gl_0040.jpg'],
           ['Dataset\Testing\meningioma\Te-me_0010.jpg'],
           ['Dataset\Testing\meningioma\Te-me_0020.jpg'],
           ['Dataset\Testing\meningioma\Te-me_0030.jpg'],
           ['Dataset\Testing\meningioma\Te-me_0040.jpg'],
           ['Dataset\Testing\\notumor\Te-no_0010.jpg'],
           ['Dataset\Testing\\notumor\Te-no_0020.jpg'],
           ['Dataset\Testing\\notumor\Te-no_0030.jpg'],
           ['Dataset\Testing\\notumor\Te-no_0040.jpg'],
           ['Dataset\Testing\pituitary\Te-pi_0010.jpg'],
           ['Dataset\Testing\pituitary\Te-pi_0020.jpg'],
           ['Dataset\Testing\pituitary\Te-pi_0030.jpg'],
           ['Dataset\Testing\pituitary\Te-pi_0040.jpg']]

# Custom CSS styling
css = """
body {
  background-color: #121212;
  color: #FFFFFF;
}
h1 {
  font-size: 36px;
  text-align: center;
  margin-bottom: 20px;
  color: #00FFAB;
}

footer {
  text-align: center;
  color: #888;
}
"""

# Gradio Interface
app = gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(type="pil", label="Upload Brain Scan"),
    outputs=gr.Label(num_top_classes=4, label="Prediction"),
    title="🧠 Tumor Classification AI",
    description="Upload a brain scan image — this AI model will classify brain tumor.",
    theme="soft",
    css=css,
    examples = examples
)

# Launch the app
app.launch()