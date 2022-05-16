import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image

# model = models.vgg19(pretrained=True).features

class VGG(nn.Module):
  def __init__(self) -> None:
      super(VGG,self).__init__()
      self.features = ['0','5','10','19','28']
      self.model = models.vgg19(pretrained=True).features[:29]
  
  def forward(self,x):
    features = []

    for layer_num, layer in enumerate(self.model):
      x = layer(x)
      if str(layer_num) in self.features:
        features.append(x)

    return features
  
def load_image(image_name):
  image = Image.open(image_name)
  image = loader(image).unsqueeze(0)
  return image.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 512

loader = transforms.Compose(
  [
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ]
)

original_img = load_image("test.jpg")
styles_img = load_image("styles.jpg")


model = VGG().to(device).eval()
# generate = torch.randn(original_img.shape,device=device,requires_grad=True)
generated = original_img.clone().requires_grad_(True)

#hyperparameters

total_steps = 6000
learning_rate = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated],lr=learning_rate)

for step in range(total_steps):
  generated_features = model(generated)
  original_img_features = model(original_img)
  styles_features = model(styles_img)

  styles_loss = original_loss = 0
  for gen,org,sty in zip(generated_features,original_img_features,styles_features):
    batch_size,channel,height,width = gen.shape
    original_loss += torch.mean((gen-org)**2)
    #Gram Matrix
    G = gen.view(channel,height*width).mm(gen.view(channel,height*width).t())
    S = sty.view(channel,height*width).mm(sty.view(channel,height*width).t())
    styles_loss += torch.mean((G-S)**2)

  total_loss = alpha*original_loss + beta*styles_loss
  optimizer.zero_grad()
  total_loss.backward()
  optimizer.step()

  if step % 200 == 0:
    print(total_loss)
    save_image(generated, "generated_"+str(step)+".png")