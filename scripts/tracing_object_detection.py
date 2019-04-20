import torch
import torchvision
from torchvision import transforms
from PIL import Image
from time import time
import numpy as np

# load jit module
traced_script_module = torch.jit.load('mb1-ssd.pt')

# read image
image = Image.open('lane_control.jpg').convert('RGB')
default_transform = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
      ])
image = default_transform(image)

# forward
boxes, labels, probs = traced_script_module(image.unsqueeze(0))
#print(output[0, :10])

# print top-5 predicted labels
#labels = np.loadtxt('synset_words.txt', dtype=str, delimiter='\n')

#data_out = output[0].data.numpy()
#sorted_idxs = np.argsort(-data_out)

#for i,idx in enumerate(sorted_idxs[:5]):
#  print('top-%d label: %s, score: %f' % (i, labels[idx], data_out[idx]))
