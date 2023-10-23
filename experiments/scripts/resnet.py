import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt

import numpy as np
import time
import os

from torchvision import transforms
from PIL import Image

import shap

torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image_path',
                    type=str,
                    required=True)

parser.add_argument('-s', '--state_dict_path',
                    type=str,
                    required=True)

parser.add_argument('-d', '--dataset',
                    help="Options: 'afad', 'morph2', or 'cacd'.",
                    type=str,
                    required=True)

parser.add_argument('--enable_shap', help='Enable SHAP explanation',
                    action='store_true')

args = parser.parse_args()
IMAGE_PATH = args.image_path
STATE_DICT_PATH = args.state_dict_path
GRAYSCALE = False

##############################################################################
folder_names = ["grads", "shuffled_grads", "pruning_experiments"] # Name of the folder to save the gradients
folder_name = folder_names[-1] # Change this to change the folder name

timestamp = int(time.time())
CURRENT_LAYER = 0
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
elif "spaced" in folder_name and os.path.exists(folder_name):
    for file in os.listdir(folder_name):
        if file.endswith(".npy"):
            os.remove(os.path.join(folder_name, file))

if folder_name == "pruning_experiments":
    all_folder_names = os.listdir(folder_name)
    if len(all_folder_names) == 0:
        os.makedirs(os.path.join(folder_name, "0"))
        folder_name = os.path.join(folder_name, "0")
    else:
        last_folder = 0
        for folder in all_folder_names:
            if int(folder) > last_folder:
                last_folder = int(folder)
        if IMAGE_PATH.endswith("Image_1.jpg"):
            last_folder += 1
        if not os.path.exists(os.path.join(folder_name, str(last_folder))):
            os.makedirs(os.path.join(folder_name, str(last_folder)))
        folder_name = os.path.join(folder_name, str(last_folder))

##############################################################################

if args.dataset == 'afad':
    NUM_CLASSES = 26
    ADD_CLASS = 15

elif args.dataset == 'morph2': # BU DATASET KULLANILIYOR
    NUM_CLASSES = 55
    ADD_CLASS = 16

elif args.dataset == 'cacd':
    NUM_CLASSES = 49
    ADD_CLASS = 14

else:
    raise ValueError("args.dataset must be 'afad',"
                     " 'morph2', or 'cacd'. Got %s " % (args.dataset))

image = Image.open(IMAGE_PATH)
custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.CenterCrop((120, 120)),
                                       transforms.ToTensor()])
image = custom_transform(image)
DEVICE = torch.device('cpu')
image = image.to(DEVICE)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def save_grad(pre_logits, post_logits, layer_name):
    if folder_name == "spaced_grads_original" or folder_name == "spaced_grads":
        np.save(f'{folder_name}/pre_logits_{layer_name}.npy', pre_logits)
        np.save(f'{folder_name}/post_logits_{layer_name}.npy', post_logits)
        return

    #np.save(f'{folder_name}/pre_logits_{timestamp}_{layer_name}.npy', pre_logits)
    np.save(f'{folder_name}/post_logits_{timestamp}_{layer_name}.npy', post_logits)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        global CURRENT_LAYER
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        pre_logits = out
        out = self.relu(out)
        post_logits = out
        CURRENT_LAYER += 1
        save_grad(pre_logits, post_logits, f"layer{CURRENT_LAYER}")

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        pre_logits = out
        out = self.relu(out)
        post_logits = out
        CURRENT_LAYER += 1
        save_grad(pre_logits, post_logits, f"layer{CURRENT_LAYER}")

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        global CURRENT_LAYER
        x = self.conv1(x)
        x = self.bn1(x)
        pre_logits = x
        x = self.relu(x)
        post_logits = x
        # so we need to create a new folder
        CURRENT_LAYER += 1
        save_grad(pre_logits, post_logits, f"layer{CURRENT_LAYER}")
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_logits = x
        logits = self.fc(x)
        post_logits = logits
        CURRENT_LAYER += 1
        save_grad(pre_logits, post_logits, f"layer{CURRENT_LAYER}")
        probas = F.softmax(logits, dim=1)

        if args.enable_shap:
            return logits
        return logits, probas


def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock,
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model


model = resnet34(NUM_CLASSES, GRAYSCALE)
model.load_state_dict(torch.load(STATE_DICT_PATH, map_location=DEVICE))
model.eval()

image = image.unsqueeze(0)

if args.enable_shap:
    background_data = torch.zeros((1, 3, 120, 120))
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(image)

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    print(f"Shape of shap_numpy: {np.shape(shap_numpy)}")
    image_numpy = image.cpu().numpy()

    most_probable_class = torch.argmax(model(image), 1).item()
    print(f"Most probable class: {most_probable_class}")
    if most_probable_class < len(shap_numpy):
        shap_numpy = np.squeeze(shap_numpy)
        most_probable_class_shap = shap_numpy[most_probable_class]
    else:
        print(f"Index {most_probable_class} is out of bounds for shap_numpy with length {len(shap_numpy)}")

    if most_probable_class_shap is not None:
        min_val = np.min(most_probable_class_shap)
        max_val = np.max(most_probable_class_shap)
        most_probable_class_shap = (most_probable_class_shap - min_val) / (max_val - min_val)

        original_image = np.transpose(image_numpy[0], (1, 2, 0))

        overlay = np.uint8(255 * (0.2 * original_image + 0.8 * most_probable_class_shap))

        plt.figure()

        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.axis("off")
        plt.imshow(np.uint8(255 * original_image))

        plt.subplot(1, 3, 2)
        plt.title("SHAP Heatmap")
        plt.axis("off")
        plt.imshow(np.uint8(255 * most_probable_class_shap))

        plt.subplot(1, 3, 3)
        plt.title("Overlay")
        plt.axis("off")
        plt.imshow(overlay)

        plt.suptitle(f"Predicted Age: {most_probable_class + ADD_CLASS} | True Age: 32")

        plt.savefig("spaced_grads/shap_overlay_original.png")
else:
    with torch.set_grad_enabled(False):
        logits, probas = model(image)
        #print(torch.argmax(probas, 1).item() + ADD_CLASS)
