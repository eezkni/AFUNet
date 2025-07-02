import torch
from utils.utils import *
import torchvision.models as models

class VGG19_relu(torch.nn.Module):
    def __init__(self, device='cuda'):
        super(VGG19_relu, self).__init__()
        cnn = getattr(models, 'vgg19')(pretrained=True)
        cnn = cnn.to(device)
        features = cnn.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, loss_map):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        if loss_map==None:
            out = {
                'relu1_1': relu1_1,
                'relu1_2': relu1_2,

                'relu2_1': relu2_1,
                'relu2_2': relu2_2,

                'relu3_1': relu3_1,
                'relu3_2': relu3_2,
                'relu3_3': relu3_3,
                'relu3_4': relu3_4,

                'relu4_1': relu4_1,
                'relu4_2': relu4_2,
                'relu4_3': relu4_3,
                'relu4_4': relu4_4,

                'relu5_1': relu5_1,
                'relu5_2': relu5_2,
                'relu5_3': relu5_3,
                'relu5_4': relu5_4,
            }
        else:
            loss_map_1 = loss_map.repeat(1, relu1_1.shape[1], 1, 1)
            loss_map_2 = torch.nn.functional.interpolate(loss_map, size=(relu2_1.shape[2], relu2_1.shape[3]), mode='bilinear', align_corners=True).repeat(1, relu2_1.shape[1], 1, 1)
            loss_map_3 = torch.nn.functional.interpolate(loss_map, size=(relu3_1.shape[2], relu3_1.shape[3]), mode='bilinear', align_corners=True).repeat(1, relu3_1.shape[1], 1, 1)
            loss_map_4 = torch.nn.functional.interpolate(loss_map, size=(relu4_1.shape[2], relu4_1.shape[3]), mode='bilinear', align_corners=True).repeat(1, relu4_1.shape[1], 1, 1)
            loss_map_5 = torch.nn.functional.interpolate(loss_map, size=(relu5_1.shape[2], relu5_1.shape[3]), mode='bilinear', align_corners=True).repeat(1, relu5_1.shape[1], 1, 1)

            out = {
                'relu1_1': relu1_1 * loss_map_1,
                'relu1_2': relu1_2 * loss_map_1,

                'relu2_1': relu2_1 * loss_map_2,
                'relu2_2': relu2_2 * loss_map_2,

                'relu3_1': relu3_1 * loss_map_3,
                'relu3_2': relu3_2 * loss_map_3,
                'relu3_3': relu3_3 * loss_map_3,
                'relu3_4': relu3_4 * loss_map_3,

                'relu4_1': relu4_1 * loss_map_4,
                'relu4_2': relu4_2 * loss_map_4,
                'relu4_3': relu4_3 * loss_map_4,
                'relu4_4': relu4_4 * loss_map_4,

                'relu5_1': relu5_1 * loss_map_5,
                'relu5_2': relu5_2 * loss_map_5,
                'relu5_3': relu5_3 * loss_map_5,
                'relu5_4': relu5_4 * loss_map_5,
            }
        return out