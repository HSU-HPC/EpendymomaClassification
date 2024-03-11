import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, \
    ResNet152_Weights

import CustomSimSiam


def build_model(args):
    if args.num == 18:
        encoder = models.resnet18
    elif args.num == 34:
        encoder = models.resnet34
    elif args.num == 50:
        encoder = models.resnet50
    elif args.num == 101:
        encoder = models.resnet101
    elif args.num == 152:
        encoder = models.resnet152
    else:
        raise ValueError("Unknown architecture")
    #
    model = CustomSimSiam.SimSiam(encoder=encoder, projection_dim=args.dim, prediction_dim=args.pred_dim,
                                  projection_layers=args.pjl, prediction_layers=args.pdl,
                                  cifarresnet=False)

    return model


def build_pretrained_resnet(args):
    if args.num == 18:
        encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    elif args.num == 34:
        encoder = models.resnet34(weights=ResNet34_Weights.DEFAULT)
    elif args.num == 50:
        encoder = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    elif args.num == 101:
        encoder = models.resnet101(weights=ResNet101_Weights.DEFAULT)
    elif args.num == 152:
        encoder = models.resnet152(weights=ResNet152_Weights.DEFAULT)
    else:
        raise ValueError("Unknown dmension for pretrained resnet encoder")

    return encoder


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.model = None
        self.typ = args.encodertyp
        self.outputsize = -1
        self.lastavg = args.lastavg

        if args.encodertyp == "simsiam":
            self.model = build_model(args)
            self.usemlp = args.withmlp
            if self.usemlp:  # will have output size of MLP
                self.outputsize = args.dim
            else:
                # dim. of output after last global average pooling layer
                dim_dict = {18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048}
                self.outputsize = dim_dict[args.num]
        elif args.encodertyp == "resnet":
            self.model = build_pretrained_resnet(args)
            if not self.lastavg:
                self.outputsize = list(self.model.layer3[-1].modules())[-2].weight.shape[0]
            else:
                self.model.fc = nn.Identity()
                dim_dict = {18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048}
                self.outputsize = dim_dict[args.num]
        else:
            raise ValueError("Only simsiam and resnet are allowed encoder architectures.")

    def forward(self, x):
        self.model.eval()
        if self.typ == "simsiam":
            # this is all called encoder (backbone + projection head) in the original simsiam paper
            x = self.model.module.encoder(x)
            if self.usemlp:
                x = self.model.module.projection_mlp(x)
        else:
            if not self.lastavg:
                x = self.model.module.conv1(x)
                x = self.model.module.bn1(x)
                x = self.model.module.relu(x)
                x = self.model.module.maxpool(x)
                x = self.model.module.layer1(x)
                x = self.model.module.layer2(x)
                x = self.model.module.layer3(x)
                x = self.model.module.avgpool(x)
                x = x.view(x.size()[0], -1)
            else:
                x = self.model.module(x)
        return x
