"""
Custom Implementation of SimSiam (cf. arXiv:2011.10566v1)
"""
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    MLP Projection Head for SimSiam
    """

    def __init__(self, num_layers: int, input_dim: int, projection_dim: int):
        """

        :param num_layers: number of total layers
        :param input_dim: the input dimension to the projection head
        :param projection_dim: output dimension of the hidden layer
        """
        super(ProjectionHead, self).__init__()

        # list for the layers
        layers = []

        # MLP layers
        for _ in range(num_layers - 1):
            # linear layer, followed by batch normalization and ReLU activation
            layers.append(nn.Linear(input_dim, input_dim, bias=False))  # no bias, because followed by BN
            layers.append(nn.BatchNorm1d(input_dim))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(input_dim, projection_dim, bias=False))  # final projection.
        layers.append(nn.BatchNorm1d(projection_dim, affine=False))  # and normalization

        self.layers = nn.Sequential(*layers)  # make them into a pipeline

    def forward(self, x):
        return self.layers(x)


class Predictor(nn.Module):
    """
    Predictor with intermediate data compression step
    """

    def __init__(self, num_layers: int, projection_dim: int, prediction_dim: int):
        super(Predictor, self).__init__()
        assert num_layers > 1  # need at least 2 layers for the bottleneck-structure described in the paper

        # first three layers sample down (for the described bottleneck structure)
        layers = [nn.Linear(projection_dim, prediction_dim, bias=False), nn.BatchNorm1d(prediction_dim),
                  nn.ReLU(inplace=True)]
        # add more layers, if num_layers (- 1 downsampling layer and -1 upsampling layer) allows for it
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(prediction_dim, prediction_dim, bias=False))
            layers.append(nn.BatchNorm1d(prediction_dim))
            layers.append(nn.ReLU(inplace=True))
        # final upsampling layer, not followed by BN/ReLU/...
        layers.append(nn.Linear(prediction_dim, projection_dim))
        # make pipeline
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SimSiam(nn.Module):
    """
    SimSiam encoder class.
    """

    def __init__(self, encoder: callable, projection_dim: int = 2048, prediction_dim: int = 512,
                 projection_layers: int = 3, prediction_layers: int = 2,
                 cifarresnet: bool = False):
        """
        Create new instance.
        :param encoder: function that will return the encoder. Should either be a normal resnet, or the cifar10 version
        of it, as implemented by Deepmind on GitHub (cf. ResnetCifar.py).
        :param projection_dim: output dimension of the projection mlp.
        :param prediction_dim: output dimension of the hidden, downsampling layer
        :param projection_layers: number of layers in the projection head
        :param prediction_layers: number of layers in the prediction head
        :param cifarresnet: whether we want to work with the CIFAR10 dataset for testing purposes
        """
        super(SimSiam, self).__init__()
        # create encoder
        if not cifarresnet:  # normal training
            # should be default pytorch resnet
            self.encoder = encoder(num_classes=projection_dim, zero_init_residual=True)
        else:
            self.encoder = encoder(low_dim=projection_dim)

        input_dim_resnet = self.encoder.fc.weight.shape[1]  # input of resnet backbone to final fc layer
        self.encoder.fc = nn.Identity(input_dim_resnet)  # deactivate fc layer of resnet --> we're replacing it with
        # mlp head

        # mlp projection head
        self.projection_mlp = ProjectionHead(num_layers=projection_layers, projection_dim=projection_dim,
                                             input_dim=input_dim_resnet)
        # mlp predictor
        self.prediction_mlp = Predictor(num_layers=prediction_layers, projection_dim=projection_dim,
                                        prediction_dim=prediction_dim)

    def forward(self, aug1_batch, aug2_batch):
        z1 = self.encoder(aug1_batch)
        z2 = self.encoder(aug2_batch)
        z1 = self.projection_mlp(z1)
        z2 = self.projection_mlp(z2)
        p1 = self.prediction_mlp(z1)
        p2 = self.prediction_mlp(z2)

        return p1, p2, z1.detach(), z2.detach()
