import torch.nn as nn


class GatedAttentionBackbone(nn.Module):
    """
    Attention Backbone as described in the original CLAM publication
    Predicts N distinct sets of attention scores corresponding to N classes (in the multi-class classification
    problem discussed by CLAM)
    --> learns per class(!) positive and negative evidence for the respective class --> later enables the computation
    of N slide-level representations
    """

    def __init__(self, encoded_dim=1024, input_dim_attention=512, projection_dim_attention=256, num_classes=2,
                 dropout=False):
        """
        Generates instance of attention backbone (with sigmoid gating and 3 fc)
        :param encoded_dim: input dimensionality to the backbone (before data compression)
        :param input_dim_attention: input dimensionality
        :param projection_dim_attention: intermediate dimension within attention backbone
        :param num_classes: number of classes in the attention backbone
        :param dropout: whether to use dropout for regularization (p=0.25 is fixed)
        """
        super(GatedAttentionBackbone, self).__init__()
        self.encoded_dim = encoded_dim
        self.input_dim_attention = input_dim_attention
        self.projection_dim_attention = projection_dim_attention
        self.num_classes = num_classes
        self.dropout = dropout

        # compression step --> shared by all classes
        self.w1 = [nn.Linear(self.encoded_dim, self.input_dim_attention), nn.ReLU()]
        # first layer of attention backbone, using sigmoid gating here
        self.u_path = [nn.Linear(self.input_dim_attention, self.projection_dim_attention), nn.Sigmoid()]
        # 2nd layer, also using sigmoid gating
        self.v_path = [nn.Linear(self.input_dim_attention, self.projection_dim_attention), nn.Tanh()]

        # adding dropout layers after compression step and after each of the paths
        if self.dropout:
            self.w1.append(nn.Dropout(0.25))
            self.u_path.append(nn.Dropout(0.25))
            self.v_path.append(nn.Dropout(0.25))

        self.w1 = nn.Sequential(*self.w1)
        self.u_path = nn.Sequential(*self.u_path)
        self.v_path = nn.Sequential(*self.v_path)

        # multi-class attention branches
        self.mc_ab = nn.Linear(self.projection_dim_attention, num_classes)

    def forward(self, x):
        x = self.w1(x)  # compressed data, required for instance level loss
        u_rep = self.u_path(x)  # 1st layer
        v_rep = self.v_path(x)  # 2nd layer
        # return compressed data and the elementwise attention (which has not been subject to softmax fct yet)
        return x, self.mc_ab(u_rep * v_rep)
