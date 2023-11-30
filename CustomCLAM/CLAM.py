import torch.nn as nn
import torch
import sys
import numpy as np
import Attention

sys.path.insert(0, '/beegfs/home/s/schumany/smooth-topk')
import topk


class CLAM(nn.Module):
    """
    CLAM-Model. In contrast to the original CLAM implementation in
    https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py
    we have a joint class for single-branch and multi-branch CLAM
    """

    def __init__(self, encoded_dim: int = 1024, input_dim_attention: int = 512, projection_dim_attention: int = 256,
                 dropout: bool = False, n_classes: int = 2, n_inst: int = 50, subtyping: bool = False,
                 multi_branch: bool = True):
        """
        Create new instance
        :type multi_branch: whether to use the multi-branch version of CLAM
        :param encoded_dim: the output dimension of the enncoder
        :param input_dim_attention: dimension after data-compression
        :param projection_dim_attention: dimension of data after u or v path in attention network
        :param dropout: whether to use dropout for regularization
        :param n_classes: number of classes in dataset
        :param n_inst: number of instances to use for instance-level clustering
        :param subtyping: whether the classes are mutually exclusive (cannot be present in the same slide)
        """

        super(CLAM, self).__init__()
        # store parameters as fields
        self.multi_branch = multi_branch
        self.encoded_dim = encoded_dim
        self.input_dim_attention = input_dim_attention
        self.projection_dim_attention = projection_dim_attention
        self.dropout = dropout
        self.n_classes = n_classes
        self.n_inst = n_inst
        self.subtyping = subtyping
        self.instance_loss = topk.svm.SmoothTop1SVM(n_classes=self.n_classes)

        #  attention backbone
        if self.multi_branch:
            self.attention_backbone = Attention.GatedAttentionBackbone(encoded_dim=self.encoded_dim,
                                                                       input_dim_attention=self.input_dim_attention,
                                                                       projection_dim_attention=self.projection_dim_attention,
                                                                       num_classes=self.n_classes, dropout=self.dropout)
        else:
            self.attention_backbone = Attention.GatedAttentionBackbone(encoded_dim=self.encoded_dim,
                                                                       input_dim_attention=self.input_dim_attention,
                                                                       projection_dim_attention=self.projection_dim_attention,
                                                                       num_classes=1, dropout=self.dropout)

        #  classifiers for bags --> each predicts a score per class
        if self.multi_branch:
            self.bag_classifiers = nn.ModuleList(
                [nn.Linear(self.input_dim_attention, 1) for _ in range(self.n_classes)])
        else:
            self.bag_classifiers = nn.ModuleList(
                [nn.Linear(self.input_dim_attention, self.n_classes)])
        # instance classifiers --> each one distinguishes strongly attended / weakly attended patches
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(self.input_dim_attention, 2) for _ in range(self.n_classes)])
        # initialize weights
        self.__initialize_weight__()

    def __initialize_weight__(self):
        """
        Initializes weights as in the original CLAM implementation
        https://github.com/mahmoodlab/CLAM/blob/master/utils/utils.py
        (Function taken from there, all credit goes to the original authors)
        :return: nothing
        """
        # iterate over module
        for m in self.modules():
            # xavier
            if isinstance(m, nn.Linear):
                # xavier normal distribution, as described in
                # Glorot, X. & Bengio, Y. (2010)
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                # there are no batchnorm1d functions in this architecture
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __evaluate_inclass__(self, x_attention: torch.Tensor, x: torch.Tensor, classifier: nn.Module):
        """
        In-Class attention-branch. Loss for positive and for negative instances (highly attended and weakly
        attended instances)
        :param x_attention: attention for the individual tiles, n_classes x n_instances
        :param x: compressed data, n_inst x input_dim_attention
        :param classifier: the instance-level classifier for the respective class
        :return: the instance-loss, the predicted labels (weakly attended / highly attended) and the respective targets.
        """
        if len(x_attention.shape) == 1:
            x_attention = x_attention.view(1, -1)
        # indices of the most highly attended instances
        top_p_ids = torch.topk(x_attention, self.n_inst)[1].squeeze()
        # corresponding instances (aka positive instances)
        top_p = torch.index_select(x, dim=0, index=top_p_ids)
        # most weakly attended indices
        top_n_ids = torch.topk(-x_attention, self.n_inst, dim=1)[1].squeeze()
        # corresponding indices (aka negative instances)
        top_n = torch.index_select(x, dim=0, index=top_n_ids)
        # positive targets
        p_targets = torch.full((self.n_inst,), 1, device=x.device)
        # negative targets
        n_targets = torch.full((self.n_inst,), 0, device=x.device)
        # concatenate targets and instances
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        # raw scores
        logits = classifier(all_instances)
        # hard-thresholded predictions
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze()
        # compute instance loss
        instance_loss = self.instance_loss(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def __evaluate_outofclass__(self, x_attention: torch.Tensor, x: torch.Tensor, classifier: nn.Module):
        """
        Out of class evaluation
        :param x_attention: attention for the individual tiles, n_classes x n_instances
        :param x: compressed data, n_inst x input_dim_attention
        :param classifier: the instance-level classifier for the respective class
        :return: the instance-loss, the predicted labels (weakly attended / highly attended) and the respective targets.
        """
        if len(x_attention.shape) == 1:
            x_attention = x_attention.view(1, -1)
        # indices with the highest attention
        top_p_ids = torch.topk(x_attention, self.n_inst)[1].squeeze()
        # corresponding tiles
        top_p = torch.index_select(x, dim=0, index=top_p_ids)
        # they should all be negative (aka weakly attended for this other class)
        p_targets = torch.full((self.n_inst,), 0, device=x.device)
        # raw scores
        logits = classifier(top_p)
        # hard-thresholded class-prediction
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze()
        # compute corresponding instance loss
        instance_loss = self.instance_loss(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def __eval_instances__(self, x_attention: torch.Tensor, x: torch.Tensor, label: torch.Tensor):
        """
        Compute instance-level loss
        :param x_attention: attention for the individual tiles, n_classes x n_instances
        :param x: compressed data, n_inst x input_dim_attention
        :param label: the class-label for this bag
        :return: dictionary with the individual predictions
        """
        total_instance_loss = 0.0  # instance level loss (accumulated)

        predictions = []  # instance-level predictions (strongly/weakly attended)
        targets = []  # instance-level targets

        # all zeros,except for the index corresponding to the label
        one_hot_label = nn.functional.one_hot(label, num_classes=self.n_classes)[0]
        # iterate over the instance-level classifiers
        for class_idx, classifier in enumerate(self.instance_classifiers):
            label = one_hot_label[class_idx].item()  # label for the respective instance classifier
            if label == 1:  # in the class
                if self.multi_branch:
                    instance_loss, instance_predictions, instance_targets = self.__evaluate_inclass__(
                        x_attention[class_idx], x, classifier)
                else:
                    instance_loss, instance_predictions, instance_targets = self.__evaluate_inclass__(
                        x_attention, x, classifier)
                # append to lists
                predictions.extend(instance_predictions.cpu().numpy())
                targets.extend(instance_targets.cpu().numpy())
            elif self.subtyping:  # out of the class, if subtyping is active (classes are mutually exclusive)
                if self.multi_branch:
                    instance_loss, instance_predictions, instance_targets = self.__evaluate_outofclass__(
                        x_attention[class_idx], x, classifier)
                else:
                    instance_loss, instance_predictions, instance_targets = self.__evaluate_outofclass__(
                        x_attention, x, classifier)
                # append to lists
                predictions.extend(instance_predictions.cpu().numpy())
                targets.extend(instance_targets.cpu().numpy())
            else:  # if the classes are not mutually exclusive
                instance_loss = 0.0
            total_instance_loss += instance_loss

        if self.subtyping:
            total_instance_loss /= len(self.instance_classifiers)  # average the instance loss

        # make results dataframe
        results = {"instance_loss": total_instance_loss, "inst_labels": np.array(targets),
                   "inst_predictions": predictions}

        return results

    def forward(self, x: torch.Tensor, label: torch.Tensor, instance_eval: bool = True, attention_only: bool = False):
        # the compressed bag n_inst x input_dim_attention and the corresponding attention scores
        # n_inst x n_classes
        x_compressed, x_attention = self.attention_backbone(x)
        x_attention = torch.transpose(x_attention, 1, 0)  # n_classes x n_instances
        # if we want to make attention plots, we only need the respective, unnormalized attention scores
        if attention_only:
            return x_attention
        raw_attention = torch.clone(x_attention)  # a copy
        x_attention = nn.functional.softmax(x_attention, 1)  # softmax normalization per class

        # dictionary for the results
        results = {}

        if instance_eval:  # if clustering is not manually disables
            results = self.__eval_instances__(x_attention, x_compressed, label)  # will overwrite the dictionary
        # create bag-level representation (should have size n_classes x input_dim_attention)
        bag_level_representations = torch.mm(x_attention, x_compressed)
        if self.multi_branch:
            # for the raw scores
            logits = torch.empty(1, self.n_classes).float().to(x.device)
            # fill the raw scores for each class
            for class_idx, classifier in enumerate(self.bag_classifiers):
                logits[0, class_idx] = classifier(bag_level_representations[class_idx])
        else:
            logits = self.bag_classifiers[0](bag_level_representations)
        # get class with highest scores
        y_class = torch.topk(logits, 1, dim=1)[1]
        # and make probability estimate by applying softmax normalization
        y_proba = nn.functional.softmax(logits, dim=1)
        # append the bag-level results
        results.update({"logits": logits, "y_class": y_class, "y_proba": y_proba, "raw_attention": raw_attention})
        return results
