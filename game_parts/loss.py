from typing import Optional
from PIL import Image

import torch
from torch import nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device=None):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class PretrainedContrastiveLoss(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input=None):
        receiver_output = receiver_output.view_as(sender_input)

        sender_input = self.model(sender_input)
        receiver_output = self.model(receiver_output)

        dot = torch.bmm(receiver_output.unsqueeze(-2), sender_input.unsqueeze(-1))
        dot = dot.mean()

        return -dot, {}


class PerceptualLoss(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.eval()
        self.criterion = nn.MSELoss()

    def forward(self, sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input=None):
        receiver_output = receiver_output.view_as(sender_input)

        receiver_output = self.model(receiver_output)  # learnable
        with torch.no_grad():
            sender_input = self.model(sender_input)  # not learnable

        loss_value = self.criterion(receiver_output, sender_input)
        if loss_value < 0:
            print("negative MSE!")

        return loss_value, {}


class ContinuousAutoencoderLoss(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model  # should be an AutoencoderTrainer instance (from continuous_training.py)
        self.model.eval()
        if self.model.criterion is not None:
            self.criterion = self.model.criterion
        else:
            self.criterion = nn.MSELoss()

    def forward(self, sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input=None):
        receiver_output = receiver_output.view_as(sender_input)

        with torch.no_grad():
            continuous_receiver_output = self.model(sender_input)  # not learnable

        loss_value = self.criterion(receiver_output, continuous_receiver_output)
        if loss_value < 0:
            print("negative MSE!")

        return loss_value, {}


class DiscriminationLoss(nn.Module):
    def __init__(self, discrimination_strategy: str):
        super().__init__()
        assert discrimination_strategy in ['vanilla', 'supervised', 'classification']
        self.discrimination_strategy = discrimination_strategy
        # self.criterion = {'vanilla': nn.CrossEntropyLoss,
        #                   'supervised': nn.CrossEntropyLoss,
        #                   'classification': nn.BCEWithLogitsLoss}[discrimination_strategy]()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, _sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input=None):
        # receiver_output is a Bx(num_distractors + 1) score matrix.
        # Each row corresponds to a message, with the matching image at the first index.
        if self.discrimination_strategy in ['vanilla', 'supervised', 'classification']:
            # every candidate except the target is a negative distractor, so it's regular cross-entropy
            bsz = receiver_output.size(0)
            labels = torch.zeros(bsz, device=receiver_output.device, dtype=torch.long)
            loss_value = self.criterion(receiver_output, labels)
        else:
            raise NotImplementedError()

        return loss_value, {}


class WeightedSumLoss(nn.Module):
    """
    a weighted sum of the other losses. Allows for example MSE with perceptual regularization.
    """
    def __init__(self, *args):
        """
        args are tuples (loss, weight)
        """
        super().__init__()
        assert args
        for tup in args:
            assert isinstance(tup, tuple) and len(tup) == 2, "WeightedSumLoss expects tuples of (loss, weight)"
            assert isinstance(tup[0], nn.Module), f"loss should be a nn.Module, not {type(tup[0])}"
            assert isinstance(tup[1], float), f"weight should be a float, not {type(tup[1])}"
        modules, self.weights = zip(*args)
        self.modules_list = nn.ModuleList(modules)

    def forward(self, *loss_args, **loss_kwargs):
        total = 0.0
        for loss_module, weight in zip(self.modules_list, self.weights):
            loss, _ = loss_module(*loss_args, **loss_kwargs)
            total = total + weight * loss
        return total, {}

    # def to(self, *args, **kwargs):
    #     # print("WeightedSumLoss.to() was called")
    #     for model, _ in self.losses_weights:
    #         model.to(*args, **kwargs)


def get_loss_function(loss_type: str, pretrained_model: nn.Module = None):
    """
    loss_type: ['MSE', 'perceptual', 'contrastive_sim']
    """
    if loss_type == 'MSE':
        assert pretrained_model is None, "mse doesn't require a pretrained encoder"
        pretrained_model = nn.Identity()
    else:
        assert pretrained_model is not None, f"{loss_type} loss requires a pretrained encoder"
    cls = {'MSE': PerceptualLoss,
           'perceptual': PerceptualLoss,
           'contrastive_sim': PretrainedContrastiveLoss,
           'continuous_autoencoder': ContinuousAutoencoderLoss}[loss_type]
    loss = cls(pretrained_model)
    return loss


def get_weighted_loss_function(mse_weight: float = 0.0,
                               perceptual_weight: float = 0.0, perceptual_model: nn.Module = None,
                               contrastive_weight: float = 0.0, contrastive_model: nn.Module = None,
                               autoencoder_weight: float = 0.0, autoencoder_model: nn.Module = None):
    assert mse_weight or perceptual_weight or contrastive_weight or autoencoder_weight
    assert perceptual_weight == 0.0 or perceptual_model is not None
    assert contrastive_weight == 0.0 or contrastive_model is not None
    assert autoencoder_weight == 0.0 or autoencoder_model is not None
    losses = []
    if mse_weight:
        losses.append((get_loss_function("MSE"), mse_weight))
    if perceptual_weight:
        losses.append((get_loss_function("perceptual", perceptual_model), perceptual_weight))
    if contrastive_weight:
        losses.append((get_loss_function("contrastive_sim", contrastive_model), contrastive_weight))
    if autoencoder_weight:
        losses.append((get_loss_function("continuous_autoencoder", autoencoder_model), autoencoder_weight))

    loss = WeightedSumLoss(*losses)
    return loss
