import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.sum()


class BCERescaleLoss(nn.Module):
    def __init__(self, cfg):
        super(BCERescaleLoss, self).__init__()
        self.min_iou = cfg.min_iou
        self.max_iou = cfg.max_iou
        self.bias = cfg.bias
    def forward(self, scores, masks, targets):
        joint_prob = torch.sigmoid(scores) * masks
        target_prob = (targets-self.min_iou)*(1-self.bias)/(self.max_iou-self.min_iou)
        target_prob[target_prob > 0] += self.bias
        target_prob[target_prob > 1] = 1
        target_prob[target_prob < 0] = 0
        loss = F.binary_cross_entropy(joint_prob, target_prob, reduction='none') * masks
        loss_value = torch.sum(loss) / torch.sum(masks)
        return loss_value, joint_prob


def build_loss_func(cfg):
    cfg = cfg.TRAINING.LOSS
    loss_func = globals()[cfg.loss_name](cfg)
    return loss_func


if __name__ == '__main__':
    from easydict import EasyDict as edict
    cfg = edict({'loss_name':'MILNCELoss', 'temp':0.05})
    print(cfg.loss_name)
    loss_func = build_loss_func(cfg)
    print(loss_func.temp)
    video_embd = torch.randn(64,1024)
    text_embd = torch.randn(1280,1024)
    print(loss_func(video_embd, text_embd))

