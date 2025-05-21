import monai
import torch
import Net.SEResNet as senet
from src.network.model import *
from src.network.monai_model import *
from src.optimizer.Adam import Adam

__all__ = ["get_model", "get_optimizer", "get_criterion", "get_lr_scheduler"]


def get_model(args):
    in_channels = args.in_channels
    out_channels = args.out_channels
    reduction = args.reduction
    feature_channels1 = args.feature_channels
    feature_channels2 = args.new_feature_channels
    hidden_channels = args.hidden_channels
    linear_channels = args.linear_channels
    dropout = args.dropout
    Model = {
        'seres5net': senet.se_resnet(in_channel=in_channels, num_classes=out_channels
                                     , reduction=reduction),
        'seres9net': senet.se_res9net(in_channel=in_channels, num_classes=out_channels
                                      , reduction=reduction),
        'preresnet20': senet.se_preactresnet20(in_channel=in_channels, num_classes=out_channels,
                                               feature_channel=feature_channels2, linear_channel=linear_channels,
                                               reduction=reduction),
    }

    return Model[args.model]


def get_criterion(opt):
    Losses = {
        'CEL': torch.nn.CrossEntropyLoss(label_smoothing=0.1),
    }

    criterion = Losses[opt.loss]

    return criterion


def get_optimizer(opt, model):
    Optimizer = {
        'sgd': torch.optim.SGD,
        'adam': Adam,
        'adamw': torch.optim.AdamW,
        'adamax': torch.optim.Adamax,
        'novograd': monai.optimizers.Novograd
    }
    optimizer = Optimizer[opt.optim](
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay
    )

    return optimizer


def get_lr_scheduler(opt, optimizer):
    Scheduler = {
        'step': torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=opt.lr_decay_period,
            gamma=opt.lr_decay_factor
        ),
        'cos': torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=opt.epochs
        )
    }
    lr_scheduler = Scheduler[opt.scheduler]

    return lr_scheduler
