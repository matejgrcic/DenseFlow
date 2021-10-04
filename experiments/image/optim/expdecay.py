import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, OneCycleLR
from denseflow.optim.schedulers import LinearWarmupScheduler

optim_choices = {'sgd', 'adam', 'adamax' , 'adamw'}


def add_optim_args(parser):

    # Model params
    parser.add_argument('--optimizer', type=str, default='adamw', choices=optim_choices)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_lr', type=float, default=3e-3)
    parser.add_argument('--data_samples', type=int, required=True)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--freeze', type=int, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--momentum_sqr', type=float, default=0.99)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--use_grad_norm', type=float, default=0.)
    parser.add_argument('--pct_start', type=float, default=0.3)



def get_optim_id(args):
    return 'expdecay'


def get_optim(args, model):
    assert args.optimizer in optim_choices

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, nesterov=True, weight_decay=10**(-5), momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, args.momentum_sqr))
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, betas=(args.momentum, args.momentum_sqr))
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.momentum, args.momentum_sqr), weight_decay=args.weight_decay)

    bs = args.batch_size
    iters = args.epochs * (int(args.data_samples * 1. / bs) + 1)

    # if args.warmup is not None:
    #     scheduler_warmup = LinearWarmupScheduler(optimizer, total_epoch=warmup_iters)
    # else:
    #     scheduler_warmup = None

    # scheduler_epoch = ExponentialLR(optimizer, gamma=args.gamma)
    # scheduler_epoch = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup, eta_min=5e-6)

    scheduler_iter = OneCycleLR(optimizer, max_lr=args.max_lr, total_steps=iters, base_momentum=0.85, max_momentum=0.95, pct_start=args.pct_start, anneal_strategy='linear')
    scheduler_epoch = None

    return optimizer, scheduler_iter, scheduler_epoch
