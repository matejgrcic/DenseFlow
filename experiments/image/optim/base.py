import torch.optim as optim

optim_choices = {'sgd', 'adam', 'adamax', 'adamw'}


def add_optim_args(parser):

    # Model params
    parser.add_argument('--optimizer', type=str, default='adamw', choices=optim_choices)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--momentum_sqr', type=float, default=0.99)


def get_optim_id(args):
    return 'base'


def get_optim(args, model):
    assert args.optimizer in optim_choices

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, args.momentum_sqr))
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, betas=(args.momentum, args.momentum_sqr))
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.momentum, args.momentum_sqr), weight_decay=0.01)

    scheduler_iter = None
    scheduler_epoch = None

    return optimizer, scheduler_iter, scheduler_epoch
