from .dense_flow import DenseFlow
def add_model_args(parser):

    # Flow params
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--growth_rate', type=int, default=None)
    parser.add_argument('--checkpointing', action='store_true', default=False)
    parser.add_argument('--block_conf', nargs='+', type=int)
    parser.add_argument('--layer_mid_chnls', nargs='+', type=int)
    parser.add_argument('--layers_conf', nargs='+', type=int)


def get_model_id(args):
    return 'densenet-flow'


def get_model(args, data_shape):

    return DenseFlow(
        data_shape=data_shape, block_config=args.block_conf, layers_config=args.layers_conf,
        layer_mid_chnls=args.layer_mid_chnls, growth_rate=args.growth_rate, checkpointing=args.checkpointing)