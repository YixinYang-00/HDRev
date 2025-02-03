from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=800, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=400, help='frequency of showing training results on console')
        
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=50000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        
        # training parameters
        parser.add_argument('--niter', type=int, default=12, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=8, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--lambda_L2', type=float, default=30.0, help='weight for L2 loss')
        parser.add_argument('--lambda_perc', type=float, default=5.0, help='weight for perceptual loss')
        parser.add_argument('--lambda_color', type=float, default=0.001, help='weight for l-LAB loss')
        self.isTrain = True
        return parser
