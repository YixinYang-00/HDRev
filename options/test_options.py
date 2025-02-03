from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=5000, help='how many test images to run')
        
        parser.add_argument('--test_on_txt', action='store_true', help='whether test on txt data, default is test on voxel data')
        # rewrite devalue values
        parser.set_defaults(phase='test')
        self.isTrain = False
        return parser
