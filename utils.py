import torch
import torch.nn as nn
import sys
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

# ===============================================================================================================================
#   Define some print statements with colour. 

TX_PURPLE                       = '\033[95m'
TX_CYAN                         = '\033[96m'
TX_DARKCYAN                     = '\033[36m'
TX_BLUE                         = '\033[94m'
TX_GREEN                        = '\033[92m'
TX_YELLOW                       = '\033[93m'
TX_RED                          = '\033[91m'
TX_BOLD                         = '\033[1m'
TX_UNDERLINE                    = '\033[4m'
TX_END                          = '\033[0m'

#   Define functions to use combinations of these. 
def cfm(text, colour=None, fmt=None):
    colour_dict                 = {
        'p'                     :   TX_PURPLE,
        'c'                     :   TX_CYAN,
        'n'                     :   TX_DARKCYAN,
        'b'                     :   TX_BLUE,
        'g'                     :   TX_GREEN,
        'y'                     :   TX_YELLOW,
        'r'                     :   TX_RED,
        'purple'                :   TX_PURPLE,
        'cyan'                  :   TX_CYAN,
        'darkcyan'              :   TX_DARKCYAN,
        'blue'                  :   TX_BLUE,
        'green'                 :   TX_GREEN,
        'yellow'                :   TX_YELLOW,
        'red'                   :   TX_RED,
    }
    fmt_dict                    = {
        'b'                     :   TX_BOLD,
        'u'                     :   TX_UNDERLINE,
        'e'                     :   TX_END,
        'bold'                  :   TX_BOLD,
        'underline'             :   TX_UNDERLINE,
        'end'                   :   TX_END,
    }

    return_string               = ''
    flag                        = False

    if fmt is not None:
        return_string          += fmt_dict[fmt]
        flag                    = True
    if colour is not None:
        return_string          += colour_dict[colour]
        flag                    = True

    return_string              += text 
    if flag:
        return_string          += fmt_dict['e']

    return return_string
# ===============================================================================================================================


#   Hyper parameters that need to be handled for backward compatibility.
#   This is a list of parameters that will be assigned to options if they
#       do not already exist, and will be given some default values. 
_BKD_CMPTBL_PARAMS                          = {
    'model'                             : {
        'r_lse'                         : 10,
        'rho'                           : 0.002,
        'use_encoder'                   : False,
        'use_decoder'                   : False,
        'predictor_type'                : 'full_conv',
        'out_relu'                      : False,
        'pretrained'                    : False,
        'projection'                    : False,
        'ch_exp'                        : 'exponential',
        'attention_arch'                : False,
        'gating_network'                : 'softmax',
        'separate_gate_fex'             : False,
    },
    'training'                          : {
        'freeze_cnn'                    : False,
        'freeze_mlp'                    : False,
        'load_which'                    : 'latest',
        'load_cnn'                      : False,
        'load_mlp'                      : False,

        'optim_alg'                     : 'sgd',
        'scale_recon'                   : 1e-3,
        'scale_imgs'                    : 1.,
        'scale_attrs'                   : 1.,
        'scale_sparse'                  : 1.,
        'neg_train_until'               : 10,
        'scale_0'                       : 1., #98.0 / (44 + 98),
        'scale_1'                       : 1., #44.0 / (44 + 98),
        'autoencoder_only_epochs'       : 0,
        'load_mlp_from'                 : False,
        'load_cnn_from'                 : False,

        'scale_decoder'                 : 1.,
        'scale_cnn'                     : 1.,
        'scale_mlp'                     : 1.,
        'scale_gate'                    : 1.,
        'scale_classifier'              : 1.,

        'n_iters'                       : 200000,
        'n_epochs'                      : 1000,

        'lr_decay_scheme'               : 'step',
        'recon_fn'                      : 'mse',

        'test'                          : False,
        'skip_train'                    : False,
    },
    'dataset'                           : {
        'mask_root'                     : False,
        'c_transform'                   : 'None',
    }
}
#   Function to fix backward compatibility.
#   Based on _BKD_CMPTBL_PARAMS
def fix_backward_compatibility(options, params=_BKD_CMPTBL_PARAMS):
    for _hyper_param in params:
        if isinstance(params[_hyper_param], dict):
            setattr(
                options, 
                _hyper_param, 
                fix_backward_compatibility(
                        getattr(options, _hyper_param),
                        params[_hyper_param]
                ),
            )
        elif not hasattr(options, _hyper_param):
            setattr(options, _hyper_param, params[_hyper_param])
    return options
# ===============================================================================================================================


def write_flush(text, stream=sys.stdout):
    stream.write(text)
    stream.flush()
# ===============================================================================================================================
    
    
def parse_input(images):
    images = images.float()/255.0
    return images
# ===============================================================================================================================


def weights_init(m, nonlinearity='relu'):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
# ===============================================================================================================================


def weights_init2(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
# ===============================================================================================================================


def get_year(t):
    if '-' in t:
        return int(t.split('-')[-1])
    elif '/' in t:
        return int(t.split('/')[-1])
# ===============================================================================================================================


def cross_correlation_loss_2(source, target, n=9):
    avg_pooler                  = lambda x: F.avg_pool2d(x, n, stride=1, 
                                                        padding=(n-1)//2, 
                                                        ceil_mode=False, 
                                                        count_include_pad=False)

    s_hat                       = avg_pooler(source)
    t_hat                       = avg_pooler(target)

    sdiff                       = source - s_hat
    tdiff                       = target - t_hat

    # Compute the numerator
    numerator                   = (sdiff * tdiff).sum() ** 2

    # Compute the denominator
    d0                          = (sdiff ** 2).sum()
    d1                          = (tdiff ** 2).sum()
    denominator                 = (d0 * d1)

    cc_loss                     = numerator / denominator
    # A higher value for cc_loss indicates better alignment. Use negative cc_loss instead. 
    return -1 * cc_loss.mean()
# ===============================================================================================================================


def cross_correlation_loss(source, target, n=9, eps=1e-5):
    w_size                      = n * n 

    sum_pooler                  = lambda x: w_size * F.avg_pool2d(x, n, stride=1, 
                                                        padding=(n-1)//2, 
                                                        ceil_mode=False, 
                                                        count_include_pad=False)

    S2                          = source * source
    T2                          = target * target
    ST                          = source * target

    S_sum                       = sum_pooler(source)
    T_sum                       = sum_pooler(target)
    S2_sum                      = sum_pooler(S2)
    T2_sum                      = sum_pooler(T2)
    ST_sum                      = sum_pooler(ST) 

    S_avg                       = 1./ w_size * S_sum
    T_avg                       = 1./ w_size * T_sum

    cross                       = ST_sum - T_avg * S_sum
    S_var                       = S2_sum - S_avg * S_sum
    T_var                       = T2_sum - T_avg * T_sum

    cc_loss                     = (cross * cross) / (S_var * T_var + eps)

    return -1 * cc_loss.sum(dim=-1).sum(dim=-1).sum(dim=-1).mean()
# ===============================================================================================================================


class RandomRotation(object):
    """
    A randomly chosen rotation is applied to a PIL image.
    """
    def __init__(self, angles_list):
        self.angles_list    = angles_list
    def __call__(self, img):
        A                   = np.random.choice(self.angles_list)
        return img.rotate(A)
# ======================================================================================================

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


# ===============================================================================
#   Define geometric transformation losses. 
#   Tensors here are defined as B x C x H x W
#

def AugTransform_flipX(T):
    # Flip the images in tensor T along X.
    return torch.flip(T, (3,))

def AugTransform_flipY(T):
    # Flip the images in tensor T along Y.
    return torch.flip(T, (2,))

def AugTransform_transpose(T):
    # Transpose the image. 
    return T.permute(0,1,3,2)

def AugTransform_rot90(T):
    # Rotate the image 90 degrees clockwise. 
    return AugTransform_transpose(AugTransform_flipX(T))

def AugTransform_rot180(T):
    # Rotate the image 180 degrees. 
    return AugTransform_flipX(AugTransform_flipY(T))

def AugTransform_rot270(T):
    # Rotate the image 270 degrees. 
    return AugTransform_transpose(AugTransform_flipY(T))


def AugTransform_D2(T):
    # Downsample the image by a factor of 2. 
    return F.interpolate(T, scale_factor=0.5, mode='bilinear', align_corners=False)

def AugTransform_D4(T):
    # Downsample the image by a factor of 4. 
    return F.interpolate(T, scale_factor=0.25, mode='bilinear', align_corners=False)

def AugTransform_U2(T):
    # Upsample the image by a factor of 2. 
    return F.interpolate(T, scale_factor=2, mode='bilinear', align_corners=False)

def AugTransform_U4(T):
    # Upsample the image by a factor of 4. 
    return F.interpolate(T, scale_factor=4, mode='bilinear', align_corners=False)


AUG_TRANSFORMS_DICT     = {
            'ID'            : {
                                'forward'   :   lambda x: x,
                                'backward'  :   lambda x: x,
                              },
            'FX'            : {
                                'forward'   :   AugTransform_flipX,
                                'backward'  :   AugTransform_flipX,
                              },
            
            'FY'            : {
                                'forward'   :   AugTransform_flipY,
                                'backward'  :   AugTransform_flipY,
                              },

            'TR'            : {
                                'forward'   :   AugTransform_transpose,
                                'backward'  :   AugTransform_transpose,
                              },
            
            'R90'           : {
                                'forward'   :   AugTransform_rot90,
                                'backward'  :   AugTransform_rot270,
                              },

            'R180'          : {
                                'forward'   :   AugTransform_rot180,
                                'backward'  :   AugTransform_rot180,
                              },

            'R270'          : {
                                'forward'   :   AugTransform_rot270,
                                'backward'  :   AugTransform_rot90,
                              },

            'D2'            : {
                                'forward'   :   AugTransform_D2,
                                'backward'  :   AugTransform_U2,
                              },

            'D4'            : {
                                'forward'   :   AugTransform_D4,
                                'backward'  :   AugTransform_U4,
                              },

            'U2'            : {
                                'forward'   :   AugTransform_U2,
                                'backward'  :   AugTransform_D2,
                              },

            'U4'            : {
                                'forward'   :   AugTransform_U4,
                                'backward'  :   AugTransform_D4,
                              },
}


