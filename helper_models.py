"""
Python script to define neural nets used.
"""

# ===============================================================================================================================
#   Imports

import  torch
from    torch                           import  nn
import  torch.nn.functional             as      F
import  torch.utils.data
from    torchvision                     import  models
from    torchvision                     import  transforms
from    torchvision.models.resnet       import  BasicBlock, Bottleneck
from    torchvision.models.squeezenet   import  squeezenet1_0, squeezenet1_1
from    torch.autograd                  import  Variable
import  os
import  sys
from    scipy.misc                      import  imread, imsave
import  pickle
import  numpy                           as      np
from    utils                           import  *
from    attr_dict                       import  *
from    PIL                             import  Image
from    contextlib                      import  ExitStack
from    cbam                            import  *

__torch_version__   = int(torch.__version__.split('.')[1])


# ===============================================================================================================================
#   Get python version. 
if sys.version.split(' ')[0][0] == '2':
    __python_version__  = 2
elif sys.version.split(' ')[0][0] == '3':
    __python_version__  = 3


# ===============================================================================================================================
#   Global variables. 

"""
Custom function to print options in a formatted manner. 
"""
def print_config(options, prefix=''):
    for key in options:
        if isinstance(options[key], AttrDict):
            write_flush(prefix+'{}:\n' %(key))
            print_config(options[key], prefix+'\t')
        else:
            write_flush(prefix+'{}: {}\n'.format(key, options[key]))
    return 



# System mode: specified what to use to train. 
#   I   : Images
#   A   : Attributes
#   G   : Gate
#   D   : Encoder-decoder
#   S   : Simclr training. Contrastive training. 
# Note: system mode 'IA' forces equal contribution from both images and attributes. 
IMGS_TR                                     = 'I'
ATTR_TR                                     = 'A'
AUTOENC_TR                                  = 'D'
GATE_TR                                     = 'G'
GATE_MSE_TR                                 = 'E'
GATE_LOSS_TR                                = 'T'
GATE_MOG_TR                                 = 'M'
ATTN_TR                                     = 'N'
SEPARATE_GATE_FEX_TR                        = 'S'
CONCAT_ATTR_TR                              = 'C'
SIMCLR_TR                                   = 'S'

IMGS_ATTR_TR                                = IMGS_TR + ATTR_TR
IMGS_SIMCLR_TR                              = IMGS_TR + SIMCLR_TR
IMGS_AUTOENC_TR                             = IMGS_TR + AUTOENC_TR
IMGS_AUTOENC_ATTR_TR                        = IMGS_TR + AUTOENC_TR + ATTR_TR
IMGS_ATTR_GATE_TR                           = IMGS_TR + ATTR_TR + GATE_TR
IMGS_ATTR_GATE_MSE_TR                       = IMGS_TR + ATTR_TR + GATE_MSE_TR
IMGS_ATTR_GATE_MOG_TR                       = IMGS_TR + ATTR_TR + GATE_MOG_TR
IMGS_ATTR_GATE_GATELOSS_TR                  = IMGS_TR + ATTR_TR + GATE_TR + GATE_LOSS_TR
IMGS_ATTN_TR                                = IMGS_TR + ATTN_TR
IMGS_CONCAT_ATTR_TR                         = IMGS_TR + CONCAT_ATTR_TR
SYSTEM_MODES                                = [
                                                IMGS_TR, 
                                                ATTR_TR, 
                                                IMGS_SIMCLR_TR,
                                                IMGS_ATTR_TR,
                                                IMGS_ATTR_GATE_TR,
                                                IMGS_ATTR_GATE_MSE_TR,
                                                IMGS_ATTR_GATE_MOG_TR,
                                                IMGS_AUTOENC_TR,
                                                IMGS_AUTOENC_ATTR_TR,
                                                IMGS_ATTR_GATE_GATELOSS_TR,
                                                IMGS_ATTN_TR,
                                                IMGS_CONCAT_ATTR_TR,
                                              ]

# Loss functions. 
BCE                                         = 'bce'
NLL                                         = 'nll'
IMPLEMENTED_LOSSES                          = [BCE, NLL]


# ===============================================================================================================================
#   Convolution functions
def conv5x5(in_planes, out_planes, stride=1, bias=False):
    '''5x5 convolution'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=bias)

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    '''3x3 convolution'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    '''1x1 convolution'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


# ===============================================================================================================================
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape      = shape

    def forward(self, x):
        return x.view(*self.shape)

# ===============================================================================================================================
#   Resnet functions and variables. 

resnet_inits            = {
        10  :       lambda flag: models.ResNet(BasicBlock, [1, 1, 1, 1]),
        14  :       lambda flag: models.ResNet(BasicBlock, [1, 1, 2, 2]),
        18  :       lambda flag: models.resnet18(pretrained=flag),
        34  :       lambda flag: models.resnet34(pretrained=flag),
        50  :       lambda flag: models.resnet50(pretrained=flag),
        101 :       lambda flag: models.resnet101(pretrained=flag),
        152 :       lambda flag: models.resnet152(pretrained=flag),
}

narrow_resnet_inits     = {
        10  :       lambda ndf, ch_exp: NarrowResNet(BasicBlock, [1, 1, 1, 1], ndf, ch_exp),
        14  :       lambda ndf, ch_exp: NarrowResNet(BasicBlock, [1, 1, 2, 2], ndf, ch_exp),
        18  :       lambda ndf, ch_exp: NarrowResNet(BasicBlock, [2, 2, 2, 2], ndf, ch_exp),
        34  :       lambda ndf, ch_exp: NarrowResNet(BasicBlock, [3, 4, 6, 3], ndf, ch_exp),
        50  :       lambda ndf, ch_exp: NarrowResNet(Bottleneck, [3, 4, 6, 3], ndf, ch_exp),
        101 :       lambda ndf, ch_exp: NarrowResNet(Bottleneck, [3, 4, 23, 3], ndf, ch_exp),
        152 :       lambda ndf, ch_exp: NarrowResNet(Bottleneck, [3, 8, 36, 3], ndf, ch_exp),
}

narrow_resnet_in_inits  = {
        10  :       lambda ndf, ch_exp: NarrowResNetInstanceNorm(BasicBlockInstanceNorm, [1, 1, 1, 1], ndf, ch_exp),
        14  :       lambda ndf, ch_exp: NarrowResNetInstanceNorm(BasicBlockInstanceNorm, [1, 1, 2, 2], ndf, ch_exp),
        18  :       lambda ndf, ch_exp: NarrowResNetInstanceNorm(BasicBlockInstanceNorm, [2, 2, 2, 2], ndf, ch_exp),
        34  :       lambda ndf, ch_exp: NarrowResNetInstanceNorm(BasicBlockInstanceNorm, [3, 4, 6, 3], ndf, ch_exp),
        50  :       lambda ndf, ch_exp: NarrowResNetInstanceNorm(BottleneckInstanceNorm, [3, 4, 6, 3], ndf, ch_exp),
        101 :       lambda ndf, ch_exp: NarrowResNetInstanceNorm(BottleneckInstanceNorm, [3, 4, 23, 3], ndf, ch_exp),
        152 :       lambda ndf, ch_exp: NarrowResNetInstanceNorm(BottleneckInstanceNorm, [3, 8, 36, 3], ndf, ch_exp),
}


assert(resnet_inits.keys() == narrow_resnet_inits.keys())
possible_resnets                        = resnet_inits.keys()
resnet_outsize                          = {
        10  :       512, 
        14  :       512, 
        18  :       512, 
        34  :       512, 
        50  :       2048, 
        101 :       2048, 
        152 :       2048,
}


class LeCunSigmoid(nn.Module):
    def __init__(self):
        super(LeCunSigmoid, self).__init__()

    def forward(self, x):
        return 1.7159 * torch.tanh(2.0 * x / 3.0)

class BasicBlockInstanceNorm(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockInstanceNorm, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




# ===============================================================================================================================
class DilatedConvBlock(nn.Module):
    def __init__(self, in_c=3, out_c=32, **kwargs):
        super(DilatedConvBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_c, out_c, **kwargs),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.main(x)

class DilatedConvResBlock(nn.Module):
    def __init__(self, in_c=3, out_c=32, **kwargs):
        super(DilatedConvResBlock, self).__init__()

        self.res    = nn.Sequential(
            nn.Conv2d(in_c, out_c, **kwargs),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )
        self.relu   = nn.ReLU(True)

    def forward(self, x):
        residual        = self.res(x)
        out             = self.relu(x + residual)
        return out

class DilatedConv2ResBlock(nn.Module):
    def __init__(self, in_c=3, out_c=32, **kwargs):
        super(DilatedConv2ResBlock, self).__init__()

        self.in_c   = in_c
        self.out_c  = out_c
        self.kwargs = kwargs

        self.res    = nn.Sequential(
            nn.Conv2d(in_c, out_c, **kwargs),
            nn.ReLU(True),

            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, dilation=1),
        )

        if out_c != in_c or kwargs['stride'] > 1:
            self.mapper = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=1, stride=kwargs['stride'], padding=0, dilation=1),
            )
        else:
            self.mapper = None

        self.relu   = nn.ReLU(True)

    def forward(self, x, **kwargs):
        residual        = self.res(x)
        if self.mapper:
            x           = self.mapper(x)
        

        out             = self.relu(x + residual)
        return out

class DilatedConv2BottleneckResBlockInstanceNorm(nn.Module):
    def __init__(self, in_c=3, out_c=32, **kwargs):
        super(DilatedConv2BottleneckResBlockInstanceNorm, self).__init__()

        self.in_c   = in_c
        self.out_c  = out_c
        self.squeeze_planes = in_c // 2
        sq_c = self.squeeze_planes

        self.res    = nn.Sequential(
            nn.Conv2d(in_c, sq_c, kernel_size=1, padding=0, dilation=1, stride=1),
            nn.InstanceNorm2d(sq_c),
            nn.ReLU(True),

            nn.Conv2d(sq_c, sq_c, **kwargs),
            nn.InstanceNorm2d(sq_c),
            nn.ReLU(True),

            nn.Conv2d(sq_c, out_c, kernel_size=1, padding=0, dilation=1, stride=1),
            nn.InstanceNorm2d(out_c),
        )

        if out_c != in_c or kwargs['stride'] > 1:
            self.mapper = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=1, stride=kwargs['stride'], padding=0),
                    nn.InstanceNorm2d(out_c),
            )
        else:
            self.mapper = None

        self.relu   = nn.ReLU(True)

    def forward(self, x, **kwargs):
        residual        = self.res(x)
        if self.mapper:
            x           = self.mapper(x)

        out             = self.relu(x + residual)
        return out




# ===============================================================================================================================


class NarrowResNet(nn.Module):
    def __init__(self, block, layers, ndf, ch_exp):
        super(NarrowResNet, self).__init__()
        self.ndf                        = ndf
        self.conv1                      = nn.Conv2d(3, ndf, kernel_size=7, stride=2, 
                                                padding=3, bias=False)
        self.bn1                        = nn.BatchNorm2d(ndf)
        self.relu                       = nn.ReLU(inplace=True)
        self.ch_exp                     = ch_exp

        if self.ch_exp == 'exponential':
            n_ch                    = [ndf*(2**i) for i in range(4)] 
        elif self.ch_exp == 'linear':
            n_ch                    = [ndf * (i + 1) for i in range(4)]

        self.maxpool                    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1                     = self._make_layer(block, n_ch[0], layers[0])
        self.layer2                     = self._make_layer(block, n_ch[1], layers[1], stride=2)
        self.layer3                     = self._make_layer(block, n_ch[2], layers[2], stride=2)
        self.layer4                     = self._make_layer(block, n_ch[3], layers[3], stride=2)

        self.avgpool                    = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            weights_init(m, nonlinearity='relu')
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample                      = None
        if stride != 1 or self.ndf != planes*block.expansion:
            downsample                  = nn.Sequential(
                    conv1x1(self.ndf, planes * block.expansion, stride), 
                    nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers                          = []
        layers.append(block(self.ndf, planes, stride, downsample))
        self.ndf                        = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.ndf, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x                               = self.conv1(x)
        x                               = self.bn1(x)
        x                               = self.relu(x)
        x                               = self.maxpool(x)
        
        x                               = self.layer1(x)
        x                               = self.layer2(x)
        x                               = self.layer3(x)
        x                               = self.layer4(x)

        x                               = self.avgpool(x)
        x                               = x.view(x.size(0), -1)
        return x

class NarrowResNetInstanceNorm(nn.Module):
    def __init__(self, block, layers, ndf, ch_exp):
        super(NarrowResNetInstanceNorm, self).__init__()
        self.ndf                        = ndf
        self.conv1                      = nn.Conv2d(3, ndf, kernel_size=7, stride=2, 
                                                padding=3, bias=False)
        self.bn1                        = nn.InstanceNorm2d(ndf)
        self.relu                       = nn.ReLU(inplace=True)
        self.ch_exp                     = ch_exp

        if self.ch_exp == 'exponential':
            n_ch                    = [ndf*(2**i) for i in range(4)] 
        elif self.ch_exp == 'linear':
            n_ch                    = [ndf * (i + 1) for i in range(4)]

        self.maxpool                    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1                     = self._make_layer(block, n_ch[0], layers[0])
        self.layer2                     = self._make_layer(block, n_ch[1], layers[1], stride=2)
        self.layer3                     = self._make_layer(block, n_ch[2], layers[2], stride=2)
        self.layer4                     = self._make_layer(block, n_ch[3], layers[3], stride=2)

        self.avgpool                    = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            weights_init(m, nonlinearity='relu')
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample                      = None
        if stride != 1 or self.ndf != planes*block.expansion:
            downsample                  = nn.Sequential(
                    conv1x1(self.ndf, planes * block.expansion, stride), 
                    nn.InstanceNorm2d(planes * block.expansion),
            )
        
        layers                          = []
        layers.append(block(self.ndf, planes, stride, downsample))
        self.ndf                        = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.ndf, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x                               = self.conv1(x)
        x                               = self.bn1(x)
        x                               = self.relu(x)
        x                               = self.maxpool(x)
        
        x                               = self.layer1(x)
        x                               = self.layer2(x)
        x                               = self.layer3(x)
        x                               = self.layer4(x)

        x                               = self.avgpool(x)
        x                               = x.view(x.size(0), -1)
        return x




#   Unified ResNet --- handles all types. 
class ResNet(nn.Module):
    def __init__(self, options):
        super(ResNet, self).__init__()
        arch                            = options.model.arch
        ndf                             = options.model.ndf
        pretrained                      = options.model.pretrain
        ch_exp                          = options.model.ch_exp
        self.get_scores                 = options.model.score
        self.n_classes                  = 1 if options.training.loss == BCE else 2
        self.mix                        = options.model.mix
        self.freeze_cnn                 = options.training.freeze_cnn

        self.latent                     = AUTOENC_TR in options.model.system_mode or not options.model.score

#resnet_n_layers                 = int(arch.split('_')[-1])     # changed to the next line after i added favgpool on 2020-10-09
        resnet_n_layers                 = int(arch.split('_')[1])
        if arch.startswith('resnet'):
            self.model                  = resnet_inits[resnet_n_layers](pretrained)
        elif arch.startswith('narrow-resnet'):
            self.model                  = narrow_resnet_inits[resnet_n_layers](ndf, ch_exp)
        elif arch.startswith('in-narrow-resnet'):
            self.model                  = narrow_resnet_in_inits[resnet_n_layers](ndf, ch_exp)

        expansion                       = 1 if resnet_n_layers in [10, 14, 18, 34] else 4
        if arch.startswith('narrow-resnet'):
            self.out_chs                = 8*expansion*options.model.ndf
        else:
            self.out_chs                = resnet_outsize[resnet_n_layers]

        if 'favgpool' in arch:
            self.out_shape              = 1
            self.out_vec_shape          = self.out_shape * self.out_shape * self.out_chs
            self.avg_pool               = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.out_shape              = options.training.img_size//32
            self.out_vec_shape          = self.out_shape*self.out_shape*self.out_chs
            self.avg_pool               = None


        self.projection                 = options.model.projection
        if self.projection:
            self.projection_head        = nn.Sequential(
                                            nn.Linear(self.out_chs, self.out_chs),
                                            nn.ReLU(True),
                                            nn.Linear(self.out_chs, self.out_chs),
                                          )

        if self.get_scores:
            self.agg_linear             = nn.Conv2d(self.out_chs, self.n_classes, self.out_shape, stride=1, padding=0)
#            self.agg_linear             = nn.Sequential(
#                    nn.Dropout(p=0.5),
#                    nn.Conv2d(self.out_chs, self.n_classes, kernel_size=1),
#                    nn.ReLU(inplace=True),
#                    nn.AdaptiveAvgPool2d((1, 1)),
#            )

        if self.mix:
            self.mixer                  = nn.Linear(self.out_vec_shape, 1)
            self.mixer.weight.data.normal_(0.0, 0.02/self.out_vec_shape)
            self.mixer.bias.data.fill_(0.0)

        if not pretrained:
            for m in self.modules():
                weights_init(m, nonlinearity='relu')
            
    def forward(self, x):
        with ExitStack() as stack:
            if self.freeze_cnn:
                stack.enter_context(torch.no_grad())
                self.model.eval()

            x                           = self.model.conv1(x)
            x                           = self.model.bn1(x)
            x                           = self.model.relu(x)
            x                           = self.model.maxpool(x)

            x                           = self.model.layer1(x)
            x                           = self.model.layer2(x)
            x                           = self.model.layer3(x)
            x                           = self.model.layer4(x)

        if self.avg_pool:
            x                           = self.avg_pool(x)

        return_dict                     = {}

        if self.mix:
            mscore                      = self.mixer(x.view(-1, self.out_vec_shape).contiguous())
            return_dict['mix']          = mscore

        if self.projection:
            return_dict['proj']         = self.projection_head(x)

        if self.get_scores:
#            ag                          = self.model.avgpool(x).view(-1, self.out_chs).contiguous()
            ag                          = self.agg_linear(x).squeeze(3).squeeze(2)
            return_dict['score']        = ag

        return_dict['latent']           = x

        return return_dict


class TinyNetV2ReLUNoBN(nn.Module):
    def __init__(self, options):
        super(TinyNetV2ReLUNoBN, self).__init__()

        ndf                     = options.model.ndf
        self.freeze_cnn         = options.training.freeze_cnn
        self.ndf                = ndf
        nc                      = options.model.nc
        self.n_classes          = 1 if options.training.loss == BCE else 2
        self.out_chs            = 2 * ndf
        self.out_shape          = options.training.img_size//32
        self.out_vec_shape      = self.out_shape*self.out_shape*self.out_chs


        self.conv_block         = nn.Sequential(
                # input is B x nc x 224 x 224
                nn.Conv2d(nc, ndf * 4, kernel_size=7, padding=3, stride=2),
                nn.ReLU(True),
                nn.MaxPool2d(2),

                # state size B x ndf * 4 x 56 x 56
                nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, padding=1, stride=1),
                nn.ReLU(True),
                nn.MaxPool2d(2),

                # state size B x ndf * 4 x 28 x 28
                nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, padding=1, stride=1),
                nn.ReLU(True),

                # state size B x ndf * 8 x 28 x 28
                nn.Conv2d(ndf * 8, ndf * 8, kernel_size=3, padding=1, stride=2),
                nn.ReLU(True),

                # state size B x ndf * 8 x 14 x 14
                nn.Conv2d(ndf * 8, ndf * 16, kernel_size=3, padding=1, stride=2),
                nn.ReLU(True),

                # state size B x ndf * 16 x 7 x 7
                nn.Conv2d(ndf * 16, self.out_chs, kernel_size=1, padding=0, stride=1),
                nn.ReLU(True),
        )

    def forward(self, x, **kwargs):
        return_dict             = {}

        y           = self.conv_block(x)
        return_dict['latent'] = y
        return return_dict

class TinyNetResidualReLUNoBN(nn.Module):
    def __init__(self, options):
        super(TinyNetResidualReLUNoBN, self).__init__()

        ndf                     = options.model.ndf
        self.freeze_cnn         = options.training.freeze_cnn
        self.ndf                = ndf
        nc                      = options.model.nc
        self.n_classes          = 1 if options.training.loss == BCE else 2
        self.out_chs            = 2 * ndf
        self.out_shape          = options.training.img_size//32
        self.out_vec_shape      = self.out_shape*self.out_shape*self.out_chs


        self.conv_block         = nn.Sequential(
                # input is B x nc x 224 x 224
                DilatedConv2ResBlock(in_c=nc, out_c=ndf * 4, kernel_size=7, padding=3, stride=2, dilation=1),
                nn.MaxPool2d(2),

                # state size B x ndf * 4 x 56 x 56
                DilatedConv2ResBlock(in_c=ndf * 4, out_c=ndf * 4, kernel_size=3, padding=1, stride=1, dilation=1),
                nn.MaxPool2d(2),

                # state size B x ndf * 4 x 28 x 28
                DilatedConv2ResBlock(in_c=ndf * 4, out_c=ndf * 8, kernel_size=3, padding=1, stride=1, dilation=1),

                # state size B x ndf * 8 x 28 x 28
                DilatedConv2ResBlock(in_c=ndf * 8, out_c=ndf * 8, kernel_size=3, padding=1, stride=2, dilation=1),

                # state size B x ndf * 8 x 14 x 14
                DilatedConv2ResBlock(in_c=ndf * 8, out_c=ndf * 16, kernel_size=3, padding=1, stride=2, dilation=1),

                # state size B x ndf * 16 x 7 x 7
                DilatedConv2ResBlock(in_c=ndf * 16, out_c=self.out_chs, kernel_size=1, padding=0, stride=1, dilation=1),
        )

    def forward(self, x, **kwargs):
        return_dict             = {}

        y           = self.conv_block(x)
        return_dict['latent'] = y
        return return_dict


class TinyNetResidualV2ReLUNoBN(nn.Module):
    def __init__(self, options):
        super(TinyNetResidualV2ReLUNoBN, self).__init__()

        ndf                     = options.model.ndf
        self.freeze_cnn         = options.training.freeze_cnn
        self.ndf                = ndf
        nc                      = options.model.nc
        self.n_classes          = 1 if options.training.loss == BCE else 2
        self.out_chs            = 2 * ndf
        self.out_shape          = options.training.img_size//32
        self.out_vec_shape      = self.out_shape*self.out_shape*self.out_chs


        self.conv_block         = nn.Sequential(
                # input is B x nc x 224 x 224
                nn.Conv2d(in_c=nc, out_c=ndf * 4, kernel_size=7, padding=3, stride=2, dilation=1),
                nn.ReLU(True),
                nn.MaxPool2d(2),

                # state size B x ndf * 4 x 56 x 56
                DilatedConv2ResBlock(in_c=ndf * 4, out_c=ndf * 4, kernel_size=3, padding=1, stride=1, dilation=1),
                nn.MaxPool2d(2),

                # state size B x ndf * 4 x 28 x 28
                DilatedConv2ResBlock(in_c=ndf * 4, out_c=ndf * 8, kernel_size=3, padding=1, stride=1, dilation=1),

                # state size B x ndf * 8 x 28 x 28
                DilatedConv2ResBlock(in_c=ndf * 8, out_c=ndf * 8, kernel_size=3, padding=1, stride=2, dilation=1),

                # state size B x ndf * 8 x 14 x 14
                DilatedConv2ResBlock(in_c=ndf * 8, out_c=ndf * 16, kernel_size=3, padding=1, stride=2, dilation=1),

                # state size B x ndf * 16 x 7 x 7
                DilatedConv2ResBlock(in_c=ndf * 16, out_c=self.out_chs, kernel_size=1, padding=0, stride=1, dilation=1),
        )

    def forward(self, x, **kwargs):
        return_dict             = {}

        y           = self.conv_block(x)
        return_dict['latent'] = y
        return return_dict
 
 

                
#   Unified SqueezeNet --- handles all types. 
class SqueezeNet(nn.Module):
    def __init__(self, options):
        super(SqueezeNet, self).__init__()

        arch                            = options.model.arch
        ndf                             = options.model.ndf
        pretrained                      = options.model.pretrain
        self.get_scores                 = options.model.score
        self.n_classes                  = 1 if options.training.loss == BCE else 2
        self.mix                        = options.model.mix
        self.img_size                   = options.training.img_size

        self.latent                     = True
        
        version                         = arch.split('_')[-1]
        if version not in ['1.0', '1.1']:
            raise ValueError('Invalid version %s specified for SqueezeNet. Must be 1.0 or 1.1' %(version))

        if version == '1.0':
            self.model                  = squeezenet1_0(pretrained=pretrained)
        elif version == '1.1':
            self.model                  = squeezenet1_1(pretrained=pretrained)
        self.model.classifier           = None

        self.out_chs                    = 512
        self.out_shape                  = self.img_size // 16 - 1

        if self.get_scores:
            self.agg_linear             = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Conv2d(512, self.n_classes, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
            )

        if not pretrained:
            for m in self.modules():
                weights_init(m, nonlinearity='relu')

    def forward(self, x):
        return_dict                     = {}

        x                               = self.model.features(x)
        
        return_dict['latent']           = x
        if self.get_scores:
            y                           = self.agg_linear(x)
            return_dict['score']        = y.view(y.size(0), -1).contiguous()

        return return_dict



# ===============================================================================================================================
#   My ResNet. Fewer parameters, but with residual connections. 

class ResidualBlockStride(nn.Module):
    def __init__(self, in_planes, out_planes, downsample=False, out_relu=False):
        super(ResidualBlockStride, self).__init__()

        self.in_planes      = in_planes
        self.out_planes     = out_planes
        self.downsample     = downsample
        self.out_relu       = out_relu
        self.stride         = 2 if self.downsample else 1

        self.conv1          = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn1            = nn.BatchNorm2d(out_planes)
        self.relu           = nn.ReLU(True)
        self.conv2          = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2            = nn.BatchNorm2d(out_planes)

        if self.stride > 1:
            self.downsample = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=2, padding=0, bias=False)

    def forward(self, x):
        identity            = x

        residual            = self.conv1(x)
        residual            = self.bn1(residual)
        residual            = self.relu(residual)

        residual            = self.conv2(residual)
        residual            = self.bn2(residual)

        if self.stride > 1:
            identity        = self.downsample(x)

        out                 = residual + identity

        if self.out_relu:
            out             = self.relu(out)

        return out

class ResidualBlockPool(nn.Module):
    def __init__(self, in_planes, out_planes, downsample=False, out_relu=False):
        super(ResidualBlockPool, self).__init__()

        self.in_planes      = in_planes
        self.out_planes     = out_planes
        self.downsample     = downsample
        self.out_relu       = out_relu
        self.stride         = 2 if self.downsample else 1

        self.conv1          = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1            = nn.BatchNorm2d(out_planes)
        self.relu           = nn.ReLU(True)
        self.conv2          = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2            = nn.BatchNorm2d(out_planes)

        if self.stride > 1:
            self.max_pool   = nn.MaxPool2d(self.stride)
            self.downsample = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=2, padding=0, bias=False)

    def forward(self, x):
        identity            = x

        residual            = self.conv1(x)
        residual            = self.bn1(residual)
        residual            = self.relu(residual)
        if self.stride > 1:
            residual        = self.max_pool(residual)

        residual            = self.conv2(residual)
        residual            = self.bn2(residual)

        if self.stride > 1:
            identity        = self.downsample(x)

        out                 = residual + identity

        if self.out_relu:
            out             = self.relu(out)

        return out



class myResNet(nn.Module):
    def __init__(self, options):
        super(myResNet, self).__init__()

        self.nc             = options.model.nc
        self.ndf            = options.model.ndf
        self.arch           = options.model.arch

        self.block_type     = self.arch.split('_')[0].split('-')[1]
        self.blocks         = [int(x) for x in self.arch.split('_')[1].split('-')]
        self.get_scores     = options.model.score
        self.n_classes      = 1 if options.training.loss == BCE else 2
        self.mix            = options.model.mix
        self.freeze_cnn     = options.training.freeze_cnn

        self.latent         = AUTOENC_TR in options.model.system_mode or not options.model.score

        self.n_planes       = [self.ndf * (2**i) for i in range(len(self.blocks))]
        self.f_size         = self.n_planes[-1]
        self.out_chs        = self.f_size
        self.out_relu       = options.model.out_relu

        self.out_shape      = options.training.img_size//32
        self.out_vec_shape  = self.out_shape*self.out_shape*self.out_chs

        conv_layers         = []

        if self.block_type.lower() not in ['stride', 'maxpool']:
            raise ValueError('Invalid block type for myResNet. Accepted values are stride and maxpool. Got %s.' %(self.block_type))

        if self.block_type == 'stride':
            ResidualBlock   = ResidualBlockStride
        elif self.block_type == 'maxpool':
            ResidualBlock   = ResidualBlockPool

        in_channels         = self.nc
        for _i, ix in enumerate(self.n_planes, 0):
            for nb in range(self.blocks[_i]):
                out_channels = ix
                conv_layers.append(ResidualBlock(in_channels, out_channels, downsample=(nb == 0), out_relu=self.out_relu))
                in_channels  = ix

        self.model          = nn.Sequential( *conv_layers )

        if self.get_scores:
            self.agg_linear = nn.Conv2d(self.out_chs, self.n_classes, self.out_shape, stride=1, padding=0)

        if self.mix:
            self.mixer      = nn.Linear(self.out_vec_shape, 1)
            self.mixer.weight.data.normal_(0.0, 0.02/self.out_vec_shape)
            self.mixer.bias.data.fill_(0.0)

        for m in self.modules():
            weights_init(m, nonlinearity='relu')

    def forward(self, x):
        with ExitStack() as stack:
            if self.freeze_cnn:
                stack.enter_context(torch.no_grad())
                self.model.eval()

            y               = self.model(x)

        return_dict         = {}

        if self.mix:
            mscore          = self.mixer(y.view(-1, self.out_vec_shape).contiguous())
            return_dict['mix']          = mscore

        if self.get_scores:
#            ag                          = self.model.avgpool(y).view(-1, self.out_chs).contiguous()
            ag              = self.agg_linear(y).squeeze(3).squeeze(2)
            return_dict['score']        = ag

        return_dict['latent']           = y

        return return_dict



# ===============================================================================================================================
#   Multi-layer perceptrons.

class MLP(nn.Module):
    def __init__(self, options):
        super(MLP, self).__init__()

        # Retrieve the list of number of hidden units. This tells
        #   us the number of layers. 
        self.nhid                   = options.model.mlp_nhid
        self.n_layers               = len(options.model.mlp_nhid)
        # Activation to use. 
        self.activation             = eval(options.model.mlp_activ)
        # Dropout to use. A dropout rate less than zero signifies no dropout.
        self.dropout_rate           = options.model.mlp_dropout
        self.use_dropout            = self.dropout_rate > 0
        
        # n_in: Number of attributes to be used. 
        n_in                        = options.model.mlp_nin
        # Number of output units
        n_out                       = 1 if options.training.loss == BCE else 2
       
        # Record layers here to build the MLP. 
        layers                      = []
        prev_h                      = n_in

        # Build the MLP layer-by-layer. 
        for next_h in self.nhid:
            layers.append(nn.Linear(prev_h, next_h))
            layers.append(self.activation())
            if self.use_dropout:
                layers.append(nn.Dropout(self.dropout_rate))
            prev_h                  = next_h

        # The final layer is a linear layer scoring the patient. 
        layers.append(nn.Linear(prev_h, n_out))

        # Put it all into a main layer.
        self.main                   = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.main(inputs)


# ===============================================================================================================================
class AttentionNetworkSimple(nn.Module):
    """
    Needs more work. 
    """
    def __init__(self, options, cnn_out_size):
        super(AttentionNetworkSimple, self).__init__()

        self.attention_arch         = options.model.attention_arch
        self.K                      = int(self.attention_arch.split('_')[1].split('-')[0])
        self.L                      = cnn_out_size

        self.options                = options
        self.f_size                 = cnn_out_size

        self.H_net                  = Identity()

        self.attention              = nn.Sequential(
                nn.Linear(self.f_size, self.K)
        )

    def forward(self, stack):
        # H = stack ::: N x L
        A                           = self.attention(stack)         # N x K
        A                           = torch.transpose(A, 1, 0)      # K x N
        A                           = F.softmax(A, dim=1)           # K x N

        M                           = torch.mm(A, stack)            # K x L
        M                           = M.view(-1, self.K * self.L)
        return M, A
# ===============================================================================================================================


# ===============================================================================================================================
#   Attention Network 
class AttentionNetwork(nn.Module):
    def __init__(self, options, cnn_out_size):
        super(AttentionNetwork, self).__init__()

        self.attention_arch         = options.model.attention_arch
        self.L                      = int(self.attention_arch.split('_')[1].split('-')[0])
        self.D                      = int(self.attention_arch.split('_')[1].split('-')[1])
        self.K                      = int(self.attention_arch.split('_')[1].split('-')[2])

        self.options                = options
        self.f_size                 = cnn_out_size

        self.H_net                  = nn.Sequential(
                nn.Linear(self. f_size, self.L),
                nn.ReLU(True),
        )

        self.attention              = nn.Sequential(
                nn.Linear(self.L, self.D),
                nn.Tanh(),
                nn.Linear(self.D, self.K),
        )

    def forward(self, stack):
        H                           = self.H_net(stack)

        A                           = self.attention(H)
        A                           = torch.transpose(A, 1, 0) 
        A                           = F.softmax(A, dim=1)

        M                           = torch.mm(A, H).view(-1, self.L * self.K)

        return M, A
        
#
# ===============================================================================================================================
#   Attention Network Gated
class AttentionNetworkGated(nn.Module):
    def __init__(self, options, cnn_out_size):
        super(AttentionNetworkGated, self).__init__()

        self.attention_arch         = options.model.attention_arch
        self.L                      = int(self.attention_arch.split('_')[1].split('-')[0])
        self.D                      = int(self.attention_arch.split('_')[1].split('-')[1])
        self.K                      = int(self.attention_arch.split('_')[1].split('-')[2])

        self.options                = options
        self.f_size                 = cnn_out_size

        self.H_net                  = nn.Sequential(
                nn.Linear(self. f_size, self.L),
                nn.ReLU(True),
        )

        self.attention              = nn.Linear(self.L, self.D)
        self.gate                   = nn.Linear(self.L, self.D)
        self.weight                 = nn.Linear(self.D, self.K)


    def forward(self, stack):
        H                           = self.H_net(stack)

        V                           = F.tanh(self.attention(H))
        U                           = torch.sigmoid(self.attention(H))

        A                           = self.weight(U * V)
        A                           = torch.transpose(A, 1, 0) 
        A                           = F.softmax(A, dim=1)

        M                           = torch.mm(A, H).view(-1, self.L * self.K)

        return M, A
        
# ===============================================================================================================================
#   Fully-convolutional neural networks (not resnets)

class FCNN(nn.Module):
    def __init__(self, options):
        super(FCNN, self).__init__()
        self.options                = options
        self.mpops                  = [int(m) for m in options.model.arch.split('_')[-1].split('-')]
        ndf                         = options.model.ndf
        self.img_size               = options.training.img_size
        self.nc                     = options.model.nc
        nc                          = self.nc
        self.ch_exp                 = options.model.ch_exp

        assert self.ch_exp in ['linear', 'exponential'],    'model.ch_exp must be one of [linear, exponential]. Got {}'.format(options.model.ch_exp)

        f_size                      = options.model.f_size if options.model.f_size > 0 else 5
        p_size                      = (f_size - 1)//2

        self.norm_layer             = eval(options.model.norm_layer)

        self.dropout_rate           = options.model.cnn_dropout
        self.use_dropout            = self.dropout_rate > 0

        self.get_scores             = options.model.score
        self.n_classes              = 1 if options.training.loss == BCE else 2
        self.mix                    = options.model.mix

        max_pool_ops                = self.mpops #[2, 2, 2, 3]

        if self.ch_exp == 'exponential':
            latent_l_units          = [ndf*(2**i) for i in range(len(max_pool_ops))] 
        elif self.ch_exp == 'linear':
            latent_l_units          = [ndf * (i + 1) for i in range(len(max_pool_ops))]

        next_nc                     = nc
        layers                      = []

        for i, m in enumerate(max_pool_ops):
            l                       =  latent_l_units[i]

            layers.append(nn.Conv2d(next_nc, l, kernel_size=f_size, stride=1, padding=p_size, bias=False))
            layers.append(self.norm_layer(l))
            layers.append(nn.LeakyReLU(0.2, True))

            layers.append(nn.Conv2d(l, l, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(self.norm_layer(l))
            layers.append(nn.LeakyReLU(0.2, True))
            
            if self.use_dropout:
                layers.append(nn.Dropout(self.dropout_rate))
            layers.append(nn.MaxPool2d(m))

            next_nc = l

        self.main                   = nn.Sequential(*layers)
        self.out_shape              = int(self.img_size/np.prod(max_pool_ops))
        self.out_chs                = latent_l_units[-1]
        self.out_vec_shape          = self.out_chs*self.out_shape*self.out_shape

        if self.get_scores:
#            self.avgpool            = nn.AdaptiveAvgPool2d((1, 1))
#            self.agg_linear         = nn.Linaer(self.out_chs, self.n_classes)
            self.agg_linear             = nn.Sequential(
                    nn.Conv2d(self.out_chs, self.n_classes, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
            )

        for m in self.modules():
            weights_init(m, nonlinearity='relu')

    def forward(self, inputs):
        return_dict                 = {}

        x                           = self.main(inputs)

        if self.get_scores:
#            ag                      = self.avgpool(x).view(-1, self.out_chs).contiguous()
            ag                      = self.agg_linear(x).squeeze(3).squeeze(2)
            return_dict['score']    = ag

        if self.mix:
            mscore                  = self.mixer(x.view(-1, self.out_vec_shape))
            return_dict['mix']      = mscore

        return_dict['latent']       = x

        return return_dict

# ===============================================================================================================================
#   NoisyAND pooling. Might be used later. 

class NoisyANDPooling(nn.Module):
    def __init__(self, options):
        super(NoisyANDPooling, self).__init__()
        n_classes                   = 1 if options.training.loss == BCE else 2
        self.b                      = nn.Parameter(data=torch.FloatTensor(n_classes).fill_(0.5))
        self.a                      = options.model.a_nand * 1.0

    def forward(self, logits):
        p_avg                       = torch.mean(logits, dim=0, keepdim=True)
        p_i_num                     = F.sigmoid(self.a*(p_avg - self.b)) - F.sigmoid(-1*self.a*self.b)
        p_i_den                     = F.sigmoid(self.a*(1 - self.b)) - F.sigmoid(-1*self.a*self.b)
        return p_i_num/p_i_den


# ===============================================================================================================================
#   Upsample decoder for autoencoder-style reconstruction loss. 
class UpsampleDecoder(nn.Module):
    def __init__(self, options):
        super(UpsampleDecoder, self).__init__()

        self.cnn_out_shape          = options.model.cnn_out_shape
        self.img_size               = options.training.img_size

        self.arch                   = options.model.arch          
        ndf                         = options.model.ndf
        nc                          = options.model.nc
        self.upsample_mode          = options.model.upsample_mode
        self.norm_layer             = eval(options.model.norm_layer)

        layers                      = []

        def conv_upsample_block(m, in_planes, out_planes):
            block                   = []
            block.append(nn.Upsample(scale_factor=m, mode=self.upsample_mode, align_corners=False))
            block.append(conv3x3(in_planes, out_planes, stride=1))
            block.append(self.norm_layer(out_planes))
            block.append(nn.ReLU())
            return nn.Sequential( *block )


        if self.arch.startswith('fcnn'):
            #   If FCNN encoder. 
            self.norm_layer         = eval(options.model.norm_layer)

            self.mpops              = [int(m) for m in options.model.arch.split('_')[-1].split('-')]
            self.mpops              = self.mpops[::-1]
            n_planes                = [ndf * (2**i) for i in range(len(self.mpops)-1, -1, -1)] + [ndf]

            out_chs                 = n_planes[0]

            for mi in range(len(self.mpops)):
                m                   = self.mpops[mi]
                in_p                = n_planes[mi]
                out_p               = n_planes[mi + 1]
                layers.append(conv_upsample_block(m, in_p, out_p))

            #   Add a final layer without upsampling. 
            layers.append(conv3x3(out_p, nc, stride=1))

            self.upsampler          = nn.Sequential( *layers )

        elif self.arch.find('resnet') > -1:
            #   If resnet encoder. 
            resnet_n_layers         = int(self.arch.split('_')[-1])

            if self.arch.startswith('narrow-resnet'):
                expansion           = 1 if resnet_n_layers in [10, 14, 18, 34] else 4
                out_chs             = 8 * expansion * ndf 
            else:
                out_chs             = resnet_outsize[resnet_n_layers]
                #   "ndf" is 64 for normal resnets. 
                ndf                 = 64

            #   Number of upsampling layers. 
            n_layers                = int(np.log2(out_chs // ndf)) + 2    
            n_planes                = [out_chs // (2 ** i) for i in range(n_layers-1)]
            n_planes               += [n_planes[-1], nc]
            
            for mi in range(n_layers):
                m                   = 2
                in_p                = n_planes[mi]
                out_p               = n_planes[mi + 1]
                layers.append(conv_upsample_block(m, in_p, out_p))

            layers.append(conv1x1(nc, nc, stride=1))

        elif 'squeezenet' in self.arch:
            # If squeezenet encoder. 
            out_chs                 = 512
           
            # Number of upsampling layers is 4. 
            n_layers                = 4
            n_planes                = [out_chs, 256, 128, 64, 3]

            layers                  = [nn.ReflectionPad2d((0, 1, 0, 1))]

            for mi in range(n_layers):
                m                   = 2
                in_p                = n_planes[mi]
                out_p               = n_planes[mi + 1]
                layers.append(conv_upsample_block(m, in_p, out_p))

            # Add a final layer after the final ReLU
            layers.append(conv1x1(nc, nc, stride=1))

        self.upsampler              = nn.Sequential( *layers )

    def forward(self, x):
        dec                         = self.upsampler(x)
        return dec


# ===============================================================================================================================
def make_cnn(options):
    if options.model.arch.startswith('fcnn'):
        cnn                         = FCNN(options)
    elif 'resnet' in options.model.arch:
        cnn                         = ResNet(options)
    elif 'squeezenet' in options.model.arch:
        cnn                         = SqueezeNet(options)
    elif 'myResNet' in options.model.arch:
        cnn                         = myResNet(options) 
    elif 'tinynetbottleneckresidualbam' in options.model.arch:
        cnn                         = WideTinyNetBottleneckINResidualBAMProj(options)
    elif 'tinynetbottleneckresidual' in options.model.arch:
        cnn                         = WideTinyNetBottleneckINResidualProj(options)
    elif 'tinynetresidualbam' in options.model.arch:
        cnn                         = WideTinyNetResidualBAMInstanceNorm(options)
    elif 'tinynetresidual' in options.model.arch:
        cnn                         = TinyNetResidualReLUNoBN(options)
    elif 'tinynet' in options.model.arch:
        cnn                         = TinyNetV2ReLUNoBN(options)
    return cnn


# ===============================================================================================================================
def make_attention_network(options, out_vec_shape):
    if 'simple' in options.model.attention_arch:
        attention_net               = AttentionNetworkSimple(options, out_vec_shape)
    elif 'ungated' in options.model.attention_arch:
        attention_net               = AttentionNetwork(options, out_vec_shape)
    else:
        attention_net               = AttentionNetworkGated(options, out_vec_shape)
    return attention_net

# ===============================================================================================================================
class Identity(nn.Module):
    """
    Module that does nothing.
    """
    def __init__(self):
        super(Identity, self).__init__()
        self.register_parameter('null', nn.Parameter(torch.FloatTensor(1).fill_(0)))
    def forward(self, x):
        return x


# ===============================================================================================================================
class GatingNetworkAvgPool(nn.Module):
    """
    A gating network that takes an aggregated feature map for a set of smears, and patient attributes. 
    It outputs the probability that the final prediction should favour the blood smears.
    """
    def __init__(self, sz_conv, ch_conv, n_attr):
        super(GatingNetworkAvgPool, self).__init__()

        self.sz_conv                = sz_conv
        self.ch_conv                = ch_conv
        self.n_attr                 = n_attr
        self.pool                   = nn.AdaptiveAvgPool2d((1, 1))
        self.conv                   = nn.Conv2d(self.ch_conv, 1, kernel_size=1, padding=0, stride=1)
        self.linear                 = nn.Linear(1 + self.n_attr, 2)
        self.tanh                   = torch.tanh
        self.softmax                = lambda x: F.softmax(x, dim=1)

    def forward(self, fmap, attr):
        self.c_smears               = self.tanh(self.conv(self.pool(fmap)).squeeze(-1).squeeze(-1))
        self.c_attr                 = attr
        self.l_in                   = torch.cat((self.c_smears, self.c_attr), dim=-1)
        probs                       = self.softmax(self.linear(self.l_in))
        return probs[:,[0]]


# ===============================================================================================================================
class GatingNetworkConcatAvgPool(nn.Module):
    """
    A gating network that takes an aggregated feature map for a set of smears, and patient attributes. 
    It outputs the probability that the final prediction should favour the blood smears.
    """
    def __init__(self, sz_conv, ch_conv, n_attr):
        super(GatingNetworkConcatAvgPool, self).__init__()

        self.sz_conv                = sz_conv
        self.ch_conv                = ch_conv
        self.n_attr                 = n_attr
        self.pool                   = nn.AdaptiveAvgPool2d((1, 1))
        self.linear                 = nn.Linear(self.ch_conv + self.n_attr, 2, bias=False)
        self.tanh                   = torch.tanh
        self.softmax                = lambda x: F.softmax(x, dim=1)

    def forward(self, fmap, attr):
        fmap                        = self.pool(fmap).squeeze(-1).squeeze(-1)
        self.c_attr                 = attr
        self.l_in                   = torch.cat((fmap, self.c_attr), dim=-1)
        probs                       = self.softmax(self.linear(self.l_in))
        return probs[:,[0]]

# ===============================================================================================================================
class GatingNetwork2LayerConcatAvgPool(nn.Module):
    """
    A gating network that takes an aggregated feature map for a set of smears, and patient attributes. 
    It outputs the probability that the final prediction should favour the blood smears.
    """
    def __init__(self, sz_conv, ch_conv, n_attr):
        super(GatingNetwork2LayerConcatAvgPool, self).__init__()

        self.sz_conv                = sz_conv
        self.ch_conv                = ch_conv
        self.n_attr                 = n_attr
        self.pool                   = nn.AdaptiveAvgPool2d((1, 1))
        self.func                   = nn.Sequential(
                nn.Linear(self.ch_conv + self.n_attr, 2, bias=False),
                nn.ReLU(True),
                nn.Linear(2, 2),
        )
        self.tanh                   = torch.tanh
        self.softmax                = lambda x: F.softmax(x, dim=1)

    def forward(self, fmap, attr):
        fmap                        = self.pool(fmap).squeeze(-1).squeeze(-1)
        self.c_attr                 = attr
        self.l_in                   = torch.cat((fmap, self.c_attr), dim=-1)
        probs                       = self.softmax(self.func(self.l_in))
        return probs[:,[0]]



# ===============================================================================================================================

class GatingNetworkSigmoid(nn.Module):
    """
    A gating network that takes an aggregated feature map for a set of smears, and patient attributes. 
    It outputs the probability that the final prediction should favour the blood smears.
    """
    def __init__(self, sz_conv, ch_conv, n_attr):
        super(GatingNetworkSigmoid, self).__init__()

        self.sz_conv                = sz_conv
        self.ch_conv                = ch_conv
        self.n_attr                 = n_attr
        self.conv                   = nn.Conv2d(self.ch_conv, 1, kernel_size=self.sz_conv, padding=0, stride=1)
        self.linear                 = nn.Linear(1 + self.n_attr, 1)
        self.tanh                   = torch.tanh
        self.softmax                = lambda x: F.softmax(x, dim=1)
        self.sigmoid                = torch.sigmoid

    def forward(self, fmap, attr):
        self.c_smears               = self.tanh(self.conv(fmap).squeeze(-1).squeeze(-1))
        self.c_attr                 = attr
        self.l_in                   = torch.cat((self.c_smears, self.c_attr), dim=-1)
        probs                       = self.sigmoid(self.linear(self.l_in))
        return probs[:,[0]]


# ===============================================================================================================================
class GatingNetwork(nn.Module):
    """
    A gating network that takes an aggregated feature map for a set of smears, and patient attributes. 
    It outputs the probability that the final prediction should favour the blood smears.
    """
    def __init__(self, sz_conv, ch_conv, n_attr):
        super(GatingNetwork, self).__init__()

        self.sz_conv                = sz_conv
        self.ch_conv                = ch_conv
        self.n_attr                 = n_attr
        self.conv                   = nn.Conv2d(self.ch_conv, 1, kernel_size=self.sz_conv, padding=0, stride=1)
        self.linear                 = nn.Linear(1 + self.n_attr, 2)
        self.tanh                   = torch.tanh
        self.softmax                = lambda x: F.softmax(x, dim=1)

    def forward(self, fmap, attr):
        self.c_smears               = self.tanh(self.conv(fmap).squeeze(-1).squeeze(-1))
        self.c_attr                 = attr
        self.l_in                   = torch.cat((self.c_smears, self.c_attr), dim=-1)
        probs                       = self.softmax(self.linear(self.l_in))
        return probs[:,[0]]


# ===============================================================================================================================
class DilatedConv2ResBlockNoBN(nn.Module):
    def __init__(self, in_c=3, out_c=32, **kwargs):
        super(DilatedConv2ResBlockNoBN, self).__init__()

        self.in_c   = in_c
        self.out_c  = out_c

        self.res    = nn.Sequential(
            nn.Conv2d(in_c, out_c, **kwargs),
            nn.ReLU(True),

            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1),
        )

        if out_c != in_c or kwargs['stride'] > 1:
            self.mapper = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=1, stride=kwargs['stride'], padding=0),
            )
        else:
            self.mapper = None

        self.relu   = nn.ReLU(True)

    def forward(self, x, **kwargs):
        residual        = self.res(x)
        if self.mapper:
            x           = self.mapper(x)

        out             = self.relu(x + residual)
        return out


# ===============================================================================================================================
class WideTinyNetBottleneckINResidualProj(nn.Module):
    """
    Based on Stergios' suggestion
    """
    def __init__(self, options):
        super(WideTinyNetBottleneckINResidualProj, self).__init__()

        ndf                     = options.model.ndf
        self.freeze_cnn         = options.training.freeze_cnn
        self.ndf                = ndf
        nc                      = options.model.nc
        self.n_classes          = 1 if options.training.loss == BCE else 2
        self.out_chs            = 4 * ndf
        nz                      = self.out_chs
        self.out_shape          = options.training.img_size//32
        self.out_vec_shape      = self.out_shape*self.out_shape*self.out_chs


        self.conv_block         = nn.Sequential(
                # input is B x nc x 224 x 224
                nn.Conv2d(nc, ndf * 4, kernel_size=7, padding=3, stride=2, dilation=1),
                nn.InstanceNorm2d(ndf * 4),
                nn.ReLU(True),
                nn.MaxPool2d(2),

                # state size B x ndf * 4 x 56 x 56
                DilatedConv2BottleneckResBlockInstanceNorm(in_c=ndf * 4, out_c=ndf * 4, kernel_size=3, padding=1, stride=1, dilation=1),

                # state size B x ndf * 4 x 56 x 56
                DilatedConv2BottleneckResBlockInstanceNorm(in_c=ndf * 4, out_c=ndf * 4, kernel_size=3, padding=1, stride=2, dilation=1),

                # state size B x ndf * 8 x 28 x 28
                DilatedConv2BottleneckResBlockInstanceNorm(in_c=ndf * 4, out_c=ndf * 8, kernel_size=3, padding=1, stride=1, dilation=1),

                # state size B x ndf * 8 x 28 x 28
                DilatedConv2BottleneckResBlockInstanceNorm(in_c=ndf * 8, out_c=ndf * 8, kernel_size=3, padding=1, stride=2, dilation=1),

                # state size B x ndf * 16 x 28 x 28
                DilatedConv2BottleneckResBlockInstanceNorm(in_c=ndf * 8, out_c=ndf * 8, kernel_size=3, padding=1, stride=1, dilation=1),

                # state size B x ndf * 16 x 28 x 28
                DilatedConv2BottleneckResBlockInstanceNorm(in_c=ndf * 8, out_c=ndf * 16, kernel_size=3, padding=1, stride=2, dilation=1),

                # state size B x ndf * 16 x 14 x 14
                DilatedConv2BottleneckResBlockInstanceNorm(in_c=ndf * 16, out_c=ndf * 16, kernel_size=3, padding=1, stride=1, dilation=1),

                # state size B x ndf * 16 x 14 x 14
                DilatedConv2BottleneckResBlockInstanceNorm(in_c=ndf * 16, out_c=ndf * 8, kernel_size=3, padding=1, stride=2, dilation=1),

                # state size B x ndf * 8 x 7 x 7
                DilatedConv2BottleneckResBlockInstanceNorm(in_c=ndf * 8, out_c=nz, kernel_size=1, padding=0, stride=1, dilation=1),
        )
        self.pooler             = nn.AdaptiveAvgPool2d((1,1))
        self.avg_pool           = lambda x: self.pooler(x).squeeze(3).squeeze(2)
        self.proj               = nn.Sequential(
            nn.Linear(nz, nz),
            nn.ReLU(True),
            nn.Linear(nz, nz),
        )

    def forward(self, x, rotation=False, **kwargs):
        return_dict             = {}

        feats                   = self.avg_pool(self.conv_block(x))
        return_dict['latent']   = feats
        y                       = self.proj(feats)
        return_dict['proj']     = y.unsqueeze(2).unsqueeze(3)
        
        return return_dict





class WideTinyNetBottleneckINResidualBAMProj(nn.Module):
    """
    Based on Stergios' suggestion
    """
    def __init__(self, options):
        super(WideTinyNetBottleneckINResidualBAMProj, self).__init__()

        ndf                     = options.model.ndf
        self.freeze_cnn         = options.training.freeze_cnn
        self.ndf                = ndf
        nc                      = options.model.nc
        self.n_classes          = 1 if options.training.loss == BCE else 2
        self.out_chs            = 4 * ndf
        nz                      = self.out_chs
        self.out_shape          = options.training.img_size//32
        self.out_vec_shape      = self.out_shape*self.out_shape*self.out_chs

 

        self.conv_block         = nn.Sequential(
                # input is B x nc x 224 x 224
                nn.Conv2d(nc, ndf * 4, kernel_size=7, padding=3, stride=2, dilation=1),
                nn.InstanceNorm2d(ndf * 4),
                nn.ReLU(True),
                nn.MaxPool2d(2),

                # state size B x ndf * 4 x 56 x 56
                DilatedConv2BottleneckResBlockInstanceNorm(in_c=ndf * 4, out_c=ndf * 4, kernel_size=3, padding=1, stride=1, dilation=1),
                BAM(ndf * 4),

                # state size B x ndf * 4 x 56 x 56
                DilatedConv2BottleneckResBlockInstanceNorm(in_c=ndf * 4, out_c=ndf * 8, kernel_size=3, padding=1, stride=2, dilation=1),
                BAM(ndf * 8),

                # state size B x ndf * 8 x 28 x 28
                DilatedConv2BottleneckResBlockInstanceNorm(in_c=ndf * 8, out_c=ndf * 8, kernel_size=3, padding=1, stride=1, dilation=1),
                BAM(ndf * 8),

                # state size B x ndf * 8 x 28 x 28
                DilatedConv2BottleneckResBlockInstanceNorm(in_c=ndf * 8, out_c=ndf * 16, kernel_size=3, padding=1, stride=2, dilation=1),
                BAM(ndf * 16),

                # state size B x ndf * 16 x 28 x 28
                DilatedConv2BottleneckResBlockInstanceNorm(in_c=ndf * 16, out_c=ndf * 16, kernel_size=3, padding=1, stride=1, dilation=1),
                BAM(ndf * 16),

                # state size B x ndf * 16 x 28 x 28
                DilatedConv2BottleneckResBlockInstanceNorm(in_c=ndf * 16, out_c=ndf * 16, kernel_size=3, padding=1, stride=2, dilation=1),
                BAM(ndf * 16),

                # state size B x ndf * 16 x 14 x 14
                DilatedConv2BottleneckResBlockInstanceNorm(in_c=ndf * 16, out_c=ndf * 16, kernel_size=3, padding=1, stride=1, dilation=1),
                BAM(ndf * 16),

                # state size B x ndf * 16 x 14 x 14
                DilatedConv2BottleneckResBlockInstanceNorm(in_c=ndf * 16, out_c=ndf * 8, kernel_size=3, padding=1, stride=2, dilation=1),
                BAM(ndf * 8),

                # state size B x ndf * 8 x 7 x 7
                DilatedConv2BottleneckResBlockInstanceNorm(in_c=ndf * 8, out_c=nz, kernel_size=1, padding=0, stride=1, dilation=1),
        )
        self.pooler             = nn.AdaptiveAvgPool2d((1,1))
        self.avg_pool           = lambda x: self.pooler(x).squeeze(3).squeeze(2)
        self.proj               = nn.Sequential(
            nn.Linear(nz, nz),
            nn.ReLU(True),
            nn.Linear(nz, nz),
        )

    def forward(self, x, rotation=False, **kwargs):
        return_dict             = {}

        feats                   = self.conv_block(x)
        return_dict['f_map']    = feats
        y                       = self.proj(self.avg_pool(feats))
        return_dict['latent']   = y.unsqueeze(2).unsqueeze(3)
        
        return return_dict





class WideTinyNetResidualBAMInstanceNorm(nn.Module):
    def __init__(self, options):
        super(WideTinyNetResidualBAMInstanceNorm, self).__init__()

        ndf                     = options.model.ndf
        self.freeze_cnn         = options.training.freeze_cnn
        self.ndf                = ndf
        nc                      = options.model.nc
        self.n_classes          = 1 if options.training.loss == BCE else 2
        self.out_chs            = 4 * ndf
        nz                      = self.out_chs
        self.out_shape          = options.training.img_size//32
        self.out_vec_shape      = self.out_shape*self.out_shape*self.out_chs


        self.conv_block         = nn.Sequential(
                # input is B x nc x 224 x 224
                nn.Conv2d(nc, ndf * 4, kernel_size=7, padding=3, stride=2, dilation=1),
                nn.ReLU(True),
                nn.MaxPool2d(2),

                # state size B x ndf * 4 x 56 x 56
                DilatedConv2ResBlockNoBN(in_c=ndf * 4, out_c=ndf * 4, kernel_size=3, padding=1, stride=1, dilation=1),
                nn.MaxPool2d(2),

                # state size B x ndf * 4 x 28 x 28
                DilatedConv2ResBlockNoBN(in_c=ndf * 4, out_c=ndf * 8, kernel_size=3, padding=1, stride=1, dilation=1),
                BAMInstanceNorm(ndf * 8),

                # state size B x ndf * 8 x 28 x 28
                DilatedConv2ResBlockNoBN(in_c=ndf * 8, out_c=ndf * 8, kernel_size=3, padding=1, stride=2, dilation=1),
                BAMInstanceNorm(ndf * 8),

                # state size B x ndf * 8 x 14 x 14
                DilatedConv2ResBlockNoBN(in_c=ndf * 8, out_c=ndf * 16, kernel_size=3, padding=1, stride=2, dilation=1),
                BAMInstanceNorm(ndf * 16),

                # state size B x ndf * 16 x 7 x 7
                DilatedConv2ResBlockNoBN(in_c=ndf * 16, out_c=nz, kernel_size=1, padding=0, stride=1, dilation=1),
        )

    def forward(self, x, **kwargs):
        return_dict             = {}

        feats       = self.conv_block(x)
        return_dict['latent'] = feats
        return return_dict



# ===============================================================================================================================
def kl_divergence(rho, rho_hat):
    return F.kl_div(torch.log(rho_hat), rho, reduce=True, size_average=False) + \
           F.kl_div(torch.log(1 - rho_hat), 1 - rho, reduce=True, size_average=False)
