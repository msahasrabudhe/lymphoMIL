import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
        return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

class ChannelGateNoNorm(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGateNoNorm, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
        return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)


class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',    nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
                        padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        return self.gate_s( in_tensor ).expand_as(in_tensor)

class SpatialGateInstanceNorm(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGateInstanceNorm, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',    nn.InstanceNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
                        padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.InstanceNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        return self.gate_s( in_tensor ).expand_as(in_tensor)



class BAMInstanceNorm(nn.Module):
    def __init__(self, gate_channel):
        super(BAMInstanceNorm, self).__init__()
        self.channel_att = ChannelGateNoNorm(gate_channel)
        self.spatial_att = SpatialGateInstanceNorm(gate_channel)
    def forward(self,in_tensor):
        att = 1 + torch.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor



class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,in_tensor):
        att = 1 + torch.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicConvInstanceNorm(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConvInstanceNorm, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.InstanceNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlockInstanceNorm(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlockInstanceNorm, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes * 4, 16 )
        else:
            self.cbam = None

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

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class ResNetCBAMWithProjectionNoReLU(nn.Module):
    def __init__(self, block, layers, ndf, att_type=None):
        super(ResNetCBAMWithProjectionNoReLU, self).__init__()
        self.inplanes = ndf
        # different model config between ImageNet and CIFAR 
        self.conv1 = nn.Conv2d(3, ndf, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7)

        self.bn1 = nn.BatchNorm2d(ndf)
        self.relu = nn.ReLU(inplace=True)

        if att_type=='BAM':
            self.bam1 = BAM(ndf * block.expansion)
            self.bam2 = BAM(ndf * 2 * block.expansion)
            self.bam3 = BAM(ndf * 4 * block.expansion)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        self.layer1 = self._make_layer(block, ndf,  layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, ndf * 2, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, ndf * 4, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, ndf * 8, layers[3], stride=2, att_type=att_type)

        nc_final    = ndf * 8 * block.expansion

        self.f_conv1    = nn.Conv2d(nc_final, nc_final, kernel_size=5, stride=1, padding=0, bias=False)
        self.f_bn1      = nn.BatchNorm2d(nc_final)
        self.f_conv2    = nn.Conv2d(nc_final, nc_final, kernel_size=3, stride=1, padding=0, bias=False)

        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type=='CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        return_dict = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if not self.bam1 is None:
            x = self.bam1(x)

        x = self.layer2(x)
        if not self.bam2 is None:
            x = self.bam2(x)

        x = self.layer3(x)
        if not self.bam3 is None:
            x = self.bam3(x)

        x = self.layer4(x)

        x = self.relu(self.f_bn1(self.f_conv1(x)))
        x = self.f_conv2(x)

        x = x.view(x.size(0), -1)
        return_dict['fg_feats'] = x
        return return_dict

class ResNetCBAMInstanceNorm(nn.Module):
    def __init__(self, block, layers, ndf, att_type=None):
        super(ResNetCBAMInstanceNorm, self).__init__()
        self.inplanes = ndf
        # different model config between ImageNet and CIFAR 
        self.conv1 = nn.Conv2d(3, ndf, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7)

        self.bn1 = nn.InstanceNorm2d(ndf)
        self.relu = nn.ReLU(inplace=True)

        if att_type=='BAM':
            self.bam1 = BAM(ndf * block.expansion)
            self.bam2 = BAM(ndf * 2 * block.expansion)
            self.bam3 = BAM(ndf * 4 * block.expansion)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        self.layer1 = self._make_layer(block, ndf,  layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, ndf * 2, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, ndf * 4, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, ndf * 8, layers[3], stride=2, att_type=att_type)

        nc_final    = ndf * 8 * block.expansion

        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type=='CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        return_dict = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if not self.bam1 is None:
            x = self.bam1(x)

        x = self.layer2(x)
        if not self.bam2 is None:
            x = self.bam2(x)

        x = self.layer3(x)
        if not self.bam3 is None:
            x = self.bam3(x)

        x = self.layer4(x)

        return_dict['fg_feats'] = x
        return return_dict



class ResNetCBAM(nn.Module):
    def __init__(self, block, layers, ndf, att_type=None):
        super(ResNetCBAM, self).__init__()
        self.inplanes = ndf
        # different model config between ImageNet and CIFAR 
        self.conv1 = nn.Conv2d(3, ndf, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7)

        self.bn1 = nn.BatchNorm2d(ndf)
        self.relu = nn.ReLU(inplace=True)

        if att_type=='BAM':
            self.bam1 = BAM(ndf * block.expansion)
            self.bam2 = BAM(ndf * 2 * block.expansion)
            self.bam3 = BAM(ndf * 4 * block.expansion)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        self.layer1 = self._make_layer(block, ndf,  layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, ndf * 2, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, ndf * 4, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, ndf * 8, layers[3], stride=2, att_type=att_type)

        nc_final    = ndf * 8 * block.expansion

        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type=='CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        return_dict = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        h0 = self.maxpool(x)

        h1 = self.layer1(h0)
        if not self.bam1 is None:
            h1 = self.bam1(h1)

        h2 = self.layer2(h1)
        if not self.bam2 is None:
            h2 = self.bam2(h2)

        h3 = self.layer3(h2)
        if not self.bam3 is None:
            h3 = self.bam3(h3)

        h4 = self.layer4(h3)

        return_dict['feats_0']  = h0
        return_dict['feats_1']  = h1
        return_dict['feats_2']  = h2
        return_dict['feats_3']  = h3
        return_dict['fg_feats'] = h4
        return return_dict

class ResNetCBAMInstanceNormWithProjection(nn.Module):
    def __init__(self, block, layers, ndf, att_type=None):
        super(ResNetCBAMInstanceNormWithProjection, self).__init__()
        self.inplanes = ndf
        # different model config between ImageNet and CIFAR 
        self.conv1 = nn.Conv2d(3, ndf, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7)

        self.bn1 = nn.InstanceNorm2d(ndf)
        self.relu = nn.ReLU(inplace=True)

        if att_type=='BAM':
            self.bam1 = BAM(ndf * block.expansion)
            self.bam2 = BAM(ndf * 2 * block.expansion)
            self.bam3 = BAM(ndf * 4 * block.expansion)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        self.layer1 = self._make_layer(block, ndf,  layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, ndf * 2, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, ndf * 4, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, ndf * 8, layers[3], stride=2, att_type=att_type)

        nc_final    = ndf * 8 * block.expansion

        self.f_conv1    = nn.Conv2d(nc_final, nc_final, kernel_size=5, stride=1, padding=0, bias=False)
        self.f_conv2    = nn.Conv2d(nc_final, nc_final, kernel_size=3, stride=1, padding=0, bias=False)

        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type=='CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        return_dict = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if not self.bam1 is None:
            x = self.bam1(x)

        x = self.layer2(x)
        if not self.bam2 is None:
            x = self.bam2(x)

        x = self.layer3(x)
        if not self.bam3 is None:
            x = self.bam3(x)

        x = self.layer4(x)

        x = self.relu(self.f_conv1(x))
        x = self.relu(self.f_conv2(x))

        x = x.view(x.size(0), -1)
        return_dict['fg_feats'] = x
        return return_dict


class ResNetCBAMWithProjection(nn.Module):
    def __init__(self, block, layers, ndf, att_type=None):
        super(ResNetCBAMWithProjection, self).__init__()
        self.inplanes = ndf
        # different model config between ImageNet and CIFAR 
        self.conv1 = nn.Conv2d(3, ndf, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7)

        self.bn1 = nn.BatchNorm2d(ndf)
        self.relu = nn.ReLU(inplace=True)

        if att_type=='BAM':
            self.bam1 = BAM(ndf * block.expansion)
            self.bam2 = BAM(ndf * 2 * block.expansion)
            self.bam3 = BAM(ndf * 4 * block.expansion)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        self.layer1 = self._make_layer(block, ndf,  layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, ndf * 2, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, ndf * 4, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, ndf * 8, layers[3], stride=2, att_type=att_type)

        nc_final    = ndf * 8 * block.expansion

        self.f_conv1    = nn.Conv2d(nc_final, nc_final, kernel_size=5, stride=1, padding=0, bias=False)
        self.f_bn1      = nn.BatchNorm2d(nc_final)
        self.f_conv2    = nn.Conv2d(nc_final, nc_final, kernel_size=3, stride=1, padding=0, bias=False)
        self.f_bn2      = nn.BatchNorm2d(nc_final)

        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type=='CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        return_dict = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        h0 = self.maxpool(x)

        h1 = self.layer1(h0)
        if not self.bam1 is None:
            h1 = self.bam1(h1)

        h2 = self.layer2(h1)
        if not self.bam2 is None:
            h2 = self.bam2(h2)

        h3 = self.layer3(h2)
        if not self.bam3 is None:
            h3 = self.bam3(h3)

        h4 = self.layer4(h3)

        h5 = self.relu(self.f_bn1(self.f_conv1(h4)))
        h6 = self.relu(self.f_bn2(self.f_conv2(h5)))

        h6 = h6.view(h6.size(0), -1)
        return_dict['feats_0']  = h0
        return_dict['feats_1']  = h1
        return_dict['feats_2']  = h2
        return_dict['feats_3']  = h3
        return_dict['feats_4']  = h4
        return_dict['feats_5']  = h5
        return_dict['fg_feats'] = h6
        return return_dict

def ResNetCBAM10(options):
    return ResNetCBAM(BasicBlock, [1, 1, 1, 1], options.ndf, att_type=options.cbam)

def ResNetCBAMInstanceNorm10(options):
    return ResNetCBAMInstanceNorm(BasicBlockInstanceNorm, [1, 1, 1, 1], options.ndf, att_type=options.cbam)

def ResNetCBAM18(options):
    return ResNetCBAM(BasicBlock, [2, 2, 2, 2], options.ndf, att_type=options.cbam)

def ResNetCBAM34(options):
    return ResNetCBAM(BasicBlock, [3, 4, 6, 3], options.ndf, att_type=options.cbam)

def ResNetCBAM10WithProjection(options):
    return ResNetCBAMWithProjection(BasicBlock, [1, 1, 1, 1], options.ndf, att_type=options.cbam)

def ResNetCBAMInstanceNorm10WithProjection(options):
    return ResNetCBAMInstanceNormWithProjection(BasicBlockInstanceNorm, [1, 1, 1, 1], options.ndf, att_type=options.cbam)


def ResNetCBAM10WithProjectionNoReLU(options):
    return ResNetCBAMWithProjectionNoReLU(BasicBlock, [1, 1, 1, 1], options.ndf, att_type=options.cbam)

def ResNetCBAM18WithProjection(options):
    return ResNetCBAMWithProjection(BasicBlock, [2, 2, 2, 2], options.ndf, att_type=options.cbam)

def ResNetCBAM34WithProjection(options):
    return ResNetCBAMWithProjection(BasicBlock, [3, 4, 6, 3], options.ndf, att_type=options.cbam)



def ResidualNet(network_type, depth, num_classes, att_type):

    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type)

    return model
