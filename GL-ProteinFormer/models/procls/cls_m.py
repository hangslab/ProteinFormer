import torch
import torch.nn as nn
from models.swin import swin_base, swin_tiny
# from models.resnet_pre import ResNet, Bottleneck
from models.resnet_pre import Bottleneck
from models.vit import vit_base_patch16

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet(nn.Module):

    def __init__(self, block, layers, in_c=3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.in_c = in_c
        if in_c == 3:
            self.conv1 = nn.Conv2d(in_c, self.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
        else:
            self.conv1_ = nn.Conv2d(in_c, self.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.num_classes = num_classes
        # if num_classes == 1000:
        #     self.fc = nn.Linear(512 * block.expansion, num_classes)
        # else:
        #     self.fc_ = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        feat_list = []
        if self.in_c == 3:
            x = self.conv1(x)
        else:
            x = self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feat_list.append(x)
        x = self.layer2(x)
        feat_list.append(x)
        x = self.layer3(x)
        feat_list.append(x)
        x = self.layer4(x)
        feat_list.append(x)

        # x = self.avgpool(feature)
        # x = torch.flatten(x, 1)
        # if self.num_classes == 1000:
        #     x = self.fc(x)
        # else:
        #     x = self.fc_(x)

        # return x, feature
        return feat_list


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

# helpers
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        x = x.flatten(2, 3).transpose(1, 2)
        x = self.norm(x)
        embed_c = x.size(2)
        x = x.transpose(1, 2).reshape(B, embed_c, H // self.patch_size, W // self.patch_size)
        return x

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, attn=None, **kwargs):
        return self.fn(self.norm(x), attn, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, attn=None):
        return self.net(x)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x).flatten(2).transpose(1, 2)
        # n = N // 21
        # x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        # x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        # x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        # x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        # x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        # x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        # x = torch.cat([x1, x2, x3], dim=1)
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, attn_last=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        
        if attn_last is not None:
            attn = attn + attn_last
            
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn

class Transformer_Conv(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                ConvFFN(in_features=dim, hidden_features=int(dim * 0.25), drop=dropout)
            ]))
    def forward(self, x, H, W):
        attn_last = None
        for attn_layer, ff in self.layers:
            x_attn, attn_last = attn_layer(x, attn_last)
            x = x_attn + x
            x = ff(x, H, W) + x
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        attn_last = None
        for attn_layer, ff in self.layers:
            x_attn, attn_last = attn_layer(x, attn_last)
            x = x_attn + x
            x = ff(x) + x
        return x



class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_layer = nn.Sequential(nn.Linear(in_features, 512),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(512, 128),
                                           nn.ReLU(inplace=True))
        self.last_layer = nn.Linear(128, n_classes)
        
    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        out_feature = self.feature_layer(x)
        x = self.last_layer(out_feature)
        return x


class CLSModel_Pre(nn.Module):
    def __init__(self, n_classes, norm_layer=nn.LayerNorm):
        super(CLSModel_Pre, self).__init__()
        self.backbone = swin_base(out_indices=[3], in_chans=3)
        # self.backbone = swin_tiny(out_indices=[3], in_chans=4)
        # self.backbone = vit_base_patch16()
        # self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], in_c=3, num_classes=n_classes)

        # self.head = ResnetDecoder(768, 7)
        self.head = ResnetDecoder(1024, n_classes)
        # self.head = ResnetDecoder(2048, n_classes)
        # self.head = None

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        # x = (x - x.mean(dim=1)) / (torch.sqrt(x.std(dim=1)) + 0.00001)
        # print('x.mean', x.mean(dim=(2,3)))
        # print('x.std', x.std(dim=(2,3)))
        feat_list = self.backbone(x)
        x = feat_list[-1]
        # feat = self.backbone(x)[-1]
        # print('x.shape', x.shape)
        # assert len(x) == 1
        # x = x[-1]
        x = self.head(x)
        return x, feat_list


class CLSModel(nn.Module):
    def __init__(self,  n_classes, norm_layer=nn.LayerNorm):
        super(CLSModel, self).__init__()
        self.backbone = swin_base(out_indices=[3], in_chans=3)
        # self.backbone = swin_tiny(out_indices=[3], in_chans=4)
        
        # self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], in_c=3, num_classes=7)
        # feat_dim = 2048
        feat_dim = 1024
        embed_dim = 512
        self.patch = PatchEmbed(patch_size=2, in_chans=feat_dim, embed_dim=embed_dim)
        self.decoder = Transformer(dim=embed_dim, depth=6, heads=8, dim_head=64, mlp_dim=embed_dim)
        # self.decoder = Transformer(dim=embed_dim, depth=1, heads=8, dim_head=64, mlp_dim=embed_dim)
        # self.decoder = Transformer_Conv(dim=embed_dim, depth=6, heads=8, dim_head=64, mlp_dim=embed_dim)
        self.head_ = ResnetDecoder(512, n_classes)
        # self.head = None
    def forward(self, x):
        B, _, H, W = x.shape
        # x = (x - x.mean(dim=1)) / (torch.sqrt(x.std(dim=1)) + 0.00001)
        # print('x.mean', x.mean(dim=(2,3)))
        # print('x.std', x.std(dim=(2,3)))
        # x, feat_list = self.backbone(x)
        # with torch.no_grad():
        feat_list = self.backbone(x)
        x = feat_list[-1]
        # feat = self.backbone(x)[-1]
        # print('x.shape', x.shape)
        # assert len(x) == 1
        # x = x[-1]
        x = self.patch(x)
        _, fc, fh, fw = x.shape
        # print('feat.shape', feat.shape)
        token = x.reshape(B, fc, fh, fw).permute(0, 2, 3, 1).flatten(1, 2)
        x = self.decoder(token)
        # x = self.decoder(token, fh, fw)
        tc = x.size(-1)
        x = x.permute(0, 2, 1).reshape(B, tc, fh, fw)
        # print('x', x.shape)
        x = self.head_(x)
        return x, feat_list       


class STModel(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm):
        super(STModel, self).__init__()
        # self.backbone = swin_base(out_indices=[3], in_chans=4)
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], in_c=4, num_classes=13)
        # self.head = ResnetDecoder(1024, 13)
        feat_dim = 2048
        embed_dim = 512
        self.patch = PatchEmbed(patch_size=4, in_chans=feat_dim, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.head = Transformer(dim=embed_dim, depth=6, heads=8, dim_head=64, mlp_dim=embed_dim)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 13)
        )
    def forward(self, x):
        if self.training:
            B, inc, H, W = x.shape
            crop_scale = 2
            crop_h = H // crop_scale
            crop_w = W // crop_scale
            unfold = nn.Unfold(kernel_size=(crop_h, crop_w), stride=(crop_h, crop_w))
            # print('x_crop.shape', unfold(x).shape)
            x = unfold(x).permute(0, 2, 1).reshape(B , int(crop_scale**2), inc, crop_h, crop_w)
            # print('x.shape', x.shape)

        B, N, _, H, W = x.shape
        # print('x.shape', x.shape)
        x = x.flatten(0, 1)
        # x = (x - x.mean(dim=1)) / (torch.sqrt(x.std(dim=1)) + 0.00001)
        # print('x.mean', x.mean(dim=(2,3)))
        # print('x.std', x.std(dim=(2,3)))

        with torch.no_grad():
            _, feat = self.backbone(x)
        
        # print('feat.shape', feat.shape)
        feat = feat[-1]
        feat = self.patch(feat)
        _, fc, fh, fw = feat.shape
        # print('feat.shape', feat.shape)
        token = feat.reshape(B, N, fc, fh, fw).permute(0, 1, 3, 4, 2).flatten(1, 3)
        # print('token.shape', token.shape)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        token = torch.cat((cls_tokens, token), dim=1)
        token = self.head(token)
        cls_token = token[:, 0]
        cls_token = self.to_latent(cls_token)
        # print('feature', feature.shape)
        cls_token = self.mlp_head(cls_token)
        # assert len(x) == 1
        # x = x[-1]
        # x = self.head(x)
        return cls_token, feat



class MultiScaleSTModel(nn.Module):
    def __init__(self, n_classes, norm_layer=nn.LayerNorm):
        super(MultiScaleSTModel, self).__init__()
        self.backbone = swin_base(out_indices=[0, 1, 2, 3], in_chans=3)
        # self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], in_c=4, num_classes=13)
        # self.head = ResnetDecoder(1024, 13)
        # feat_dim = [256, 512, 1024, 2048]
        feat_dim = [128, 256, 512, 1024]
        # patch_sizes = [16, 16, 16, 16]
        embed_dim = 1024
        self.patch = PatchEmbed(patch_size=2, in_chans=1920, embed_dim=embed_dim)
        self.downsample = nn.ModuleList()
        # self.patch = nn.ModuleList()
        for i in range(4):
            # print('i', i)
            if i < 3:
                # print('kernelsize', 2**(3-i))
                avgpool = nn.AvgPool2d(kernel_size=2**(3-i), stride=2**(3-i))
            else:
                # print('downsample identity')
                avgpool = nn.Identity()
            self.downsample.append(avgpool)
            # embedding = PatchEmbed(patch_size=16, in_chans=feat_dim[i], embed_dim=embed_dim)
            # self.patch.append(embedding)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # self.head = Transformer(dim=embed_dim, depth=6, heads=8, dim_head=64, mlp_dim=embed_dim)
        # self.head = Transformer(dim=embed_dim, depth=3, heads=8, dim_head=64, mlp_dim=embed_dim)
        self.decoder = Transformer_Conv(dim=embed_dim, depth=6, heads=8, dim_head=64, mlp_dim=embed_dim)
        self.head_ = ResnetDecoder(embed_dim, n_classes)
        # self.to_latent = nn.Identity()
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, n_classes)
        # )
    def forward(self, x):
        # if self.training:
        #     B, inc, H, W = x.shape
        #     crop_scale = 2
        #     crop_h = H // crop_scale
        #     crop_w = W // crop_scale
        #     unfold = nn.Unfold(kernel_size=(crop_h, crop_w), stride=(crop_h, crop_w))
        #     # print('x_crop.shape', unfold(x).shape)
        #     x = unfold(x).permute(0, 2, 1).reshape(B , int(crop_scale**2), inc, crop_h, crop_w)
        #     # print('x.shape', x.shape)

        # B, N, _, H, W = x.shape
        B, _, H, W = x.shape
        # print('x.shape', x.shape)
        # x = x.flatten(0, 1)
        # x = (x - x.mean(dim=1)) / (torch.sqrt(x.std(dim=1)) + 0.00001)
        # print('x.mean', x.mean(dim=(2,3)))
        # print('x.std', x.std(dim=(2,3)))

        # with torch.no_grad():
        feat_list = self.backbone(x)
        
        # print('feat.shape', feat.shape)
        # with torch.no_grad():
        token = []
        for i in range(4):
            feat = feat_list[i]
            feat = self.downsample[i](feat)
            # feat = self.patch[i](feat)
            # print('i', i)
            # print('feat.shape', feat.shape)
            _, fc, fh, fw = feat.shape
            # print('feat.shape', feat.shape)
            # token.append(feat.reshape(B, N, fc, fh, fw).permute(0, 1, 3, 4, 2).flatten(1, 3))
            token.append(feat)
        # print('token.shape', token.shape)
        token = torch.cat(token, dim=1)
        token = self.patch(token)
        _, fc, fh, fw = token.shape
        # print('feat.shape', feat.shape)
        # token = token.reshape(B, N, fc, fh, fw).permute(0, 1, 3, 4, 2).flatten(1, 3)
        # print('token.shape', token.shape)
        # cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # token = torch.cat((cls_tokens, token), dim=1)
        # token = self.decoder(token)
        # cls_token = token[:, 0]
        # cls_token = self.to_latent(cls_token)

        # # print('feature', feature.shape)
        # cls_token = self.mlp_head(cls_token)
        # print('x', x.shape)
        token = token.reshape(B, fc, fh, fw).permute(0, 2, 3, 1).flatten(1, 2)
        x = self.decoder(token, fh, fw)
        tc = x.size(-1)
        # print('x', x.shape)
        x = x.permute(0, 2, 1).reshape(B, tc, fh, fw)
        # assert len(x) == 1
        # x = x[-1]
        # x = self.head(x)
        # return cls_token, feat
        x = self.head_(x)
        return x, feat_list  