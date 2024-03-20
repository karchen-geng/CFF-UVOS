import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from transformers import SegformerModel


# basic modules
class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('relu', nn.ReLU())
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = Conv(c, c, 3, 1, 1)
        self.conv2 = nn.Sequential(ConvRelu(c, c, 1, 1, 0), Conv(c, c, 1, 1, 0))
        self.conv3 = nn.Sequential(ConvRelu(2, 16, 3, 1, 1), Conv(16, 1, 3, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        c = torch.sigmoid(self.conv2(F.adaptive_avg_pool2d(x, output_size=(1, 1))) + self.conv2(
            F.adaptive_max_pool2d(x, output_size=(1, 1))))
        x = x * c
        s = torch.sigmoid(
            self.conv3(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)))
        x = x * s
        return x


# encoding module
class Encoder(nn.Module):
    def __init__(self, ver):
        super().__init__()
        self.ver = ver

        # ResNet-101 backbone
        if ver == 'rn101':
            backbone = tv.models.resnet101(pretrained=True)
            self.conv1 = backbone.conv1
            self.bn1 = backbone.bn1
            self.relu = backbone.relu
            self.maxpool = backbone.maxpool
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
            self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # MiT-b1 backbone
        if ver == 'mitb1':
            self.backbone = SegformerModel.from_pretrained('nvidia/mit-b1')
            self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img):

        # ResNet-101 backbone
        if self.ver == 'rn101':
            x = (img - self.mean) / self.std
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            s4 = x
            x = self.layer2(x)
            s8 = x
            x = self.layer3(x)
            s16 = x
            x = self.layer4(x)
            s32 = x
            return {'s4': s4, 's8': s8, 's16': s16, 's32': s32}

        # MiT-b1 backbone
        if self.ver == 'mitb1':
            x = (img - self.mean) / self.std
            x = self.backbone(x, output_hidden_states=True).hidden_states
            s4 = x[0]
            s8 = x[1]
            s16 = x[2]
            s32 = x[3]
            #  8-256-96-96,,8-512-48-48,,8-1024-24-24,,8-2048-12-12
            return {'s4': s4, 's8': s8, 's16': s16, 's32': s32}


# decoding module
class Decoder(nn.Module):
    def __init__(self, ver):
        super().__init__()

        # ResNet-101 backbone
        if ver == 'rn101':
            self.conv1 = ConvRelu(2048, 256, 1, 1, 0)
            self.blend1 = ConvRelu(256, 256, 3, 1, 1)
            self.cbam1 = CBAM(256)
            self.conv2 = ConvRelu(1024, 256, 1, 1, 0)
            self.blend2 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam2 = CBAM(256)
            self.conv3 = ConvRelu(512, 256, 1, 1, 0)
            self.blend3 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam3 = CBAM(256)
            self.conv4 = ConvRelu(256, 256, 1, 1, 0)
            self.blend4 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam4 = CBAM(256)
            self.predictor = Conv(256, 2, 3, 1, 1)

        # MiT-b1 backbone
        if ver == 'mitb1':
            self.conv1 = ConvRelu(512, 256, 1, 1, 0)
            self.blend1 = ConvRelu(256, 256, 3, 1, 1)
            self.cbam1 = CBAM(256)
            self.conv2 = ConvRelu(320, 256, 1, 1, 0)
            self.blend2 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam2 = CBAM(256)
            self.conv3 = ConvRelu(128, 256, 1, 1, 0)
            self.blend3 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam3 = CBAM(256)
            self.conv4 = ConvRelu(64, 256, 1, 1, 0)
            self.blend4 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam4 = CBAM(256)
            self.predictor = Conv(256, 2, 3, 1, 1)

    def forward(self, app_feats, mo_feats):
        x = self.conv1(app_feats['s32'] + mo_feats['s32'])
        x = self.cbam1(self.blend1(x))
        s16 = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = torch.cat([self.conv2(app_feats['s16'] + mo_feats['s16']), s16], dim=1)
        x = self.cbam2(self.blend2(x))
        s8 = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = torch.cat([self.conv3(app_feats['s8'] + mo_feats['s8']), s8], dim=1)
        x = self.cbam3(self.blend3(x))
        s4 = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = torch.cat([self.conv4(app_feats['s4'] + mo_feats['s4']), s4], dim=1)
        x = self.predictor(self.cbam4(self.blend4(x)))
        final_score = F.interpolate(x, scale_factor=4, mode='bicubic')
        return final_score



class CascadeBroadcastingModule(nn.Module):
    def __init__(self):
        super(CascadeBroadcastingModule, self).__init__()
        self.conv_s4 = nn.Conv2d(256*2, 1, 3, 1, 1)
        self.conv_s8 = nn.Conv2d(512*2, 1, 3, 1, 1)
        self.conv_s16 = nn.Conv2d(1024*2, 1, 3, 1, 1)
        self.conv_s32 = nn.Conv2d(2048*2, 1, 3, 1, 1)

    def forward(self, img_feats, flow_feats):

        concat_s4 = torch.cat([img_feats['s4'], flow_feats['s4']], dim=1)
        concat_s4 = torch.sigmoid(self.conv_s4(concat_s4))
        img_feats['s4'] = img_feats['s4'] * concat_s4 + img_feats['s4']

        # F.interpolate(img_s4, size=[h, w], mode='bicubic')

        concat_s8 = torch.cat([img_feats['s8'], flow_feats['s8']], dim=1)
        concat_s8 = torch.sigmoid(self.conv_s8(concat_s8))
        img_feats['s8'] = img_feats['s8'] * concat_s8 + img_feats['s8']

        concat_s16 = torch.cat([img_feats['s16'], flow_feats['s16']], dim=1)
        concat_s16 = torch.sigmoid(self.conv_s16(concat_s16))
        img_feats['s16'] = img_feats['s16'] * concat_s16 + img_feats['s16']

        concat_s32 = torch.cat([img_feats['s32'], flow_feats['s32']], dim=1)
        concat_s32 = torch.sigmoid(self.conv_s32(concat_s32))
        img_feats['s32'] = img_feats['s32'] * concat_s32 + img_feats['s32']


        return img_feats

class SGFM(nn.Module):
    def __init__(self):
        super(SGFM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1_layer4_channel_wise = nn.Conv2d(2048, 2048, 1, bias=True)
        self.conv1x1_layer4_spatial = nn.Conv2d(2048, 1, 1, bias=True)
        self.conv = nn.Conv2d(2048 * 2, 1, 3, 1, 1)
        self.conv_final = nn.Conv2d(2048 * 2, 1, 3, 1, 1)


    def sgfm(self, img_feat, flow_feat, app_feats_cbm, conv1x1_channel_wise, conv1x1_spatial):

        flow_feat_map = conv1x1_spatial(flow_feat)
        flow_feat_map = nn.Sigmoid()(flow_feat_map)  # 8-1-96-96
        spatial_attentioned_img_feat = flow_feat_map * img_feat  # 8-256-96-96


        feat_vec = self.avg_pool(spatial_attentioned_img_feat)  # 8-256-1-1
        feat_vec = conv1x1_channel_wise(feat_vec)  # 8-256-1-1
        feat_vec = nn.Softmax(dim=1)(feat_vec) * feat_vec.shape[1]  # 8-256-1-1
        channel_attentioned_img_feat = spatial_attentioned_img_feat * feat_vec
        sgfm_feat = channel_attentioned_img_feat + img_feat

        concat_s32 = torch.cat([img_feat, app_feats_cbm], dim=1)
        concat_s32 = torch.sigmoid(self.conv(concat_s32))
        img_feats = img_feat * concat_s32 + img_feat

        concat_final = torch.cat([img_feats, sgfm_feat], dim=1)
        concat_final = torch.sigmoid(self.conv_final(concat_final))
        final_feat = img_feat * concat_final + img_feat

        return final_feat
    # Soft Attention
    def sa(self, f1, f2):
        c1 = self.c_f1(f1)  # channel -> 1
        c2 = self.c_f2(f2)  # channel -> 1

        n, c, h, w = c1.shape
        c1 = c1.view(-1, h * w)
        c2 = c2.view(-1, h * w)

        c1 = F.softmax(c1, dim=1)
        c2 = F.softmax(c2, dim=1)

        c1 = c1.view(n, c, h, w)
        c2 = c2.view(n, c, h, w)

        # Hadamard product
        f1_sa = c1 * f1
        f2_sa = c2 * f2

        return f1_sa, f2_sa

    def forward(self, img_feats, flow_feats, app_feats_cbm):  # 8-256-96-96,,8-512-48-48,,8-1024-24-24,,8-1024-12-12

        img_feats['s32'] = self.sgfm(img_feats['s32'], flow_feats['s32'], app_feats_cbm['s32'], self.conv1x1_layer4_channel_wise,
                                                  self.conv1x1_layer4_spatial) + img_feats['s32']

        return img_feats




class VOS(nn.Module):
    def __init__(self, ver):
        super().__init__()
        self.app_encoder = Encoder(ver)
        self.mo_encoder = Encoder(ver)
        self.sgfm = SGFM()
        self.cbm = CascadeBroadcastingModule()
        self.decoder = Decoder(ver)


# TMO model
class CFF_UVOS(nn.Module):
    def __init__(self, ver, aos):
        super().__init__()
        self.vos = VOS(ver)
        self.aos = aos

    def forward(self, imgs, flows):
        B, L, _, H1, W1 = imgs.size()
        _, _, _, H2, W2 = flows.size()
        # B:8,L:1
        # resize to 384p
        s = 384
        imgs = F.interpolate(imgs.view(B * L, -1, H1, W1), size=(s, s), mode='bicubic').view(B, L, -1, s, s)
        flows = F.interpolate(flows.view(B * L, -1, H2, W2), size=(s, s), mode='bicubic').view(B, L, -1, s, s)

        # for each frame
        score_lst = []
        mask_lst = []
        for i in range(L):
            # adaptive output selection off
            if B != 1 or not self.aos:
                # query frame prediction
                app_feats = self.vos.app_encoder(imgs[:, i])
                mo_feats = self.vos.mo_encoder(flows[:, i])
                # 级联增强模块
                app_feats_cbm = self.vos.cbm(app_feats, mo_feats)
                app_feats_mix = self.vos.sgfm(app_feats, mo_feats, app_feats_cbm)
                final_score = self.vos.decoder(app_feats_mix, mo_feats)
                final_score = F.interpolate(final_score, size=(H1, W1), mode='bicubic')

            # adaptive output selection on
            if B == 1 and self.aos:
                # query frame prediction
                app_feats = self.vos.app_encoder(imgs[:, i])
                mo_feats_img = self.vos.mo_encoder(imgs[:, i])
                mo_feats_flow = self.vos.mo_encoder(flows[:, i])

                app_feats_cbm = self.vos.cbm(app_feats, mo_feats_flow)
                app_feats_mix = self.vos.sgfm(app_feats, mo_feats_flow, app_feats_cbm)
                final_score_img = self.vos.decoder(app_feats_mix, mo_feats_img)
                final_score_flow = self.vos.decoder(app_feats_mix, mo_feats_flow)
                # print(self.aos)
                # adaptive output selection
                h = 0.05
                pred_seg_img = torch.softmax(final_score_img, dim=1)
                conf_img = torch.sum(pred_seg_img[pred_seg_img > 1 - h] - (1 - h))
                pred_seg_flow = torch.softmax(final_score_flow, dim=1)
                conf_flow = torch.sum(pred_seg_flow[pred_seg_flow > 1 - h] - (1 - h))
                w = (conf_img > conf_flow).float()
                final_score = w * final_score_img + (1 - w) * final_score_flow
                final_score = F.interpolate(final_score, size=(H1, W1), mode='bicubic')

            # store soft scores
            if B != 1:
                score_lst.append(final_score)

            # store hard masks
            if B == 1:
                pred_seg = torch.softmax(final_score, dim=1)
                pred_mask = torch.max(pred_seg, dim=1, keepdim=True)[1]
                mask_lst.append(pred_mask)

        # generate output
        output = {}
        if B != 1:
            output['scores'] = torch.stack(score_lst, dim=1)
        if B == 1:
            output['masks'] = torch.stack(mask_lst, dim=1)
        return output


