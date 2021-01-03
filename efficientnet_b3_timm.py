import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
import timm

class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention, self.gamma.item()


class efficientnet_b3(nn.Module):
    def __init__(self, args):
        super(efficientnet_b3, self).__init__()
        self.choice = args.choice
        self.num_classes = args.num_classes  # number of output classes for model
        self.device = args.device
        self.mixup_alpha = args.mixup_alpha
        self.is_contrastive = args.is_contrastive
        self.projection_size = args.projection_size
        self.dropout_prob = args.dropout_prob
        self.predict_cat_label = args.predict_cat_label
        self.init_model_extension = args.init_model_extension
        self.use_self_attn = args.use_self_attn

        self.self_attn_layers = args.self_attn_layers
        if self.use_self_attn:
            self.self_attn_layers = args.self_attn_layers.split(',')
            if self.self_attn_layers[-1] == '':
                self.self_attn_layers = self.self_attn_layers[:-1]

        if self.choice == 1:
            self.model = timm.create_model('tf_efficientnet_b3' + self.init_model_extension, pretrained=True)
            print("Loaded weights from : ", 'tf_efficientnet_b3' + self.init_model_extension)
        else:
            self.model = timm.create_model('tf_efficientnet_b3' + self.init_model_extension, pretrained=False)

        self.model = nn.Sequential(*list(self.model.children())[:-1])  # throw away last FC layer and swish activation

        if self.is_contrastive:
            self.projection_head = nn.Linear(in_features=args.embedding_size, out_features=self.projection_size)

        if self.predict_cat_label:
            self.cat_label_classifier = nn.Linear(in_features=args.embedding_size, out_features=4)

        self.classifier = nn.Linear(in_features=args.embedding_size, out_features=self.num_classes)

        if self.use_self_attn:
            if '0' in self.self_attn_layers:
                self.sa_0 = SelfAttention(in_dim=24)
            if '1' in self.self_attn_layers:
                self.sa_1 = SelfAttention(in_dim=32)
            if '2' in self.self_attn_layers:
                self.sa_2 = SelfAttention(in_dim=48)
            if '3' in self.self_attn_layers:
                self.sa_3 = SelfAttention(in_dim=96)
            if '4' in self.self_attn_layers:
                self.sa_4 = SelfAttention(in_dim=136)
            if '5' in self.self_attn_layers:
                self.sa_5 = SelfAttention(in_dim=232)
            if '6' in self.self_attn_layers:
                self.sa_6 = SelfAttention(in_dim=384)


    def perform_mixup(self, args, data, label, cat_label=None):
        self.mixup_lambdas = torch.from_numpy(
            np.random.beta(args.mixup_alpha, args.mixup_alpha, size=(data.size()[0], 1))).to(self.device).float()
        self.shuffled_idx_mm = torch.randperm(data.size()[0])

        if label is not None:
            mix_label = (self.mixup_lambdas * label) + ((1 - self.mixup_lambdas) * label[self.shuffled_idx_mm, :].float())
            if cat_label is not None:
                mix_cat_label = (self.mixup_lambdas * cat_label) + ((1 - self.mixup_lambdas) * cat_label[self.shuffled_idx_mm, :].float())
            else:
                mix_cat_label = None
        else:
            mix_label = None
            mix_cat_label = None

        if len(data.size()) == 4:
            self.mixup_lambdas = self.mixup_lambdas.unsqueeze(2).unsqueeze(2)
        mix_data = (self.mixup_lambdas * data) + ((1 - self.mixup_lambdas) * data[self.shuffled_idx_mm, :])

        return mix_data, mix_label, mix_cat_label

    def rand_bbox(self, size, lam):
        if len(size) > 2:
            W = size[2]
            H = size[3]
            if 'torch' in str(lam.dtype):
                cut_rat = np.sqrt(1. - lam.cpu().numpy())
            else:
                cut_rat = np.sqrt(1. - lam)
            # cut_rat = lam.cpu().numpy()
            cut_w = (W * cut_rat).astype(np.int)
            cut_h = (H * cut_rat).astype(np.int)
            # uniform
            # if not self.use_passed_lambda:
            # self.cx = np.random.randint(W)
            # self.cy = np.random.randint(H)

            cut_w = np.reshape(np.array(cut_w), (-1,))
            cut_h = np.reshape(np.array(cut_h), (-1,))

            self.cx = []
            for i in range(len(cut_w)):
                self.cx.append(np.random.choice(np.arange(cut_w[i] // 2, W - cut_w[i] // 2)))
            self.cx = np.array(self.cx)

            self.cy = []
            for i in range(len(cut_h)):
                self.cy.append(np.random.choice(np.arange(cut_h[i] // 2, H - cut_h[i] // 2)))
            self.cy = np.array(self.cy)

            # print("self.use_passed_lambda: ", self.use_passed_lambda)
            bbx1 = np.clip(self.cx - cut_w // 2, 0, W)
            bby1 = np.clip(self.cy - cut_h // 2, 0, H)
            bbx2 = np.clip(self.cx + cut_w // 2, 0, W)
            bby2 = np.clip(self.cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def perform_cutmix(self, args, data, label):
        if not self.use_passed_lambda:
            self.shuffled_idx_mcm = torch.randperm(data.size()[0])
        else:
            self.shuffled_idx_mcm = self.shuffled_idx_mm
        if not self.use_passed_lambda:
            self.mixup_lambdas = np.random.beta(args.mixup_alpha, args.mixup_alpha, size=(data.size()[0], 1))

        if len(data.size()) > 2:
            mix_data = data
            # bbx1, bby1, bbx2, bby2 = self.rand_bbox(mix_data.size(), self.mixup_lambdas)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(mix_data.size(), self.mixup_lambdas)

            # adjust lambda to exactly match pixel ratio
            # self.mixup_lambdas = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
            bbx1 = torch.from_numpy(bbx1).to(self.device)
            bby1 = torch.from_numpy(bby1).to(self.device)
            bbx2 = torch.from_numpy(bbx2).to(self.device)
            bby2 = torch.from_numpy(bby2).to(self.device)
            for i in range(bbx1.size()[0]):
                mix_data[i, :, bbx1[i]:bbx2[i], bby1[i]:bby2[i]] = data[self.shuffled_idx_mcm[i], :, bbx1[i]:bbx2[i],
                                                                   bby1[i]:bby2[i]]
            # self.mixup_lambdas = torch.from_numpy(self.mixup_lambdas).to(self.device)
            self.mixup_lambdas = torch.from_numpy(self.mixup_lambdas).to(self.device)
        elif len(data.size()) == 2:
            self.mixup_lambdas = torch.from_numpy(self.mixup_lambdas).to(self.device).float()
            self.mixup_lambdas_orig = self.mixup_lambdas

            mix_data = data[self.shuffled_idx_mcm, :]
            # drop_features_count = torch.round(data.size()[1] * (self.prev_mixup_lambda)).squeeze(1).to(dtype=torch.long, device=self.device)
            drop_features_count = torch.round(data.size()[1] * (self.mixup_lambdas)).squeeze(1).to(dtype=torch.long,
                                                                                                   device=self.device)

            drop_init_index = torch.Tensor(data.size()[0]).random_(0, data.size()[1]).to(dtype=torch.long,
                                                                                         device=self.device)
            invalid_start_index = drop_init_index + drop_features_count > data.size()[0]
            drop_init_index[invalid_start_index] = drop_init_index[invalid_start_index] - drop_features_count[
                invalid_start_index]
            drop_features_mask_sample_1 = torch.ones_like(data).to(self.device)
            # drop_features_mask_sample_1[:, drop_init_index:(drop_init_index+drop_features_count)] = 0
            for i in range(drop_features_mask_sample_1.size()[0]):
                drop_features_mask_sample_1[i, drop_init_index[i]:drop_init_index[i] + drop_features_count[i]] = 0
                drop_features_mask_sample_1 = drop_features_mask_sample_1.to(torch.long)
            mix_data = (data * drop_features_mask_sample_1) + (mix_data * (1 - drop_features_mask_sample_1))
        if len(self.mixup_lambdas.size()) == 4:
            self.mixup_lambdas = self.mixup_lambdas.squeeze(2).squeeze(2)
        # mix_label = (self.mixup_lambdas * label) + ((1-self.mixup_lambdas) * label[self.shuffled_idx_mcm, :].double())
        mix_label = (self.mixup_lambdas * label) + ((1 - self.mixup_lambdas) * label[self.shuffled_idx_mcm, :].double())

        if 'torch' not in str(self.mixup_lambdas.dtype):
            self.mixup_lambdas = torch.from_numpy(self.mixup_lambdas)
        self.mixup_lambdas_orig = self.mixup_lambdas
        if len(self.mixup_lambdas_orig.size()) > 2:
            self.mixup_lambdas_orig = self.mixup_lambdas_orig.squeeze(2).squeeze(2)
        if 'torch' not in str(self.shuffled_idx_mcm.dtype):
            self.shuffled_idx_mcm = torch.from_numpy(self.shuffled_idx_mcm)

        if args.gt_plus_mix:
            return torch.cat([data, mix_data], dim=0), torch.cat([label, mix_label])
        else:
            del data, label
            return mix_data, mix_label

    def forward(self, args, x, label, cat_label=None, enable_mixup=False, training=False):
        gammas = []
        if enable_mixup:
            self.mixup_layer = np.random.choice(np.arange(1, 7))  # Choose one layer for mixup, randomly.
        else:
            self.mixup_layer = None
            self.mixup_lambdas = None

        output = x

        output = self.model[2](self.model[1](self.model[0](output)))
        # -----------------
        # i: 0   output.size(): torch.Size([16, 24, 192, 192])
        # i: 1   output.size(): torch.Size([16, 32, 96, 96])
        # i: 2   output.size(): torch.Size([16, 48, 48, 48])
        # i: 3   output.size(): torch.Size([16, 96, 24, 24])
        # i: 4   output.size(): torch.Size([16, 136, 24, 24])
        # i: 5   output.size(): torch.Size([16, 232, 12, 12])
        # i: 6   output.size(): torch.Size([16, 384, 12, 12])

        # loop over all the blocks in efficientnet-b4
        for i in range(7):
            if enable_mixup and self.mixup_layer == i:  # offsetting to (i+1) for input mixup
                if args.mixup_method == 'manifold_mixup':
                    output, label, cat_label = self.perform_mixup(args, output, label, cat_label)
                if args.mixup_method == 'manifold_cutmix':
                    output, label, cat_label = self.perform_cutmix(args, output, label, cat_label)
            output = self.model[3][i](output)
            # print("i: {}    output.size(): {}".format(i, output.size()))

            if str(i) == '0' and self.use_self_attn and (str(i) in self.self_attn_layers):
                # print("att at: {}".format(i))
                output, att_map, gamma_val = self.sa_0(output)
                gammas.append(gamma_val)
            if str(i) == '1' and self.use_self_attn and (str(i) in self.self_attn_layers):
                # print("att at: {}".format(i))
                output, att_map, gamma_val = self.sa_1(output)
                gammas.append(gamma_val)
            if str(i) == '2' and self.use_self_attn and (str(i) in self.self_attn_layers):
                # print("att at: {}".format(i))
                output, att_map, gamma_val = self.sa_2(output)
                gammas.append(gamma_val)
            if str(i) == '3' and self.use_self_attn and (str(i) in self.self_attn_layers):
                # print("att at: {}".format(i))
                output, att_map, gamma_val = self.sa_3(output)
                gammas.append(gamma_val)
            if str(i) == '4' and self.use_self_attn and (str(i) in self.self_attn_layers):
                # print("att at: {}".format(i))
                output, att_map, gamma_val = self.sa_4(output)
                gammas.append(gamma_val)
            if str(i) == '5' and self.use_self_attn and (str(i) in self.self_attn_layers):
                # print("att at: {}".format(i))
                output, att_map, gamma_val = self.sa_5(output)
                gammas.append(gamma_val)
            if str(i) == '6' and self.use_self_attn and (str(i) in self.self_attn_layers):
                # print("att at: {}".format(i))
                output, att_map, gamma_val = self.sa_6(output)
                gammas.append(gamma_val)
        # -----------------
        output = self.model[4](output)  # conv_head
        output = self.model[5](output)  # batch-norm
        output = self.model[6](output)  # SiLU
        output = self.model[7](output)  # Select Adaptive Pooling
        embeddings = F.dropout(output, p=self.dropout_prob, training=training, inplace=True)

        # -----------------
        if self.predict_cat_label:
            cat_label_output = self.cat_label_classifier(embeddings)
        else:
            cat_label_output = None

        output = self.classifier(output)

        return output, embeddings, label, cat_label_output, cat_label, gammas
