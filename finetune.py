import copy
import os.path as osp
from collections import OrderedDict
import math
from pcgrad import PCGrad
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import logging
from CLIP import clip, clip_ori
from CLIP.simple_tokenizer import SimpleTokenizer as _Tokenizer
from logger import create_logger
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import math
from scipy import io as sio
import torch.utils.data
import torchvision.models as models
from imgaug import augmenters as iaa
import pandas as pd
import torchvision.transforms as transforms
import torchvision
import time
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
from skimage import io
import os
import torchvision.transforms.functional as tf
from PIL import Image
import cv2
import torchvision.models as models
from functools import partial

import matplotlib.pyplot as plt
import lmdb
from prefetch_generator import BackgroundGenerator
import random
from torch.cuda.amp import autocast as autocast
import argparse
import statistics

_tokenizer = _Tokenizer()

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def angle_between_gradients(grad1, grad2):
    # Flatten the gradients
    grad1_flat = torch.cat([g.view(-1) for g in grad1])
    grad2_flat = torch.cat([g.view(-1) for g in grad2])

    # Compute the cosine of the angle
    cos_theta = torch.dot(grad1_flat, grad2_flat) / (torch.norm(grad1_flat) * torch.norm(grad2_flat))

    # Compute the angle in degrees
    angle = torch.acos(cos_theta) * (180.0 / torch.pi)

    return angle.item()

@torch.no_grad()
def _update_teacher_model(model, zs_model, keep_rate=0.996):
    # Extract the parameters from the student model (model) where gradients are enabled
    student_model_dict = {
        key: value for key, value in model.named_parameters() if value.requires_grad
    }
    # for key, value in student_model_dict.items():
    #     print(f"{key}: {value.size()}")
    # Initialize a new OrderedDict to store the updated parameters for the teacher model (zs_model)
    new_teacher_dict = OrderedDict()

    for key, value in zs_model.state_dict().items():
        # Check if the corresponding parameter exists in the student model and is not part of 'module.prompt_learner2.ctx'
        if key in student_model_dict:
            # Update the parameter using the Exponential Moving Average (EMA) formula
            new_teacher_dict[key] = student_model_dict[key] * (1 - keep_rate) + value * keep_rate
        else:
            # For parameters not meeting the conditions, keep the original values from the teacher model
            new_teacher_dict[key] = value
    # for key, value in new_teacher_dict.items():
    #     print(f"{key}: {value.size()}")
    # Update the teacher model (zs_model) with the new parameters
    zs_model.load_state_dict(new_teacher_dict)

class ProGradLoss(nn.Module):
    def __init__(self, T, lambda_val=1.0):
        super(ProGradLoss, self).__init__()
        self.T = T  # Temperature for softmax scaling
        self.lambda_val = lambda_val  # Lambda value for ProGrad adjustment

    def forward(self, stu_logits, tea_logits, target):
        # Compute the cross-entropy loss
        output2_0 = F.softmax(stu_logits, dim=-1)

        xe_loss = -torch.sum(
            target[:, 0] * torch.log(output2_0[:, 0]) + (1 - target[:, 0]) * torch.log(output2_0[:, 1])) / \
                output2_0.shape[0]

        # Compute the KL divergence loss
        tea_prob = F.softmax(tea_logits / self.T, dim=-1)
        kl_loss = -tea_prob * F.log_softmax(stu_logits / self.T, dim=-1) * self.T**2
        kl_loss = kl_loss.sum(1).mean()

        return xe_loss, kl_loss, output2_0

    def prograd_update(self, Gd, Gg):
        # Compute the dot product between Gd and Gg
        dot_product = torch.dot(Gd.view(-1), Gg.view(-1))
        if dot_product <= 0:
            Gprograd = Gd
        else:
            # Project Gd onto the orthogonal direction of Gg
            Gprograd = Gd - self.lambda_val * dot_product / torch.linalg.norm(Gg)**2 * Gg

        return Gprograd

def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

def load_clip_ori_to_cpu(backbone_name):
    url = clip_ori._MODELS[backbone_name]
    model_path = clip_ori._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip_ori.build_model(state_dict or model.state_dict())

    return model

class IQACLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        classes = ['high', 'low']
        scenes = ['animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant', 'still_life', 'others']
        prompts = torch.cat([clip_ori.tokenize(f"a photo with {c} quality.") for c in classes]).cuda()
        prompts_sce = torch.cat([clip_ori.tokenize(f"a photo of a {c}") for c in scenes]).cuda()
        self.model_text = TextCLIP(self.clip_model).cuda()
        with torch.no_grad():
            text_features = self.model_text(prompts)
            text_features = text_features / text_features.norm(dim=-1,
                                                               keepdim=True)
            text_features_sce = self.model_text(prompts_sce)
            text_features_sce = text_features_sce / text_features_sce.norm(dim=-1,
                                                               keepdim=True)
        self.text_features_sce = text_features_sce
        self.text_features = text_features
        self.model_image = self.clip_model.visual
    def forward(self, image):
        image_features = self.model_image(image)
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()

        text_features = self.text_features
        text_features = text_features.to(image_features.device)
        logits = logit_scale * image_features @ text_features.t()

        text_features_sce = self.text_features_sce
        text_features_sce = text_features_sce.to(image_features.device)
        logits_sce = logit_scale * image_features @ text_features_sce.t()
        return logits, logits_sce

class TextCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, prompts, tokenized_prompts, flag = 0):
        if flag:
            x = self.token_embedding(prompts).type(self.dtype)
            x = x + self.positional_embedding.type(self.dtype)
        else:
            x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, class_token_position):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 4
        ctx_init = False
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        CSC = False
        self.class_token_position = class_token_position
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class PromptLearner_ori(nn.Module):
    def __init__(self, classnames, clip_model, class_token_position):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 4
        ctx_init = False
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        CSC = False
        self.class_token_position = class_token_position
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip_ori.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip_ori.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP_ori(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner2 = PromptLearner_ori(classnames[2], clip_model, 'end')
        self.tokenized_prompts2 = self.prompt_learner2.tokenized_prompts

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        text_features2 = self.text_encoder(self.prompt_learner2(), self.tokenized_prompts2)
        text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
        logits2 = logit_scale * image_features @ text_features2.t()

        return logits2

class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        # self.prompt_learner0 = PromptLearner(classnames[0], clip_model, 'end')
        # self.tokenized_prompts0 = self.prompt_learner0.tokenized_prompts
        #
        # self.prompt_learner1 = PromptLearner(classnames[1], clip_model, 'end')
        # self.tokenized_prompts1 = self.prompt_learner1.tokenized_prompts
        scenes = ['animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant', 'still_life', 'others']

        self.prompt_learner2 = PromptLearner(classnames[2], clip_model, 'end')
        self.tokenized_prompts2 = self.prompt_learner2.tokenized_prompts

        self.image_encoder = clip_model.visual

        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.prompts_sce = torch.cat([clip.tokenize(f"a photo of a {c}") for c in scenes])

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features[1]
        
        
        logit_scale = self.logit_scale.exp()

        text_features2 = self.text_encoder(self.prompt_learner2(), self.tokenized_prompts2)
        text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
        logits2 = logit_scale * image_features @ text_features2.t()
        if self.training:
            self.prompts_sce = self.prompts_sce.cuda()
            text_features_sce = self.text_encoder(self.prompts_sce, self.prompts_sce, flag = 1)
            text_features_sce = text_features_sce / text_features_sce.norm(dim=-1,
                                                                           keepdim=True)
            
            logits_sce = logit_scale * image_features @ text_features_sce.t()

            return logits2, logits_sce
        else:
            return logits2


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Mydataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = torch.FloatTensor(labels)
    def __getitem__(self, index):
        return torch.from_numpy(self.imgs[index]),self.labels[index]
    def __len__(self):
        return (self.imgs).shape[0]

def train(model, train_loader, optimizer, scaler, epoch, device, all_train_loss, logger,zs_model):
    model.train()
    zs_model.eval()
    st = time.time()
    op0=[]
    op1=[]
    op2=[]
    tg=[]
    total_gradient_gd = None
    total_gradient_gg = None
    criterion = ProGradLoss(T=1.0,lambda_val=config.lda)
    for batch_idx, (data,  target) in enumerate(train_loader):
        data,  target = data.to(device),  target.to(device)
        torch.random.manual_seed(len(train_loader) * epoch + batch_idx)
        rd_ps = torch.randint(20, (3,))
        data = data[:, :, rd_ps[0]:rd_ps[0] + 224, rd_ps[1]:rd_ps[1] + 224]
        if rd_ps[1] < 10:
            data = torch.flip(data, dims=[3])

        data = data.float()
        data /= 255
        data[:, 0] -= 0.485
        data[:, 1] -= 0.456
        data[:, 2] -= 0.406
        data[:, 0] /= 0.229
        data[:, 1] /= 0.224
        data[:, 2] /= 0.225
        with autocast():
            output_0, output_sce = model(data)
            with torch.no_grad():
                zs_clip_output, zs_output_sce = zs_model(data)
            xe_loss, _, output2_0 = criterion(output_0,zs_clip_output.detach(),target)
            _, kl_loss, _ = criterion(output_sce, zs_output_sce.detach(), target)
            loss= xe_loss
        optimizer.zero_grad()
        # Compute gradients for xe_loss
        xe_loss.backward(retain_graph=True)
        # Store Gd (gradient of xe_loss)

        Gd = [param.grad.clone() for param in model.parameters() if param.grad is not None]
        # Clear gradients to compute Gg
        optimizer.zero_grad()
        # Compute gradients for kl_loss
        kl_loss.backward()
        # Store Gg (gradient of kl_loss)
        Gg = [param.grad.clone() for param in model.parameters() if param.grad is not None]
        params_with_grads = [param for param in model.parameters() if param.grad is not None]
        if epoch % 5 == 4:
            angle = angle_between_gradients(Gg, Gd)
        else:
            angle = None

        for param, gd, gg in zip(params_with_grads, Gd, Gg):
            # Compute Gprograd for each parameter
            param.grad = criterion.prograd_update(gd, gg)
            # Update parameters manually

        optimizer.step()
        optimizer.zero_grad()


        all_train_loss.append(loss.item())
        tg = np.concatenate((tg, target[:, 0].cpu().numpy()))

        op0 = np.concatenate((op0, output2_0[:, 0].detach().cpu().numpy()))

    if epoch %100==0:
        pearson_corr = pd.Series(op0[::1]).corr(pd.Series(tg[::1]), method="pearson")
        spearman_corr = pd.Series(op0[::1]).corr(pd.Series(tg[::1]), method="spearman")
        logger.info('Train ALL Pearson0: %f', pearson_corr)
        logger.info('Train ALL Spearman0: %f', spearman_corr)

    return all_train_loss, angle


def test(model, test_loader, epoch, device, all_test_loss):
    model.eval()
    test_loss = 0

    op0 = []
    op1 = []
    op2 = []
    tg = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data,  target = data.to(device),target.to(device)
            data = data[:, :,10:10 + 224, 10:10 + 224]

            data = data.float()
            data /= 255
            data[:, 0] -= 0.485
            data[:, 1] -= 0.456
            data[:, 2] -= 0.406
            data[:, 0] /= 0.229
            data[:, 1] /= 0.224
            data[:, 2] /= 0.225
            with autocast():
                output_0 = model(data)
                output2_0 = F.softmax(output_0[:, :2])
                loss0 = -torch.sum( target[:, 0] * torch.log(output2_0[:, 0]) + (1 - target[:, 0]) * torch.log(output2_0[:, 1])) /  output_0.shape[0]

            all_test_loss.append(loss)
            test_loss += loss
            tg = np.concatenate((tg, target[:, 0].cpu().numpy()))

            op0 = np.concatenate((op0, output2_0[:, 0].detach().cpu().numpy()))


    test_loss /= (batch_idx + 1)
    pl0=pd.Series((op0[::1])).corr((pd.Series(tg[::1])), method="pearson")
    sr0=pd.Series((op0[::1])).corr((pd.Series(tg[::1])), method="spearman")

    if epoch==-2:
        print('Test ALL Pearson0:',pl0,'Test  ALL Spearman0:', sr0 )

    return all_test_loss, pl0, sr0


def mysampling (x,nums=3):
    hist,bins=np.histogram(x,bins=nums)
    inds=[]
    for i in range(len(bins)-1):
        inds.append(np.where(((x<bins[i+1])*(x>bins[i]))==1)[0])
    return inds

def normalization(data):
    range = np.max(data) - np.min(data)
    return (data - np.min(data)) / range

def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
    device = torch.device("cuda")
    if config.dataset=='clive':
        all_data = sio.loadmat('./livew_244.mat')
    elif config.dataset == 'koniq':
        all_data = sio.loadmat('./Koniq_244.mat')
    elif config.dataset == 'pipal':
        all_data = sio.loadmat('./PIPAL_244.mat')

    X = all_data['X']
    Y = all_data['Y'].transpose(1, 0)
    Y = Y.reshape(Y.shape[0], 1)

    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest'].transpose(1, 0)
    Ytest = Ytest.reshape(Ytest.shape[0], 1)
    del all_data
    X = np.concatenate((X, Xtest), axis=0)
    Y = np.concatenate((Y, Ytest), axis=0)
    # Y = normalization(Y)
    ind = np.arange(0, len(X))
    num_image = config.num_image
    logger = logging.getLogger('Train {}'.format(num_image))
    
    if config.pretrained:
        os.makedirs('./prompt_log_align/' + 'upper_clip' + '_' + str(config.dataset) + '_' + str(num_image), exist_ok=True)
        # train for one epoch
        create_logger(
            logger,
            output_dir='./prompt_log_align/' + 'upper_clip' + '_' + str(config.dataset) + '_' + str(num_image),
        )
    else:
        os.makedirs('./prompt_log_align/' + 'semantic_clip' + '_' + 'lda ' + str(
            config.lda) + '_' + str(config.dataset) + '_' + str(num_image), exist_ok=True)
        # train for one epoch
        create_logger(
            logger,
            output_dir='./prompt_log_align/' + 'semantic_clip' + '_' + 'lda ' + str(
                config.lda) + '_' + str(config.dataset) + '_' + str(num_image),
        )
    best_plccs = []
    best_srccs = []

    for i in range(10):
        np.random.shuffle(ind)  # Shuffle the indices
        # Calculate the split index for 80% of the data
        split_idx = int(len(ind) * 0.8)
        # Split ind into training and testing indices
        inds_train = ind[:split_idx]
        inds_train = np.random.choice(inds_train, size=num_image, replace=False)
        inds_test = ind[split_idx:]
        # Index into X and Y to create training and testing subsets
        Xtest = X[inds_test]
        Ytest = Y[inds_test]
        Ytest = normalization(Ytest)
        Xtrain = X[inds_train]
        Ytrain = Y[inds_train]
        Ytrain = normalization(Ytrain)

        prompt_path = '/data/lxd/xmu_NR-IQA-IJCAJ/NR-IQA-IJCAJ/MetaIQA-master/metapth/meta_vt_prompt/best_prompt.pt'

        checkpoint = torch.load(prompt_path, map_location="cpu")
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        classnames = [['good','bad' ],['clear','unclear' ],['high quality','low quality']]
        clip_model = load_clip_to_cpu('ViT-B/16').float()
        print("Building custom CLIP")
        model = CustomCLIP(classnames, clip_model)
        model.load_state_dict(new_state_dict, strict=False)
        #build zero-shot teacher
        if config.pretrained:
            print("Building IQACLIP Teacher")

            clip_model_zs = load_clip_to_cpu('ViT-B/16').float()
            zs_model = CustomCLIP(classnames, clip_model_zs)
            checkpoint = torch.load('./model_checkpoint/koniq_8058.pt')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            zs_model.load_state_dict(new_state_dict)

        else:
            print("Building CLIP Teacher")
            clip_model_zs = load_clip_ori_to_cpu('ViT-B/16').float()
            zs_model = IQACLIP(clip_model_zs)
        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        for name, param in model.named_parameters():

            if name_to_update  in name:
                param.requires_grad_(True)


            if name_to_update not in name:
                param.requires_grad_(False)

        for name, param in model.named_parameters():
            if "text_encoder.transformer.resblocks.8" in name:
                param.requires_grad_(True)
            if "text_encoder.transformer.resblocks.9" in name:
                param.requires_grad_(True)
            if "text_encoder.transformer.resblocks.10" in name:
                param.requires_grad_(True)
            if "text_encoder.transformer.resblocks.11" in name:
                param.requires_grad_(True)
            if "text_encoder.ln_final" in name:
                param.requires_grad_(True)

            if "image_encoder.transformer.resblocks.8" in name:
                param.requires_grad_(True)
            if "image_encoder.transformer.resblocks.9" in name:
                param.requires_grad_(True)
            if "image_encoder.transformer.resblocks.10" in name:
                param.requires_grad_(True)
            if "image_encoder.transformer.resblocks.11" in name:
                param.requires_grad_(True)
            if "image_encoder.ln_post" in name:
                param.requires_grad_(True)


        model = nn.DataParallel(model).to(device)
        zs_model = nn.DataParallel(zs_model).to(device)
        zs_model.eval()


        train_dataset = Mydataset(Xtrain, Ytrain)
        test_dataset = Mydataset(Xtest,  Ytest)



        max_plsp=-1
        min_loss = 1e8
        lr = 0.01
        weight_decay = 1e-4
        batch_size = 32*6
        epochs = 2000
        num_workers_train = 0
        num_workers_test = 0
        ct=0

        train_loader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_train,pin_memory=True)
        test_loader = DataLoaderX(test_dataset, batch_size=batch_size*18, shuffle=False, num_workers=num_workers_test,pin_memory=True)

        all_train_loss = []
        all_test_loss = []
        all_test_loss,pl,pl2= test(model, test_loader, -1, device, all_test_loss)
        logger.info('Training image: {}, Split: {}, End! PLCC: {}, SRCC: {}'.format(num_image, i, pl, pl2))

        ct = 0
        lr = 0.001
        scaler =  torch.cuda.amp.GradScaler()
        best_plcc = -1
        best_srcc = -1

        for epoch in range(epochs):

            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
            start = time.time()
            all_train_loss, angle = train(model, train_loader, optimizer, scaler, epoch, device, all_train_loss,logger,zs_model)
      
            if epoch%5==4:
                ct += 1
                all_test_loss, pl,pl2 = test(model, test_loader, epoch, device, all_test_loss)
                logger.info('Training image: {}, Split: {}, End! PLCC: {}, SRCC: {}, Angle: {}'.format(num_image, i, pl, pl2, angle))
     
            if pl + pl2 > best_plcc + best_srcc:
                best_plcc = pl
                best_srcc = pl2
  
                ct = 0


            if ct > 5 and epoch > 10:
                lr *= 0.3
                ct = 0
                if lr<5e-7:
                    break
        best_plccs.append(best_plcc)
        best_srccs.append(best_srcc)
        logger.info('Training image: {}, Split: {}, Best PLCC: {}, Best SRCC: {}'.format(num_image, i, best_plccs[i], best_srccs[i]))
    # After the loop, calculate the average PLCC and SRCC
    average_plcc = sum(best_plccs) / len(best_plccs)
    average_srcc = sum(best_srccs) / len(best_srccs)
    median_plcc = statistics.median(best_plccs)
    median_srcc = statistics.median(best_srccs)

    logger.info(f"Average PLCC over 10 iterations: {average_plcc}")
    logger.info(f"Average SRCC over 10 iterations: {average_srcc}")
    logger.info(f"Median PLCC over 10 iterations: {median_plcc}")
    logger.info(f"Median SRCC over 10 iterations: {median_srcc}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='clive',
                        help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--num_image', type=int, default=50,
                        help='Number of few-shot image')
    parser.add_argument('--pretrained',  action='store_true')
    parser.add_argument('--lda', type=float, default=5,
                        help='weight of align')
    config = parser.parse_args()

    setup_seed(2024)
    main(config)
