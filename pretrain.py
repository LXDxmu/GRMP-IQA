from __future__ import print_function, division
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import nn
import pandas as pd
from skimage import transform
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from logger import create_logger
from PIL import Image
import time
import math
import copy
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import warnings
warnings.filterwarnings("ignore")
import random
from scipy.stats import spearmanr
use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True

import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from CLIP import clip
from CLIP.simple_tokenizer import SimpleTokenizer as _Tokenizer

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

from torch.cuda.amp import autocast as autocast


_tokenizer = _Tokenizer()


class ImageRatingsDataset(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform
        # Pre-calculate the maximum and minimum ratings
        self.rating_min = self.images_frame.iloc[:, 1].min()
        self.rating_max = self.images_frame.iloc[:, 1].max()

    def normalization(self, rating):
        # Use pre-calculated maximum and minimum values for normalization
        return (rating - self.rating_min) / (self.rating_max - self.rating_min)

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        # try:
            img_name = str(os.path.join(self.root_dir,str(self.images_frame.iloc[idx, 0])))
            im = Image.open(img_name).convert('RGB')
            if im.mode == 'P':
                im = im.convert('RGB')
            image = np.asarray(im)
            #image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
            rating = self.images_frame.iloc[idx, 1]
            rating = self.normalization(rating)
            sample = {'image': image, 'rating': rating}

            if self.transform:
                sample = self.transform(sample)
            return sample
        # except Exception as e:
        #     pass



class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        return {'image': image, 'rating': rating}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'rating': rating}


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        if random.random() < self.p:
            image = np.flip(image, 1)
            # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        return {'image': image, 'rating': rating}


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        im = image /1.0#/ 255
        im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        im[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        image = im
        return {'image': image, 'rating': rating}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double(),
                'rating': torch.from_numpy(np.float64([rating])).double()}

def computeSpearman(dataloader_valid, model):
    ratings = []
    predictions = []
    with torch.no_grad():
        cum_loss = 0
        for batch_idx, data in enumerate(dataloader_valid):
            inputs = data['image']
            batch_size = inputs.size()[0]
            labels = data['rating'].view(batch_size, -1)
            # labels = labels / 100.0
            if use_gpu:
                try:
                    inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                except:
                    print(inputs, labels)
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs_a = model(inputs)
            outputs_a = F.softmax(outputs_a[:, :2])
            ratings.append(labels.float().cpu().numpy())
            predictions.append(outputs_a[:, 0].float().cpu().numpy().reshape(-1, 1))

    ratings_i = np.vstack(ratings)
    predictions_i = np.vstack(predictions)
    a = ratings_i[:,0]
    b = predictions_i[:,0]
    sp = spearmanr(a, b)
    return sp

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

class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        # self.prompt_learner0 = PromptLearner(classnames[0], clip_model, 'end')
        # self.tokenized_prompts0 = self.prompt_learner0.tokenized_prompts
        #
        # self.prompt_learner1 = PromptLearner(classnames[1], clip_model, 'end')
        # self.tokenized_prompts1 = self.prompt_learner1.tokenized_prompts

        self.prompt_learner2 = PromptLearner(classnames[2], clip_model, 'end')
        self.tokenized_prompts2 = self.prompt_learner2.tokenized_prompts

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features[1]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        # text_features0 = self.text_encoder(self.prompt_learner0(), self.tokenized_prompts0)
        # text_features0 = text_features0 / text_features0.norm(dim=-1, keepdim=True)
        # logits0 = logit_scale * image_features @ text_features0.t()
        #
        # text_features1 = self.text_encoder(self.prompt_learner1(), self.tokenized_prompts1)
        # text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
        # logits1 = logit_scale * image_features @ text_features1.t()

        text_features2 = self.text_encoder(self.prompt_learner2(), self.tokenized_prompts2)
        text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
        logits2 = logit_scale * image_features @ text_features2.t()

        return logits2

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

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

def train_model():
    epochs = 50
    # epochs = 100
    task_num = 5
    noise_num1 = 24
    noise_num2 = 25
    logger = logging.getLogger('Train {}'.format(epochs))
    os.makedirs('./prompt_pretrain_log/', exist_ok=True)
    create_logger(
        logger,
        output_dir='./prompt_pretrain_log/',
    )
    classnames = [['good', 'bad'], ['clear', 'unclear'], ['high quality', 'low quality']]
    clip_model = load_clip_to_cpu('ViT-B/16').float()
    print("Building custom CLIP")
    model = CustomCLIP(classnames, clip_model)
    print("Turning off gradients in both the image and the text encoder")

    exclude_key = 'prompt'
    for n, m in model.named_parameters():
        if exclude_key:
            if isinstance(exclude_key, str):
                if not exclude_key in n:
                    m.requires_grad = False
            elif isinstance(exclude_key, list):
                count = 0
                for i in range(len(exclude_key)):
                    i_layer = str(exclude_key[i])
                    if i_layer in n:
                        count += 1
                if count == 0:
                    m.requires_grad = False
                elif count > 0:
                    print('Finetune layer in backbone:', n)
            else:
                assert AttributeError("Dont support the type of exclude_key!")
        else:
            m.requires_grad = False

    name_to_update = "prompt_learner"

    for name, param in model.named_parameters():

        if name_to_update in name:
            param.requires_grad_(True)
            # print(name)
        #
        # if name_to_update not in name:
        #     param.requires_grad_(False)

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)

    criterion = nn.MSELoss()
    # ignored_params = list(map(id, model.parameters()))
    # base_params = filter(lambda p: id(p) not in ignored_params,
    #                      model.parameters())
    # prompt_lr = 0.000005
    # weight_decay = 0.001
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
    #                                     lr=prompt_lr, weight_decay=weight_decay)
    model.cuda()
    meta_model = copy.deepcopy(model)  # global model, θ
    temp_model = copy.deepcopy(model)
    meta_model = nn.DataParallel(meta_model).cuda()
    temp_model = nn.DataParallel(temp_model).cuda()

    optimizer = optim.Adam(temp_model.parameters(), lr=1e-4)  # inner learning rate α
    # The global optimizer is only used for outer updates
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=1e-2)  # outer learning rate β
    spearman = 0
    for epoch in range(epochs):
        running_loss = 0.0
        # optimizer = exp_lr_scheduler(optimizer, epoch)

        list_noise = list(range(noise_num1))
        np.random.shuffle(list_noise)
        logger.info(f'############# TID 2013 train phase epoch {epoch + 1} ###############')
        count = 0
        meta_model.train()  # Set global model to training mode
        theta_updates = []  # For storing θ updates for all tasks

        list_noise = list(range(noise_num1))
        np.random.shuffle(list_noise)

        for index in list_noise:
            # Each task's inner model and optimizer
            # Load the support set and query set data for the current task
            dataloader_train, dataloader_valid = load_data('train', 'tid2013', index, logger)
            if dataloader_train == 0 or dataloader_valid == 0:
                continue
            dataiter = iter(enumerate(dataloader_valid))
            temp_model.train()  # Set model to training mode
            # Inner optimization: update on the support set
            for batch_idx, data in enumerate(dataloader_train):
                inputs = data['image']
                batch_size = inputs.size()[0]
                labels = data['rating'].view(batch_size, -1)
                # labels = labels / 10.0
                if use_gpu:
                    try:
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                    except:
                        print(inputs, labels)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = temp_model(inputs)
                outputs = F.softmax(outputs[:, :2])
                loss = -torch.sum(labels[:, 0] * torch.log(outputs[:, 0]) + (1 - labels[:, 0]) * torch.log(outputs[:, 1])) / \
                        outputs.shape[0]

                loss.backward()
                optimizer.step()

                idx, data_val = next(dataiter)
                if idx >= len(dataloader_valid)-1:
                    dataiter = iter(enumerate(dataloader_valid))
                inputs_val = data_val['image']
                batch_size1 = inputs_val.size()[0]
                labels_val = data_val['rating'].view(batch_size1, -1)
                # labels_val = labels_val / 10.0
                if use_gpu:
                    try:
                        inputs_val, labels_val = Variable(inputs_val.float().cuda()), Variable(labels_val.float().cuda())
                    except:
                        print(inputs_val, labels_val)
                else:
                    inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

                optimizer.zero_grad()
                outputs_val = temp_model(inputs_val)
                outputs_val = F.softmax(outputs_val[:, :2])
                loss_val = -torch.sum(labels_val[:, 0] * torch.log(outputs_val[:, 0]) + (1 - labels_val[:, 0]) * torch.log(outputs_val[:, 1])) / \
                        outputs_val.shape[0]
                loss_val.backward()
                optimizer.step()

                try:
                    running_loss += loss_val.item()
                except:
                    print('unexpected error, could not calculate loss or do a sum.')

            # Calculate the difference between θ_i and the global model θ for outer optimization
            theta_update = [(param_meta.data - param_temp.data) for param_meta, param_temp in
                            zip(meta_model.parameters(), temp_model.parameters())]
            theta_updates.append(theta_update)
            count += 1
        # Outer optimization: update global model parameters θ
        with torch.no_grad():
            for i, param in enumerate(meta_model.parameters()):
                # Take the average of all tasks' θ_i updates, then update with outer learning rate β
                update_step = sum(theta_update[i] for theta_update in theta_updates) / len(list_noise)
                param.data -= update_step * 1e-2  # outer learning rate β

        epoch_loss = running_loss / count
        logger.info('current loss = %f', epoch_loss)
        running_loss = 0.0
        list_noise = list(range(noise_num2))
        np.random.shuffle(list_noise)
        # list_noise.remove(ii)
        logger.info(f'############# Kadid train phase epoch {epoch + 1} ###############')
        count = 0
        meta_model.train()  # Set global model to training mode

        theta_updates = []  # For storing θ updates for all tasks

        list_noise = list(range(noise_num1))
        np.random.shuffle(list_noise)

        for index in list_noise:
            # Each task's inner model and optimizer
            optimizer = optim.Adam(temp_model.parameters(), lr=1e-4)  # inner learning rate α

            # Load the support set and query set data for the current task
            dataloader_train, dataloader_valid = load_data('train', 'kadid10k', index, logger)
            if dataloader_train == 0 or dataloader_valid == 0:
                continue
            dataiter = iter(enumerate(dataloader_valid))
            temp_model.train()  # Set model to training mode
            # Inner optimization: update on the support set
            for batch_idx, data in enumerate(dataloader_train):
                inputs = data['image']
                batch_size = inputs.size()[0]
                labels = data['rating'].view(batch_size, -1)
                # labels = labels / 10.0
                if use_gpu:
                    try:
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                    except:
                        print(inputs, labels)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = temp_model(inputs)
                outputs = F.softmax(outputs[:, :2])
                loss = -torch.sum(labels[:, 0] * torch.log(outputs[:, 0]) + (1 - labels[:, 0]) * torch.log(outputs[:, 1])) / \
                        outputs.shape[0]

                loss.backward()
                optimizer.step()

                idx, data_val = next(dataiter)
                if idx >= len(dataloader_valid)-1:
                    dataiter = iter(enumerate(dataloader_valid))
                inputs_val = data_val['image']
                batch_size1 = inputs_val.size()[0]
                labels_val = data_val['rating'].view(batch_size1, -1)
                # labels_val = labels_val / 10.0
                if use_gpu:
                    try:
                        inputs_val, labels_val = Variable(inputs_val.float().cuda()), Variable(labels_val.float().cuda())
                    except:
                        print(inputs_val, labels_val)
                else:
                    inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

                optimizer.zero_grad()
                outputs_val = temp_model(inputs_val)
                outputs_val = F.softmax(outputs_val[:, :2])
                loss_val = -torch.sum(labels_val[:, 0] * torch.log(outputs_val[:, 0]) + (1 - labels_val[:, 0]) * torch.log(outputs_val[:, 1])) / \
                        outputs_val.shape[0]
                loss_val.backward()
                optimizer.step()

                try:
                    running_loss += loss_val.item()
                except:
                    print('unexpected error, could not calculate loss or do a sum.')
                count += 1
            # Calculate the difference between θ_i and the global model θ for outer optimization
            theta_update = [(param_meta.data - param_temp.data) for param_meta, param_temp in
                            zip(meta_model.parameters(), temp_model.parameters())]
            theta_updates.append(theta_update)

        # Outer optimization: update global model parameters θ
        with torch.no_grad():
            for i, param in enumerate(meta_model.parameters()):
                # Take the average of all tasks' θ_i updates, then update with outer learning rate β
                update_step = sum(theta_update[i] for theta_update in theta_updates) / len(list_noise)
                param.data -= update_step * 1e-2  # outer learning rate β

        epoch_loss = running_loss / count
        logger.info('current loss = %f', epoch_loss)
        logger.info('############# test phase epoch %2d ###############' % epoch)
        dataloader_train, dataloader_valid = load_data('test', 0, logger=logger)
        meta_model.eval()
        meta_model.cuda()
        sp = computeSpearman(dataloader_valid, meta_model)[0]
        if sp > spearman:
            spearman = sp
            best_model = copy.deepcopy(meta_model)
            torch.save(best_model.state_dict(),
                   './model_pth_final_deep_prompt/best_prompt.pt')
        logger.info('new srocc %4f, best srocc %4f', sp, spearman)
        if epoch % 10 == 9:
            torch.save(meta_model.state_dict(),
                   './model_pth_final_deep_prompt/TID2013_KADID10K_prompt_'+ str(epoch) + '.pt')

def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=2):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate =  0.9**(epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        logger.info('decay_rate is set to %s', decay_rate)

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def load_data(mod = 'train', dataset = 'tid2013', worker_idx = 0, logger=None): #worker_idx is the index of the selected task

    if dataset == 'tid2013':
        data_dir = os.path.join('./tid2013')
        worker_orignal = pd.read_csv(os.path.join(data_dir, 'image_labeled_by_per_noise.csv'), sep=',')
        image_path = './tid2013/distorted_images/'
    else:
        data_dir = os.path.join('./kadid10k')
        worker_orignal = pd.read_csv(os.path.join(data_dir, 'image_labeled_by_per_noise.csv'), sep=',')
        image_path = './kadid10k/images/'
    workers_fold = "noise/"
    if not os.path.exists(workers_fold):
        os.makedirs(workers_fold)
    # print(worker_orignal['noise'].unique())#[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
    worker = worker_orignal['noise'].unique()[worker_idx] # .unique() removes duplicate elements, worker represents the selected task index
    logger.info("----worker number: %2d---- %s", worker_idx, worker)
    if mod == 'train':
        percent = 0.8
        images = worker_orignal[worker_orignal['noise'].isin([worker])][['image', 'dmos']]

        train_dataframe, valid_dataframe = train_test_split(images, train_size=percent)
        train_path = workers_fold + "train_scores_" + str(worker) + ".csv"
        test_path = workers_fold + "test_scores_" + str(worker) + ".csv"
        train_dataframe.to_csv(train_path, sep=',', index=False)
        valid_dataframe.to_csv(test_path, sep=',', index=False)

        output_size = (224, 224)
        transformed_dataset_train = ImageRatingsDataset(csv_file=train_path,
                                                        root_dir=image_path,
                                                        transform=transforms.Compose([Rescale(output_size=(256, 256)),
                                                                                      RandomHorizontalFlip(0.5),
                                                                                      RandomCrop(
                                                                                          output_size=output_size),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        transformed_dataset_valid = ImageRatingsDataset(csv_file=test_path,
                                                        root_dir=image_path,
                                                        transform=transforms.Compose([Rescale(output_size=(224, 224)),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        dataloader_train = DataLoader(transformed_dataset_train, batch_size=75,
                                  shuffle=False, num_workers=0, collate_fn=my_collate)
        dataloader_valid = DataLoader(transformed_dataset_valid, batch_size=50,
                                      shuffle=False, num_workers=0, collate_fn=my_collate)
    else:
        cross_data_path = './LIVE_WILD/image_labeled_by_score.csv'
        transformed_dataset_valid_1 = ImageRatingsDataset(csv_file=cross_data_path,
                                                        root_dir='./ChallengeDB_release/Images',
                                                        transform=transforms.Compose([Rescale(output_size=(224, 224)),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        dataloader_train = 0
        dataloader_valid = DataLoader(transformed_dataset_valid_1, batch_size= 50,
                                        shuffle=False, num_workers=0, collate_fn=my_collate)

    return dataloader_train, dataloader_valid


train_model()
