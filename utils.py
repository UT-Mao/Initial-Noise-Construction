import torch
import random

import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torch import autocast
from ldm.util import instantiate_from_config
from itertools import combinations


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def sampling(model, sampler, prompt, n_samples, scale=7.5, steps=50, conjunction=False, mask_cond=None, img=None):
    H = W = 512
    C = 4
    f = 8
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                for n in range(n_samples):
                    for bid, p in enumerate(prompt):
                        
                        uc = model.get_learned_conditioning([""])
                        _c = model.get_learned_conditioning(p)
                        c = {'k': [_c], 'v': [_c]}
                        shape = [C, H // f, W // f]
                        
                        samples_ddim, _ = sampler.sample(S=steps,
                                                            conditioning=c,
                                                            batch_size=1,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=uc,
                                                            eta=0.0,
                                                            x_T=img,
                                                            quiet=True,
                                                            mask_cond = mask_cond,
                                                            save_attn_maps=True)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image = x_samples_ddim
                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                        all_samples.append(x_checked_image_torch)
    return all_samples

def diff(t1, t2):
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    return difference

def intersection(t1, t2):
    i = np.intersect1d(t1, t2)
    return torch.from_numpy(i) 

def block(value, scale_factor=4):
    vs = []
    for v in value:
        e = torch.zeros(256)
        e[v] = 1
        e = rearrange(e, '(w h)-> w h', w=16)
        e_resized = F.interpolate(e.reshape(1,1,16,16), scale_factor=scale_factor)[0][0]
        e_resized = rearrange(e_resized, 'w h -> (w h)')
        vs.append(torch.where(e_resized==1)[0])
    return vs

def image_to_blocks(img):
    # input: [1,4,64,64] image
    # output: list of blocks, lenth is 256
    # block : [1, 4, 4, 4]
    blocks = []
    for i in range(16):
        for j in range(16):
            block = img[:, :, i * 4: (i + 1) * 4, j * 4: (j + 1) * 4]
            blocks.append(block)
    return blocks

def generate(model, sampler, img_, prompt, ind=None):
    mask_cond = {
        'is_use': False,
    }
    ddim_steps = 50
    n_samples = 1
    scale = 7.5
    all_samples = sampling(model, sampler, prompt, 
                            n_samples, scale, 
                            ddim_steps, mask_cond=mask_cond, conjunction=False, img=img_)
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=int(np.sqrt(n_samples)))
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    img = Image.fromarray(grid.astype(np.uint8))
    attn_maps = [item[0][0] for item in sampler.attn_maps['input_blocks.8.1.transformer_blocks.0.attn2']]
    maps = [torch.mean(item, axis=0) for item in attn_maps]
    maps = [rearrange(item, 'w h d -> d w h')[None,:] for item in maps]
    maps = rearrange(torch.cat(maps,dim=0), 't word w h -> word t w h')
    if ind is not None:
        plt.subplot(1, 5, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 5, 2)
        plt.imshow(maps[ind[0]][0],cmap = 'gray')
        plt.axis("off")
        plt.subplot(1, 5, 3)
        plt.imshow(maps[ind[1]][0],cmap = 'gray')
        plt.axis("off")
        plt.subplot(1, 5, 4)
        plt.imshow(maps[ind[0]][-1],cmap = 'gray')
        plt.axis("off")
        plt.subplot(1, 5, 5)
        plt.imshow(maps[ind[1]][-1],cmap = 'gray')
        plt.axis("off")
        plt.show()
    else:
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        
def preprocess_prompts(prompts):
    if isinstance(prompts, (list, tuple)):
        return [p.lower().strip().strip(".").strip() for p in prompts]
    elif isinstance(prompts, str):
        return prompts.lower().strip().strip(".").strip()
    else:
        raise NotImplementedError
    
def intersection(t1, t2):
    i = np.intersect1d(t1, t2)
    return torch.from_numpy(i) 

def attention_to_score(attns):
    # input: [16, 16] attention maps
    # out put: list of score
    scores = []
    for attn in attns:
        scores.append(rearrange(attn, 'w h -> (w h)').tolist())
    return scores

def score_normalize(scores):
    std = torch.std(scores, unbiased=False)
    mean = torch.mean(scores) 
    scores = (scores - mean)/std
    scores = (scores - torch.min(scores)) / (torch.max(scores) - torch.min(scores))
    return scores


class pixel_block_base:
    def __init__(self, model, sampler, labels):
        self.model = model
        self.sampler = sampler
        self.H = 512
        self.W = 512
        self.C = 4
        self.f = 8
        self.normalize = False
        self.shape = [1, self.C, self.H // self.f, self.W // self.f]
        self.cond = {'is_use': False}
        
        self.base = {}
        self.base['blocks'] = []
        for w in labels:
            self.base[w] = {}
        
        self.labels = labels
        self.combinations = list(combinations(labels, 2))
        for pair in self.combinations:
            self.base[pair[0]][pair[1]] = torch.tensor([])
            self.base[pair[1]][pair[0]] = torch.tensor([])
        
        self.prompt = []
        for pair in self.combinations:
            self.prompt.append('a ' + pair[0] + ' and a ' + pair[1] + '.') 
        
    def _get_attention(self, prompt, img, scale=7.5, steps=50):
        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    for bid, p in enumerate(prompt):
                        p = preprocess_prompts(p)
                        uc = self.model.get_learned_conditioning([""])
                        kv = self.model.get_learned_conditioning(p[0])
                        c = {'k':[kv], 'v': [kv]}
                        shape = [self.C, self.H // self.f, self.W // self.f]
                        self.sampler.get_attention(S=steps,
                                        conditioning=c,
                                        batch_size=1,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc,
                                        eta=0.0,
                                        x_T=img,
                                        quiet=True,
                                        mask_cond=self.cond,
                                        save_attn_maps=True)
        all_attn_maps = [item[0][0] for item in self.sampler.attn_maps['input_blocks.8.1.transformer_blocks.0.attn2']]
        avg_maps = [torch.mean(item, axis=0) for item in all_attn_maps]
        avg_maps = [rearrange(item, 'w h d -> d w h')[None,:] for item in avg_maps]
        avg_maps = rearrange(torch.cat(avg_maps,dim=0), 't word w h -> word t w h')
        return avg_maps
    
    
    def make_base(self, n_img):
        for i in range(n_img):
            img_ = torch.randn(self.shape).cuda()
            _blocks = image_to_blocks(img_.clone().cpu())
            self.base['blocks'].append(torch.stack(_blocks, dim=0))
            for p in range(len(self.prompt)):
                caption = [[self.prompt[p]]]
                w = self.combinations[p]
                
                avg_maps = self._get_attention(caption, img_)
                
                m = [avg_maps[2][0]]
                _scores = attention_to_score(m)[0]
                self.base[w[0]][w[1]] = torch.cat([self.base[w[0]][w[1]], score_normalize(torch.tensor(_scores))], dim=0)
                m = [avg_maps[5][0]]
                _scores = attention_to_score(m)[0]
                self.base[w[1]][w[0]] = torch.cat([self.base[w[1]][w[0]], score_normalize(torch.tensor(_scores))], dim=0)
        self.base['blocks'] = torch.cat(self.base['blocks'], dim=0)
        
        # 对于每个类别，算一次总分， a 在与 b，c，d。。等类别同时出现时的得分相加
        for k in self.labels:
            avg = 0
            counter = 0
            for w in self.base[k].keys():
                counter = counter + 1 
                avg = avg + self.base[k][w]
            self.base[k]['average'] = avg / counter
    
    def make_base_by_list(self, n_img, comb_list):
        # comb_list : ['a', 'b']
        
        for i in trange(n_img):
            img_ = torch.randn(self.shape).cuda()
            _blocks = image_to_blocks(img_.clone().cpu())
            self.base['blocks'].append(torch.stack(_blocks, dim=0))
            for p in trange(len(comb_list)):
                caption = [['A ' + comb_list[p][0] + ' and a ' + comb_list[p][1]+ '.'] ]
                w = comb_list[p]
                avg_maps = self._get_attention(caption, img_)
                
                m = [avg_maps[2][0]]
                _scores = attention_to_score(m)[0]
                self.base[w[0]][w[1]] = torch.cat([self.base[w[0]][w[1]], score_normalize(torch.tensor(_scores))], dim=0)

                m = [avg_maps[5][0]]
                _scores = attention_to_score(m)[0]
                self.base[w[1]][w[0]] = torch.cat([self.base[w[1]][w[0]], score_normalize(torch.tensor(_scores))], dim=0)
        self.base['blocks'] = torch.cat(self.base['blocks'], dim=0)
                
            
        
    def generate_region_mask(self, regions):
        region_mask = []
        for r in regions:
            z = torch.zeros(16,16)
            z[r[1]:r[3], r[0]:r[2]] = 1
            region_mask.append(z)
        region_mask = rearrange(torch.stack(region_mask, dim = 0), 'n w h -> n (w h)')
        if len(regions) == 2:
            mask_1 = torch.where(region_mask[0] == 1)[0]
            mask_2 = torch.where(region_mask[1] == 1)[0]
            if mask_1.shape[0] > mask_2.shape[0]:
                region_mask[0][intersection(mask_1, mask_2)] = 0
            else:
                region_mask[1][intersection(mask_1, mask_2)] = 0
        if len(regions) == 3:
            mask_1 = torch.where(region_mask[0] == 1)[0]
            mask_2 = torch.where(region_mask[1] == 1)[0]
            mask_3 = torch.where(region_mask[2] == 1)[0]
            if mask_1.shape[0] > mask_2.shape[0]:
                region_mask[0][intersection(mask_1, mask_2)] = 0
            else:
                region_mask[1][intersection(mask_1, mask_2)] = 0
                
            if mask_1.shape[0] > mask_3.shape[0]:
                region_mask[0][intersection(mask_1, mask_3)] = 0
            else:
                region_mask[2][intersection(mask_1, mask_3)] = 0
                
            if mask_2.shape[0] > mask_3.shape[0]:
                region_mask[1][intersection(mask_2, mask_3)] = 0
            else:
                region_mask[2][intersection(mask_2, mask_3)] = 0
                
        bg_mask =  1 - region_mask.sum(axis=0)
        return region_mask, bg_mask

    def make_img(self, region_mask, bg_mask, obj_blocks, bg_blocks, recalibration):
        # input : masks indicating locations and blocks selected for corresponding contents
        img = torch.zeros([1,4,64,64])
        for i in range(region_mask.shape[0]):
            num = torch.where(region_mask[i] != 0)[0].shape[0]
            r = rearrange(region_mask[i], '(w h) -> w h', w=16)
            total_num = obj_blocks[i].shape[0]
            if num > total_num:
                sampled_index = []
                while num > total_num:
                    sampled_index = sampled_index + random.sample(range(0, total_num), total_num)
                    num = num - total_num
                sampled_index = sampled_index + random.sample(range(0, total_num), num)
            else:
                sampled_index = random.sample(range(0, total_num), num)
            positions = (r == 1).nonzero(as_tuple=False)
            selected_blocks_obj = obj_blocks[i][sampled_index]
            
            # recalibration
            if recalibration:
                print(selected_blocks_obj.mean())
                print((selected_blocks_obj - selected_blocks_obj.mean()).pow(2).mean())
            
            for j in range(len(positions)): 
                p, q = positions[j]
                img[:, :, 4 * p : 4 * p + 4, 4 * q : 4 * q + 4] = selected_blocks_obj[j]
        
        bg_num = torch.where(bg_mask != 0)[0].shape[0]
        r = rearrange(bg_mask, '(w h) -> w h', w=16)
        bg_total_num = bg_blocks.shape[0]
        if bg_num > bg_total_num:
            sampled_index = []
            while bg_num > bg_total_num:
                sampled_index = sampled_index + random.sample(range(0, bg_total_num), bg_total_num)
                bg_num = bg_num - bg_total_num
            sampled_index = sampled_index + random.sample(range(0, bg_total_num), bg_num)
        else:
            sampled_index = random.sample(range(0, bg_total_num), bg_num)
        bg_positions = (r == 1).nonzero(as_tuple=False)
        selected_blocks_bg = bg_blocks[sampled_index]
        for i in range(len(bg_positions)):
            p, q = bg_positions[i]
            img[:, :, 4 * p : 4 * p + 4, 4 * q : 4 * q + 4] = selected_blocks_bg[i]
        return img
    
    def product_image(self, words, regions, t_pos_1 = 0.5, t_bg_1 = 0.3, t_pos_2 = 0.3, t_neg_2 = 0.3, t_bg_2 = 0.1, recalibration = False):
        if len(words) == 1:
            # Fetch pre-collected pixel blocks and their scores
            word = words[0]
            scores = self.base[word]['average']
            blocks = self.base['blocks']
            # select blocks for obj 
            # threshold [TO DO: sort?]
            blocks_index = torch.where(scores > t_pos_1)[0].numpy()
            # select blocks for bg
            blocks_index_bg = torch.where(scores < t_bg_1)[0].numpy()
            obj_blocks = [blocks[blocks_index]]
            bg_blocks = blocks[blocks_index_bg]
            region_mask, bg_mask = self.generate_region_mask(regions)
            img = self.make_img(region_mask, bg_mask, obj_blocks, bg_blocks, recalibration)
            
        elif len(words) == 2:
            score_1 = self.base[words[0]][words[1]]
            score_2 = self.base[words[1]][words[0]]
            blocks = self.base['blocks']
            # for class 1 : 
            blocks_index_1 = intersection(torch.where(score_1 > t_pos_2)[0].numpy(), torch.where(score_2 < t_neg_2)[0].numpy())
            # for class 2 : 
            blocks_index_2 = intersection(torch.where(score_2 > t_pos_2)[0].numpy(), torch.where(score_1 < t_neg_2)[0].numpy())
            # for background : 
            blocks_index_bg = intersection(torch.where(score_1 < t_bg_2)[0].numpy(), torch.where(score_2 < t_bg_2)[0].numpy())
            obj_blocks = [blocks[blocks_index_1], blocks[blocks_index_2]]
            bg_blocks = blocks[blocks_index_bg]
            region_mask, bg_mask = self.generate_region_mask(regions)
            img = self.make_img(region_mask, bg_mask, obj_blocks, bg_blocks, recalibration)
        
        elif len(words) == 3:
            score_1_2 = self.base[words[0]][words[1]]
            score_1_3 = self.base[words[0]][words[2]]
            # for class 1 : 
            blocks_index_1 = intersection(torch.where(score_1_2 > t_pos_2)[0].numpy(), torch.where(score_1_3 > t_pos_2)[0].numpy())
            
            score_2_1 = self.base[words[1]][words[0]]
            score_2_3 = self.base[words[1]][words[2]]
            # for class 2 : 
            blocks_index_2 = intersection(torch.where(score_2_1 > t_pos_2)[0].numpy(), torch.where(score_2_3 > t_pos_2)[0].numpy())
            
            score_3_1 = self.base[words[2]][words[0]]
            score_3_2 = self.base[words[2]][words[1]]
            # for class 2 : 
            blocks_index_3 = intersection(torch.where(score_3_1 > t_pos_2)[0].numpy(), torch.where(score_3_2 > t_pos_2)[0].numpy())
            
            blocks = self.base['blocks']
            score_1 = self.base[words[0]]['average']
            score_2 = self.base[words[1]]['average']
            score_3 = self.base[words[2]]['average']
            
            # for background : 
            blocks_index_bg = intersection(torch.where(score_1 < t_bg_2)[0].numpy(), torch.where(score_2 < t_bg_2)[0].numpy())
            blocks_index_bg = intersection(blocks_index_bg, torch.where(score_3 < t_bg_2)[0].numpy())
            
            obj_blocks = [blocks[blocks_index_1], blocks[blocks_index_2], blocks[blocks_index_3]]
            bg_blocks = blocks[blocks_index_bg]
            region_mask, bg_mask = self.generate_region_mask(regions)
            img = self.make_img(region_mask, bg_mask, obj_blocks, bg_blocks, recalibration)
        return img, region_mask