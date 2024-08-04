# The Lottery Ticket Hypothesis in Denoising: Towards Semantic-Driven Initialization (ECCV 2024)

## [<a href="https://ut-mao.github.io/noise.github.io/" target="_blank">Project Page</a>] [<a href="https://arxiv.org/abs/2312.08872" target="_blank">Paper</a>]

Refer to our previous work for more discussion about initial noise in diffusion!
- [<a href="https://ut-mao.github.io/swap.github.io/" target="_blank">Guided Image Synthesis via Initial Image Editing in Diffusion Model</a>] 

![teaser](./teaser.png)


## Setup

Our codebase is built on [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
and has shared dependencies and model architecture.

### Creating a Conda Environment

```
conda env create -f environment.yaml
conda activate ldm
```

### Downloading StableDiffusion Weights

Download the StableDiffusion weights from the [CompVis organization at Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
(download the `sd-v1-4.ckpt` file), and link them:
```
mkdir -p models/ldm/stable-diffusion-v1/
ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt 
```
## Hands on

Play with [hands-on](./Hands_on.ipynb) to try our approach right away, refer to [utils.py](./utils.py) for the implementation.

## Citation
```
@article{mao2024theLottery,
  title={The Lottery Ticket Hypothesis in Denoising: Towards Semantic-Driven Initialization},
  author={Mao, Jiafeng and Wang, Xueting and Aizawa, Kiyoharu},
  journal={ECCV},
  year={2024}
}
```
