# FoleyCrafter (fork) 
### Official original: https://foleycrafter.github.io/

Sound effects are the unsung heroes of cinema and gaming, enhancing realism, impact, and emotional depth for an immersive audiovisual experience. **FoleyCrafter** is a video-to-audio generation framework which can produce realistic sound effects semantically relevant and synchronized with videos.

**Your star is our fuel! <img alt="" width="30" src="https://camo.githubusercontent.com/2f4f0d02cdf79dc1ff8d2b053b4410b13bc2e39cbc8a96fcdc6f06538a3d6d2b/68747470733a2f2f656d2d636f6e74656e742e7a6f626a2e6e65742f736f757263652f616e696d617465642d6e6f746f2d636f6c6f722d656d6f6a692f3335362f736d696c696e672d666163652d776974682d6865617274735f31663937302e676966"> We're revving up the engines with it! <img alt="" width="30" src="https://camo.githubusercontent.com/028a75f875b8c3aa1b3c80bbf7dd27973c4bb654fffcf0bdc0b6f1b0674ce481/68747470733a2f2f656d2d636f6e74656e742e7a6f626a2e6e65742f736f757263652f74656c656772616d2f3338362f737061726b6c65735f323732382e77656270">**


[FoleyCrafter: Bring Silent Videos to Life with Lifelike and Synchronized Sounds]()

[Yiming Zhang](https://github.com/ymzhang0319),
[Yicheng Gu](https://github.com/VocodexElysium),
[Yanhong Zeng†](https://zengyh1900.github.io/),
[Zhening Xing](https://github.com/LeoXing1996/),
[Yuancheng Wang](https://github.com/HeCheng0625),
[Zhizheng Wu](https://drwuz.com/),
[Kai Chen†](https://chenkai.site/)

(†Corresponding Author)


## What's New
- [ ] Release training code.
- [x] `2024/07/01` Release the model and code of FoleyCrafter.

## Setup

### Requirements
```bash
Python >= 3.10
```

### Prepare Environment
Use the following command to install dependencies:
```bash
# create a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
\.venv\Scripts\activate     # Windows

# install GIT LFS for checkpoints download
git lfs install
```

### Install FFmpeg
`convert_videos.py` relies on an FFmpeg binary. Install one for your platform:

```bash
# Debian/Ubuntu
sudo apt-get update && sudo apt-get install ffmpeg

# macOS with Homebrew
brew install ffmpeg

# Windows with winget
winget install ffmpeg
```

### Download Checkpoints
The checkpoints will be downloaded automatically by running `inference.py`.

You can also download manually using following commands.
<li> Download FoleyCrafter</li>

```bash
git clone https://huggingface.co/ymzhang319/FoleyCrafter checkpoints/
```

<li> Download the text-to-audio base model. We use Auffusion</li>

```bash
git clone https://huggingface.co/auffusion/auffusion-full-no-adapter checkpoints/auffusion
```

Put checkpoints as follows:
```
└── checkpoints
    ├── semantic
    │   ├── semantic_adapter.bin
    ├── vocoder
    │   ├── vocoder.pt
    │   ├── config.json
    ├── temporal_adapter.ckpt
    │   │
    └── timestamp_detector.pth.tar
```

## Dataset preparation

Place your raw clips in a folder named `clips/` in the project root. Running
`convert_videos.py` will automatically scan this directory (or a custom
`--input_dir`) and write the processed dataset into
`data/AudioSetStrong/train` (or a custom `--output_dir`):

```bash
# extract frames and log‑mel features for every clip in examples/vggsound
python convert_videos.py \
  --input_dir=examples/vggsound \
  --output_dir=/tmp/sample_dataset
```

By default the script trims each clip to 150 frames; adjust with `--frames`.

Each clip is trimmed to the first 150 frames, then probed to gather its fps,
duration, codecs and other metadata. The frames and a log‑mel spectrogram are
saved under `video/` and `feature/` subfolders so the resulting directory can be
passed directly to `train_time_detector.py`.

## Training

A lightweight CLI for fine-tuning the time detection module is available:

```bash
python train_time_detector.py \
  --data_path=data/AudioSetStrong/train/feature \
  --video_path=data/AudioSetStrong/train/video \
  --output_dir=checkpoints/time_detector_finetune
```

Pass `--resume` with a checkpoint path to continue training from a previous run.

## Gradio demo

You can launch the Gradio interface for FoleyCrafter by running the following command:

```bash
python app.py
```



## Inference
### Video To Audio Generation
```bash
python inference.py --save_dir=output/sora/
```

- Temporal Alignment with Visual Cues
```bash
python inference.py \
--temporal_align \
--input=input/avsync \
--save_dir=output/avsync/
```

### Text-based Video to Audio Generation

- Using Prompt

```bash
# case1
python inference.py \
--input=input/PromptControl/case1/ \
--seed=10201304011203481429 \
--save_dir=output/PromptControl/case1/

python inference.py \
--input=input/PromptControl/case1/ \
--seed=10201304011203481429 \
--prompt='noisy, people talking' \
--save_dir=output/PromptControl/case1_prompt/

# case2
python inference.py \
--input=input/PromptControl/case2/ \
--seed=10021049243103289113 \
--save_dir=output/PromptControl/case2/

python inference.py \
--input=input/PromptControl/case2/ \
--seed=10021049243103289113 \
--prompt='seagulls' \
--save_dir=output/PromptControl/case2_prompt/
```

- Using Negative Prompt
```bash
# case 3
python inference.py \
--input=input/PromptControl/case3/ \
--seed=10041042941301238011 \
--save_dir=output/PromptControl/case3/

python inference.py \
--input=input/PromptControl/case3/ \
--seed=10041042941301238011 \
--nprompt='river flows' \
--save_dir=output/PromptControl/case3_nprompt/

# case4
python inference.py \
--input=input/PromptControl/case4/ \
--seed=10014024412012338096 \
--save_dir=output/PromptControl/case4/

python inference.py \
--input=input/PromptControl/case4/ \
--seed=10014024412012338096 \
--nprompt='noisy, wind noise' \
--save_dir=output/PromptControl/case4_nprompt/

```

### Commandline Usage Parameters
```console
options:
  -h, --help            show this help message and exit
  --prompt PROMPT       prompt for audio generation
  --nprompt NPROMPT     negative prompt for audio generation
  --seed SEED           ramdom seed
  --temporal_align TEMPORAL_ALIGN
                        use temporal adapter or not
  --temporal_scale TEMPORAL_SCALE
                        temporal align scale
  --semantic_scale SEMANTIC_SCALE
                        visual content scale
  --input INPUT         input video folder path
  --ckpt CKPT           checkpoints folder path
  --save_dir SAVE_DIR   generation result save path
  --pretrain PRETRAIN   generator checkpoint path
  --device DEVICE
```


## BibTex
```
@misc{zhang2024pia,
  title={FoleyCrafter: Bring Silent Videos to Life with Lifelike and Synchronized Sounds},
  author={Yiming Zhang, Yicheng Gu, Yanhong Zeng, Zhening Xing, Yuancheng Wang, Zhizheng Wu, Kai Chen},
  year={2024},
  eprint={2407.01494},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```


## Contact Us

**Yiming Zhang**: zhangyiming@pjlab.org.cn

**YiCheng Gu**: yichenggu@link.cuhk.edu.cn

**Yanhong Zeng**: zengyanhong@pjlab.org.cn

## LICENSE
Please check [LICENSE](./LICENSE) for the part of FoleyCrafter for details.
If you are using it for commercial purposes, please check the license of the [Auffusion](https://github.com/happylittlecat2333/Auffusion).

## Acknowledgements
The code is built upon [Auffusion](https://github.com/happylittlecat2333/Auffusion), [CondFoleyGen](https://github.com/XYPB/CondFoleyGen) and [SpecVQGAN](https://github.com/v-iashin/SpecVQGAN).

We recommend a toolkit for Audio, Music, and Speech Generation [Amphion](https://github.com/open-mmlab/Amphion) :gift_heart:.
