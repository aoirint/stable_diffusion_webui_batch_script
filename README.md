# Stable Diffusion WebUI Batch Generation Script

## Usage

```yaml
jobs:
  - sd_model_checkpoint: "animagineXL40_v4Zero.safetensors [f15812e65c]"
    sd_vae: "sdxl_vae.safetensors"
    prompt: |-
      1girl, solo
      masterpiece, high score, great score, absurdres
    negative_prompt: |-
      lowres, bad anatomy, bad hands, text, error, missing finger, extra digits
      fewer digits, cropped, worst quality, low quality, low score, bad score
      average score, signature, watermark, username, blurry
    steps: 28
    width: 1024
    height: 1024
    cfg_scale: 5
    sampling_method: "Euler a"
    scheduler: "Karras"
    seed: -1
    num_images: 100
```

```shell
uv sync

uv run python -m stable_diffusion_webui_batch_script --config_file work/a.yaml
```
