import asyncio
import base64
import csv
import logging
import random
import re
import shutil
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import BaseModel

logger = logging.getLogger("stable_diffusion_webui_batch_script")


async def set_model_and_vae(
    sd_model_checkpoint: str,
    sd_vae: str,
) -> None:
    payload = {
        "sd_model_checkpoint": sd_model_checkpoint,
        "sd_vae": sd_vae,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://127.0.0.1:7860/sdapi/v1/options",
            json=payload,
        )
        response.raise_for_status()

    logger.info(f"✅ モデル設定: {sd_model_checkpoint}, VAE: {sd_vae}")


def parse_sd_model_checkpoint(
    sd_model_checkpoint: str,
) -> tuple[str, str]:
    m = re.search(r"^(.+?)\[(.+?)\]$", sd_model_checkpoint)
    if not m:
        raise ValueError(f"Invalid sd_model_checkpoint format: {sd_model_checkpoint}")

    sd_model_name = m.group(1)
    sd_model_hash = m.group(2)

    return sd_model_name, sd_model_hash


async def generate_job(
    sd_model_checkpoint: str,
    sd_vae: str,
    prompt: str,
    negative_prompt: str,
    sampling_method: str,
    scheduler: str,
    steps: int,
    width: int,
    height: int,
    cfg_scale: float,
    seed: int,
    num_images: int,
    output_dir: Path,
) -> None:
    await set_model_and_vae(
        sd_model_checkpoint=sd_model_checkpoint,
        sd_vae=sd_vae,
    )

    sd_model_name, sd_model_hash = parse_sd_model_checkpoint(
        sd_model_checkpoint=sd_model_checkpoint,
    )

    log_file = output_dir / "log.csv"
    with log_file.open("w", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "prompt",
                "seed",
                "width",
                "height",
                "sampler",
                "cfgs",
                "steps",
                "filename",
                "negative_prompt",
                "sd_model_name",
                "sd_model_hash",
            ]
        )

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    logger.info(f"🪄 シード: {seed}")
    logger.info(f"🖼 生成枚数: {num_images}")

    for image_index in range(num_images):
        current_seed = seed + image_index
        output_name = f"{image_index:05}-{current_seed}.png"

        payload: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "sampler_name": sampling_method,
            "scheduler": scheduler,
            "steps": steps,
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "seed": current_seed,
            "batch_size": 1,
        }

        logger.info(f"🖼 生成中 {output_name}")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://127.0.0.1:7860/sdapi/v1/txt2img",
                json=payload,
                timeout=httpx.Timeout(
                    10,
                    read=None,
                ),
            )
            response.raise_for_status()
        result = response.json()

        img_data = result["images"][0]
        img_binary = base64.b64decode(img_data.split(",", 1)[-1])

        output_file = output_dir / output_name
        with output_file.open("wb") as fp:
            fp.write(img_binary)

        with log_file.open("a", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    prompt,
                    str(current_seed),
                    str(width),
                    str(height),
                    sampling_method,
                    str(cfg_scale),
                    str(steps),
                    output_name,
                    negative_prompt,
                    sd_model_name,
                    sd_model_hash,
                ]
            )


class ConfigJob(BaseModel):
    sd_model_checkpoint: str
    sd_vae: str
    prompt: str
    negative_prompt: str
    sampling_method: str
    scheduler: str
    steps: int
    width: int
    height: int
    cfg_scale: float
    seed: int
    num_images: int


class Config(BaseModel):
    jobs: list[ConfigJob]


async def run_batch_from_config(config: Config, output_dir: Path) -> None:
    for job in config.jobs:
        await generate_job(
            sd_model_checkpoint=job.sd_model_checkpoint,
            sd_vae=job.sd_vae,
            prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            sampling_method=job.sampling_method,
            scheduler=job.scheduler,
            steps=job.steps,
            width=job.width,
            height=job.height,
            cfg_scale=job.cfg_scale,
            seed=job.seed,
            num_images=job.num_images,
            output_dir=output_dir,
        )


async def run_batch_from_yaml(config_file: Path, base_output_dir: Path) -> None:
    with config_file.open("r", encoding="utf-8") as fp:
        config_dict = yaml.safe_load(fp)

    config = Config.model_validate(config_dict)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = base_output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # YAMLファイルを出力先にコピー
    copied_config_file = output_dir / "config.yml"
    shutil.copy(config_file, copied_config_file)
    logger.info(f"📄 YAML設定ファイルをコピーしました: {copied_config_file}")

    await run_batch_from_config(
        config=config,
        output_dir=output_dir,
    )


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = ArgumentParser(description="Stable Diffusion WebUI Batch Script")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="YAML configuration file path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for generated images",
    )
    args = parser.parse_args()

    config_file_string: str = args.config_file
    config_file: Path = Path(config_file_string)

    output_dir_string: str = args.output_dir
    output_dir: Path = Path(output_dir_string)

    if not config_file.exists():
        logger.error(f"⚠ YAMLファイルが見つかりません: {config_file}")
        return

    await run_batch_from_yaml(
        config_file=config_file,
        base_output_dir=output_dir,
    )


if __name__ == "__main__":
    asyncio.run(main())
