import json
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data import inspect_data, prepare_data


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    if cfg.stat == "prepare":
        result = prepare_data(config["data"])
    elif cfg.stat == "inspect":
        result = inspect_data(config["data"], config["inspect"])
    else:
        raise ValueError(f"未知任务 stat={cfg.stat}，目前支持 prepare、inspect")

    logging.info("\n%s", json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("issues"):
        raise ValueError("数据检查未通过，具体原因见上方 issues 和 inspection.json")


if __name__ == "__main__":
    main()
