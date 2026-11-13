import json
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data import inspect_data, prepare_data
from src.tokenizer_tasks import check_tokenizer, train_tokenizer


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    if cfg.stat == "prepare":
        result = prepare_data(config["data"])
    elif cfg.stat == "inspect":
        result = inspect_data(config["data"], config["inspect"])
    elif cfg.stat == "tokenizer":
        result = train_tokenizer(config["data"], config["tokenizer"])
    elif cfg.stat == "tokenizer_check":
        result = check_tokenizer(config["data"], config["tokenizer"])
    elif cfg.stat == "attention":
        from src.attention_tasks import run_attention
        result = run_attention(config["attention"], config["progress"])
    elif cfg.stat in ("science", "science_prepare", "science_analyze"):
        from src.science_tasks import run_science
        result = run_science(config, cfg.stat)
    elif cfg.stat in ("benchmark", "train", "eval", "generate", "plot"):
        # 延迟导入，数据与分词器任务不要求安装 PyTorch。
        from hydra.core.hydra_config import HydraConfig
        from src.training import run_task
        result = run_task(config, cfg.stat, HydraConfig.get().runtime.output_dir)
    else:
        raise ValueError(f"未知任务 stat={cfg.stat}，支持 prepare、inspect、tokenizer、tokenizer_check、benchmark、train、eval、generate、plot、attention、science、science_prepare、science_analyze")

    logging.info("\n%s", json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("issues"):
        raise ValueError("检查未通过，具体原因见上方 issues 和检查报告")


if __name__ == "__main__":
    main()
