"""统一的终端进度显示，不改变训练或采样逻辑。"""

from tqdm import tqdm


def progress_bar(iterable=None, *, config=None, **kwargs):
    config = config or {}
    return tqdm(iterable, disable=not config.get("enabled", True),
                mininterval=config.get("min_interval", 0.5), dynamic_ncols=True, **kwargs)
