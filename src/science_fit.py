"""固定训练 token 预算下的参数量幂律拟合，不估计 Chinchilla 最优算力。"""

import numpy as np


def predict(fit, parameters):
    return fit["floor"] + fit["amplitude"] * (np.asarray(parameters) / 1e6) ** -fit["alpha"]


def fit_power_law(parameters, losses, options):
    n, y = np.asarray(parameters, dtype=float), np.asarray(losses, dtype=float)
    if len(n) < 3 or len(set(n)) != len(n) or len(n) != len(y):
        raise ValueError("拟合至少需要三个不同参数量及对应 loss")
    if not np.isfinite(n).all() or not np.isfinite(y).all() or (n <= 0).any() or (y <= 0).any():
        raise ValueError("参数量和 loss 必须有限且为正")
    alphas = np.linspace(options["alpha_min"], options["alpha_max"], options["grid_points"])
    x = (n[None, :] / 1e6) ** -alphas[:, None]
    xc, yc = x - x.mean(axis=1, keepdims=True), y - y.mean()
    amplitude = np.maximum(0, (xc * yc).sum(axis=1) / (xc * xc).sum(axis=1))
    floor = y.mean() - amplitude * x.mean(axis=1)
    floor = np.clip(floor, 0, y.min())
    amplitude = np.maximum(0, (x * (y[None, :] - floor[:, None])).sum(axis=1) / (x * x).sum(axis=1))
    predictions = floor[:, None] + amplitude[:, None] * x
    squared_error = ((predictions - y) ** 2).sum(axis=1)
    index = int(squared_error.argmin())
    order = np.argsort(n)
    warnings = []
    if index in (0, len(alphas) - 1):
        warnings.append("alpha 位于搜索边界，外推可能不稳定")
    if (np.diff(y[order]) > 0).any():
        warnings.append("实测均值并非随参数量单调下降，幂律假设支持较弱")
    variance = float(((y - y.mean()) ** 2).sum())
    if variance < 1e-12:
        warnings.append("loss 几乎恒定，指数不可辨识")
    return dict(formula="L(N)=floor+amplitude*(N/1e6)^(-alpha)",
                floor=float(floor[index]), amplitude=float(amplitude[index]), alpha=float(alphas[index]),
                rmse=float(np.sqrt(squared_error[index] / len(n))),
                r_squared=1 - float(squared_error[index]) / variance if variance >= 1e-12 else None,
                warnings=warnings)


def summarize_scaling(rows, model_names, holdout_name, options, seed):
    # 名单明确排除 holdout，而不是把它一起拟合后再声称外推。
    groups = {name: sorted((r for r in rows if r["model"] == name and r["corpus"] == "clean"),
                           key=lambda r: r["seed"]) for name in model_names}
    seed_sets = [{r["seed"] for r in group} for group in groups.values()]
    if not all(groups.values()) or any(s != seed_sets[0] for s in seed_sets):
        raise ValueError("Scaling 模型缺少相同随机种子的完整结果")
    sizes = [group[0]["parameters"] for group in groups.values()]
    if any(len({r["parameters"] for r in group}) != 1 for group in groups.values()):
        raise ValueError("同名模型参数量不一致")
    losses = np.asarray([[r["final_val_loss"] for r in group] for group in groups.values()])
    fit = fit_power_law(sizes, losses.mean(axis=1), options)
    held = sorted((r for r in rows if r["model"] == holdout_name and r["corpus"] == "clean"), key=lambda r: r["seed"])
    if {r["seed"] for r in held} != seed_sets[0] or len({r["parameters"] for r in held}) != 1:
        raise ValueError("外推验证模型结果不完整")
    target = held[0]["parameters"]
    if target <= max(sizes):
        raise ValueError("外推目标参数量必须大于拟合模型")
    predicted = float(predict(fit, target))
    actual = float(np.mean([r["final_val_loss"] for r in held]))
    bootstrap = []
    generator = np.random.default_rng(seed)
    for _ in range(options["bootstrap_samples"]):
        indices = generator.integers(0, losses.shape[1], size=losses.shape[1])
        candidate = fit_power_law(sizes, losses[:, indices].mean(axis=1), options)
        bootstrap.append(float(predict(candidate, target)))
    # 留一规模验证只有三个训练点，仍是弱诊断，不能作为充分泛化证明。
    loo = []
    for index in range(len(sizes)):
        keep = [i for i in range(len(sizes)) if i != index]
        candidate = fit_power_law(np.asarray(sizes)[keep], losses.mean(axis=1)[keep], options)
        loo.append(dict(parameters=sizes[index], actual=float(losses[index].mean()),
                        predicted=float(predict(candidate, sizes[index]))))
    return dict(fit=fit, fitted_models=list(model_names), parameters=sizes,
                mean_loss=losses.mean(axis=1).tolist(),
                std_loss=losses.std(axis=1, ddof=1 if losses.shape[1] > 1 else 0).tolist(),
                holdout=dict(model=holdout_name, parameters=target, predicted_loss=predicted,
                             actual_loss=actual, absolute_error=abs(predicted - actual),
                             relative_error=abs(predicted - actual) / actual,
                             seed_bootstrap_prediction_interval=np.percentile(bootstrap, [2.5, 97.5]).tolist()),
                leave_one_size_out=loo,
                limitations=["仅四个规模拟合三个参数，只有一个残差自由度，不能证明普遍 scaling law。",
                             "固定 token 预算，不是固定 FLOPs、充分收敛或 Chinchilla compute-optimal 实验。",
                             "种子 bootstrap 仅反映本组运行波动，不涵盖幂律模型误设；默认仅三个种子。",
                             "共享词表和验证集；不可与先前 3 epoch 的 0.1B loss 混作等预算结果。"])
