"""响应区间掩码、DPO 目标及 on-policy 分组策略梯度。"""

import torch
from torch.nn import functional as F


def response_batch(examples, tokenizer, context, device, responses=None):
    sequences, starts = [], []
    for index, example in enumerate(examples):
        prefix = tokenizer.encode(example["prompt"])
        response = responses[index] if responses is not None else tokenizer.encode(example["chosen"], add_eos=True)
        if not prefix or not response or len(prefix) + len(response) - 1 > context:
            raise ValueError("指令+回答超出上下文，或 prompt/response 为空")
        sequences.append(prefix + response)
        starts.append(len(prefix) - 1)
    length = max(len(ids) - 1 for ids in sequences)
    x = torch.full((len(sequences), length), tokenizer.pad_id, dtype=torch.long, device=device)
    y = torch.full_like(x, -100)
    for row, (ids, begin) in enumerate(zip(sequences, starts)):
        x[row, :len(ids) - 1] = torch.tensor(ids[:-1], device=device)
        y[row, begin:len(ids) - 1] = torch.tensor(ids[begin + 1:], device=device)
    return x, y


def response_log_probs(logits, targets, banned=()):
    values = logits.float()
    if banned:
        # 用有限极小值避免 KL 中的 0 * (-inf - -inf)。
        values = values.clone()
        values[..., list(banned)] = -1e9
    log_probs = values.log_softmax(-1)
    mask = targets != -100
    selected = log_probs.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (selected * mask).sum(-1), log_probs, mask


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta):
    margin = beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))
    return -F.logsigmoid(margin).mean(), margin.detach()


def relative_advantages(rewards):
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-4)


def reinforce_loss(policy_logits, reference_logits, targets, rewards, kl_beta, banned):
    scores, policy_logp, mask = response_log_probs(policy_logits, targets, banned)
    with torch.no_grad():
        _, reference_logp, _ = response_log_probs(reference_logits, targets, banned)
        advantages = relative_advantages(rewards)
    # 对采样前缀上的完整下一 token 分布计算 KL，不用 token 差值冒充精确 KL。
    token_kl = (policy_logp.exp() * (policy_logp - reference_logp)).sum(-1)
    kl = (token_kl * mask).sum() / mask.sum()
    loss = -(advantages * scores).mean() + kl_beta * kl
    return loss, dict(kl=float(kl.detach()), reward=float(rewards.mean()),
                      zero_advantage_group=bool((advantages == 0).all()))
