"""本地离线演示页：只读取实验结果，按需加载一个模型进行生成。"""

import json
import logging
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit

from src.data import file_hash


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


class DemoService:
    def __init__(self, cfg):
        self.cfg = cfg
        self.static = Path(__file__).resolve().parent.parent / "web"
        self.lock = threading.Lock()
        self.model = self.tokenizer = self.active_name = None
        self.active_stamp = None
        self.metric_cache = None
        self.metric_stamp = None

    def model_paths(self):
        root = Path(self.cfg["alignment"]["output_dir"])
        return dict(base=Path(self.cfg["alignment"]["checkpoint"]),
                    **{name: root / name / "final.pt" for name in ("sft", "dpo", "rlvr")})

    def assets(self):
        paths = {"training": Path(self.cfg["web"]["training_dir"]) / "loss_curve.png"}
        for dtype in ("float16", "bfloat16"):
            for batch in (1, 4):
                for phase, suffix in (("forward", ""), ("forward_backward", "_forward_backward")):
                    paths[f"attention_{dtype}_{batch}_{phase}"] = Path(self.cfg["web"]["attention_dir"]) / f"attention_{dtype}_b{batch}{suffix}.png"
        for name in ("data_quality", "scaling_law", "validation_curves"):
            paths[name] = Path(self.cfg["web"]["science_dir"]) / f"{name}.png"
        paths["alignment"] = Path(self.cfg["alignment"]["output_dir"]) / "comparison.png"
        return paths

    def training_summary(self):
        root = Path(self.cfg["web"]["training_dir"])
        metrics = root / "metrics.jsonl"
        info = read_json(root / "model_info.json") or {}
        if not metrics.is_file():
            return dict(available=False, parameters=info.get("parameters"))
        stamp = (metrics.stat().st_mtime_ns, metrics.stat().st_size)
        if stamp != self.metric_stamp:
            best, step = None, 0
            for line in metrics.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 写入中的最后半行不会被当作完成结果。
                step = max(step, row.get("step", 0))
                if "val_loss" in row and math.isfinite(row["val_loss"]):
                    best = row["val_loss"] if best is None else min(best, row["val_loss"])
            self.metric_cache = dict(available=True, parameters=info.get("parameters"), best_val_loss=best,
                                     perplexity=math.exp(best) if best is not None and best < 80 else None,
                                     steps=step, plan=info.get("training_plan"))
            self.metric_stamp = stamp
        return self.metric_cache

    def summary(self):
        attention = read_json(Path(self.cfg["web"]["attention_dir"]) / "results.json")
        science_root = Path(self.cfg["web"]["science_dir"])
        science = read_json(science_root / "report.json")
        plan = read_json(science_root / "plan.json")
        alignment = read_json(Path(self.cfg["alignment"]["output_dir"]) / "report.json")
        identical = None
        if plan:
            files = plan.get("tokens", {}).get("files", {})
            if "raw" in files and "clean" in files:
                identical = files["raw"]["sha256"] == files["clean"]["sha256"]
        return dict(training=self.training_summary(),
                    attention=dict(status=attention.get("status"), checks=len(attention.get("checks", [])),
                                   failed_checks=sum(not r["passed"] for r in attention.get("checks", [])),
                                   benchmarks=attention.get("benchmarks", [])) if attention else None,
                    science=dict(scaling=science["scaling"], statistics=science["data"]["statistics"],
                                 data_quality=science["data_quality"], identical_tokens=identical) if science else None,
                    alignment=dict(evaluation=[{k:v for k,v in r.items() if k != "samples"} for r in alignment["evaluation"]],
                                   samples={r["model"]: r["samples"][:5] for r in alignment["evaluation"]},
                                   limitations=alignment["limitations"]) if alignment else None,
                    models={name:path.is_file() for name,path in self.model_paths().items()},
                    assets=[name for name,path in self.assets().items() if path.is_file()],
                    scope="第 1–4 部分；单卡实验。第 5 部分 Agent 未做，多卡与真实网页清洗实证未覆盖。")

    def generate(self, payload):
        import copy
        import gc
        import re
        import time
        import torch
        from src.alignment_data import reward, task_prompt
        from src.alignment_tasks import model_from_checkpoint, sample_responses
        from src.tokenizer import BPETokenizer
        from src.training import generate, setup

        name = payload.get("model", "base")
        mode = payload.get("mode", "continuation")
        max_tokens = payload.get("max_new_tokens", 144)
        seed = payload.get("seed", 123)
        if name not in self.model_paths() or mode not in ("continuation", "instruction"):
            raise ValueError("未知模型或生成模式")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 256 or type(seed) is not int or not 0 <= seed < 2 ** 31:
            raise ValueError("生成长度需为 1–256，种子需为非负整数")
        words = payload.get("words", [])
        if mode == "instruction":
            if not isinstance(words, list) or len(words) != 2 or any(not isinstance(w, str) or not re.fullmatch(r"[A-Za-z]{1,24}", w) for w in words):
                raise ValueError("请输入两个不同的英文单词，例如 cat、tree")
            words = [w.lower() for w in words]
            if words[0] == words[1]:
                raise ValueError("两个关键词需要不同")
            prompt = task_prompt(words)
        else:
            prompt = payload.get("prompt", "")
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 1200 or not prompt.isascii():
                raise ValueError("请输入 1–1200 字符的英文故事开头")
        path = self.model_paths()[name]
        if not path.is_file():
            raise ValueError("未找到预训练权重，请检查 alignment.checkpoint" if name == "base" else "未找到微调权重，请检查结果目录或先运行 stat=align")
        with self.lock:
            options = copy.deepcopy(self.cfg)
            device, precision = setup(options)
            tokenizer_path = Path(self.cfg["alignment"]["tokenizer_dir"]) / "tokenizer.json"
            stamp = (str(path.resolve()), path.stat().st_mtime_ns, tokenizer_path.stat().st_mtime_ns)
            if self.active_stamp != stamp:
                self.model = None
                self.active_stamp = None
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                instance = BPETokenizer.load(tokenizer_path)
                self.tokenizer = dict(instance=instance, sha256=file_hash(tokenizer_path))
                self.model = model_from_checkpoint(path, self.tokenizer, device).eval()
                self.active_stamp, self.active_name = stamp, name
            began = time.perf_counter()
            if mode == "instruction":
                length = min(max_tokens, self.model.context - len(self.tokenizer["instance"].encode(prompt)) + 1)
                rng = torch.Generator(device=device).manual_seed(seed)
                ids = sample_responses(self.model, self.tokenizer["instance"], prompt, 1, length, rng, device, precision)[0]
                text = self.tokenizer["instance"].decode(ids, errors="replace")
                result = dict(text=text, new_tokens=len(ids), verification=reward(text, words, self.cfg["alignment"]["min_words"]))
            else:
                result = generate(self.model, self.tokenizer["instance"], dict(prompt=prompt, max_new_tokens=max_tokens,
                                  temperature=0.8, top_k=50, seed=seed), device, precision)
            result.update(model=name, mode=mode, seconds=time.perf_counter() - began)
            return result


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value, content_type="application/json; charset=utf-8"):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode() if isinstance(value, (dict, list)) else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlsplit(self.path)
            try:
                if url.path == "/api/summary":
                    return self.reply(200, service.summary())
                static = {"/": ("index.html", "text/html; charset=utf-8"),
                          "/style.css": ("style.css", "text/css; charset=utf-8"),
                          "/app.js": ("app.js", "text/javascript; charset=utf-8")}
                if url.path in static:
                    name, mime = static[url.path]
                    return self.reply(200, (service.static / name).read_bytes(), mime)
                if url.path == "/asset":
                    key = parse_qs(url.query).get("key", [""])[0]
                    path = service.assets().get(key)
                    if path is not None and path.is_file():
                        return self.reply(200, path.read_bytes(), "image/png")
                self.reply(404, dict(error="未找到文件"))
            except (OSError, ValueError, KeyError) as exc:
                logging.exception("读取演示结果失败")
                self.reply(500, dict(error=f"读取结果失败：{exc}"))

        def do_POST(self):
            if urlsplit(self.path).path != "/api/generate":
                return self.reply(404, dict(error="未知接口"))
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                return self.reply(403, dict(error="不接受跨站生成请求"))
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384 or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    return self.reply(400, dict(error="请求需为不超过 16KiB 的 JSON"))
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("请求必须是 JSON 对象")
                self.reply(200, service.generate(data))
            except (ValueError, KeyError) as exc:
                self.reply(400, dict(error=str(exc)))
            except Exception:
                logging.exception("生成失败")
                self.reply(500, dict(error="生成失败，请查看服务器终端日志"))

        def log_message(self, format, *args):
            logging.info("web: " + format, *args)
    return Handler


def serve(cfg):
    host, port = cfg["web"]["host"], cfg["web"]["port"]
    if host not in ("127.0.0.1", "localhost") or type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("本地演示请使用 127.0.0.1/localhost 和 1024–65535 端口，通过 SSH 转发访问")
    server = ThreadingHTTPServer((host, port), handler_for(DemoService(cfg)))
    logging.info("本地演示页面 http://127.0.0.1:%d；远程服务器需 SSH 端口转发", port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("演示服务已停止")
    finally:
        server.server_close()
    return dict(stopped=True)
