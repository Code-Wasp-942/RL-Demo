import argparse
import os
import re
import subprocess
import sys
from typing import Dict, Optional


UPDATE_RE = re.compile(
    r"update=(?P<update>\d+)/(?:\d+)\s+global_step=(?P<global_step>\d+)\s+"
    r"mean_rew=(?P<mean_rew>[-+]?\d*\.?\d+)\s+"
    r"kl=(?P<kl>[-+]?\d*\.?\d+)\s+"
    r"clipfrac=(?P<clipfrac>[-+]?\d*\.?\d+)"
)


class MagnusLogger:
    def __init__(self, enabled: bool, project: str, run_name: str, tags: str):
        self.enabled = enabled
        self.project = project
        self.run_name = run_name
        self.tags = [item.strip() for item in tags.split(",") if item.strip()]
        self.module = None
        self.run = None

        if not self.enabled:
            return

        self.module = self._import_magnus_module()
        if self.module is None:
            print("[magnus] SDK not found, fallback to console only")
            self.enabled = False
            return

        self.run = self._start_run()

    def _import_magnus_module(self):
        candidates = [
            ("magnus", "sdk"),
            ("magnus_sdk", None),
        ]
        for module_name, attr in candidates:
            try:
                module = __import__(module_name, fromlist=[attr] if attr else [])
                if attr:
                    return getattr(module, attr, None)
                return module
            except Exception:
                continue
        return None

    def _start_run(self):
        module = self.module
        if module is None:
            return None

        tags = self.tags
        kwargs = {"project": self.project, "run_name": self.run_name}
        if tags:
            kwargs["tags"] = tags

        try:
            if hasattr(module, "init"):
                return module.init(**kwargs)
            if hasattr(module, "start_run"):
                return module.start_run(**kwargs)
            if hasattr(module, "Run"):
                return module.Run(**kwargs)
        except Exception as exc:
            print(f"[magnus] start failed: {exc}")

        print("[magnus] Unsupported SDK API, fallback to console only")
        self.enabled = False
        return None

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if not self.enabled:
            return

        module = self.module
        run = self.run

        try:
            if run is not None:
                if hasattr(run, "log_metrics"):
                    run.log_metrics(metrics=metrics, step=step)
                    return
                if hasattr(run, "log_metric"):
                    for key, value in metrics.items():
                        run.log_metric(key, value, step=step)
                    return

            if module is not None:
                if hasattr(module, "log_metrics"):
                    module.log_metrics(metrics=metrics, step=step)
                    return
                if hasattr(module, "log_metric"):
                    for key, value in metrics.items():
                        module.log_metric(key, value, step=step)
                    return
        except Exception as exc:
            print(f"[magnus] log failed: {exc}")

    def close(self):
        if not self.enabled:
            return
        try:
            if self.run is not None:
                if hasattr(self.run, "finish"):
                    self.run.finish()
                    return
                if hasattr(self.run, "end"):
                    self.run.end()
                    return
            if self.module is not None and hasattr(self.module, "finish"):
                self.module.finish()
        except Exception as exc:
            print(f"[magnus] close failed: {exc}")


def parse_metrics(line: str) -> Optional[Dict[str, float]]:
    match = UPDATE_RE.search(line)
    if match is None:
        return None

    return {
        "update": float(match.group("update")),
        "global_step": float(match.group("global_step")),
        "mean_rew": float(match.group("mean_rew")),
        "kl": float(match.group("kl")),
        "clipfrac": float(match.group("clipfrac")),
    }


def build_command(ppo_args):
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_train.py")
    return [sys.executable, script_path, *ppo_args]


def run_ppo_and_log(ppo_args, logger: MagnusLogger) -> int:
    command = build_command(ppo_args)
    print("[magnus] launching:", " ".join(command))

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        metrics = parse_metrics(line)
        if metrics is not None:
            step = int(metrics["global_step"])
            logger.log_metrics(metrics, step=step)

    process.wait()
    return process.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Magnus wrapper for ppo_train.py")
    parser.add_argument("--project", type=str, default="rl-demo")
    parser.add_argument("--run-name", type=str, default="ppo-4link")
    parser.add_argument("--tags", type=str, default="ppo,rl,4link")
    parser.add_argument("--disable-magnus", action="store_true", default=False)
    parser.add_argument("ppo_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    ppo_args = args.ppo_args
    if ppo_args and ppo_args[0] == "--":
        ppo_args = ppo_args[1:]

    return args, ppo_args


def main():
    args, ppo_args = parse_args()
    logger = MagnusLogger(
        enabled=not args.disable_magnus,
        project=args.project,
        run_name=args.run_name,
        tags=args.tags,
    )

    try:
        code = run_ppo_and_log(ppo_args, logger)
    finally:
        logger.close()

    sys.exit(code)


if __name__ == "__main__":
    main()
