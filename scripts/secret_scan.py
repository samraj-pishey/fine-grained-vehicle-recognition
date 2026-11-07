"""Pre-push secret and restricted-asset scan of the working tree AND git history.

Exit code 1 on any finding. Prints file:line and pattern name only - never the matched secret value.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "huggingface_token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "github_token": re.compile(r"(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}"),
    "openai_like_key": re.compile(r"sk-[A-Za-z0-9_-]{32,}"),
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "google_api_key": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    "private_key_block": re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "kaggle_key_json": re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),
    "url_with_credentials": re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
    "generic_assignment": re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[=:]\s*['\"][^'\"\s]{12,}['\"]"),
}
RESTRICTED_SUFFIXES = {".tgz", ".mat", ".pt", ".pth", ".ckpt", ".npz", ".parquet", ".joblib"}
RESTRICTED_NAMES = {".env", "kaggle.json", "token", "credentials.json"}
ALLOW_FILES = {"scripts/secret_scan.py"}  # contains the patterns themselves


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    return [f for f in out.split("\0") if f]


def scan_text(name: str, text: str) -> list[str]:
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        for pname, rx in PATTERNS.items():
            if rx.search(line):
                hits.append(f"{name}:{i}: {pname}")
    return hits


def main() -> int:
    findings: list[str] = []
    for f in tracked_files():
        p = ROOT / f
        if Path(f).suffix.lower() in RESTRICTED_SUFFIXES or Path(f).name in RESTRICTED_NAMES:
            findings.append(f"{f}: restricted file type would be published")
        if f in ALLOW_FILES or not p.is_file() or p.stat().st_size > 5_000_000:
            continue
        try:
            findings += scan_text(f, p.read_text(errors="ignore"))
        except OSError:
            pass
        # dataset images must never be committed
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and re.fullmatch(r"\d{5}\.jpg", p.name):
            findings.append(f"{f}: looks like a Stanford Cars image filename")
    # history: every blob ever committed
    if (ROOT / ".git").exists():
        log = subprocess.run(["git", "log", "--all", "-p", "--no-color"], cwd=ROOT, capture_output=True,
                             text=True, errors="ignore").stdout
        cur = "history"
        for line in log.splitlines():
            if line.startswith("diff --git"):
                cur = line.split(" b/")[-1]
            if line.startswith("+") and cur not in ALLOW_FILES:
                for pname, rx in PATTERNS.items():
                    if rx.search(line):
                        findings.append(f"history:{cur}: {pname}")
    findings = sorted(set(findings))
    for f in findings:
        print("FINDING", f)
    print(f"secret scan: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
