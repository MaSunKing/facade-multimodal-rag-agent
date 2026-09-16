"""Scan public files/history without printing possible secret values."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "api_secret": re.compile(r"\bsk-[A-Za-z0-9_-]{24,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "literal_api_key": re.compile(r"(?:API_KEY|api_key|access_token)\s*[=:]\s*['\"]([A-Za-z0-9_-]{32,})['\"]"),
}
PRIVATE_FILE = re.compile(r"(?:^|/)(?:\.env|[^/]+\.(?:db|sqlite3?|key|pem|safetensors))$")
TEXT_SUFFIXES = {".py", ".ts", ".tsx", ".css", ".md", ".json", ".yml", ".yaml", ".txt", ".toml", ".ps1", ".svg"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def scan(path, content, findings, revision="working_tree"):
    text = content.decode("utf-8", errors="replace")
    for rule, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({"path": path, "revision": revision, "rule": rule,
                             "line": text.count("\n", 0, match.start()) + 1})
    if revision == "working_tree":
        for rule, pattern in {"personal_path": r"[Cc]:[/\\]Users[/\\](?!Public)",
                              "production_address": r"https://[\w.-]+\.(?:ts\.net|tcloudbaseapp\.com)",
                              "unrelated_documentation": r"简历|求职|秋招|招聘方"}.items():
            if Path(path).suffix == ".md" or rule != "unrelated_documentation":
                if re.search(pattern, text):
                    findings.append({"path": path, "revision": revision, "rule": rule})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--history", action="store_true")
    args = parser.parse_args(); findings = []
    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
    paths = sorted(set(p for p in paths if p))
    count = 0
    for path in paths:
        file = ROOT / path
        if PRIVATE_FILE.search(path):
            findings.append({"path": path, "rule": "private_file", "revision": "working_tree"})
        if file.is_file() and (file.suffix in TEXT_SUFFIXES or file.name in {".env.example", ".gitignore"}):
            scan(path, file.read_bytes(), findings); count += 1
    historical = 0
    if args.history:
        entries = git("rev-list", "--objects", "--all").decode().splitlines()
        with subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
            for entry in entries:
                parts = entry.split(" ", 1)
                if len(parts) != 2:
                    continue
                oid, path = parts
                if PRIVATE_FILE.search(path):
                    findings.append({"path": path, "rule": "historical_private_file", "revision": oid})
                if Path(path).suffix not in TEXT_SUFFIXES and Path(path).name not in {".env.example", ".gitignore"}:
                    continue
                proc.stdin.write((oid + "\n").encode()); proc.stdin.flush()
                header = proc.stdout.readline().split()
                size = int(header[2]); content = proc.stdout.read(size); proc.stdout.read(1)
                scan(path, content, findings, oid); historical += 1
            proc.stdin.close()
    report = {"text_files_scanned": count, "historical_objects_scanned": historical,
              "findings": findings, "passed": not findings,
              "scope": "pattern-based scan, not a guarantee of absence of every secret or personal detail"}
    folder = ROOT / "runtime/public_smoke"; folder.mkdir(parents=True, exist_ok=True)
    (folder / "security_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    raise SystemExit(0 if not findings else 1)


if __name__ == "__main__":
    main()
