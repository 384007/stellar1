#!/usr/bin/env python3
"""
Bulk-delete failed GitHub Actions runs for .github/workflows/deploy-modal.yml.

This removes historical red-X check results tied to those runs (GitHub drops the
status from commits when the run record is deleted).

Usage:
  export GITHUB_TOKEN=ghp_xxxx   # classic PAT: repo scope, or fine-grained: Actions read+write
  python3 scripts/delete_failed_modal_workflow_runs.py

Optional:
  GITHUB_REPOSITORY=owner/repo   # default: parsed from `git remote get-url origin`
  WORKFLOW_FILE=deploy-modal.yml
  DRY_RUN=1                      # only print run ids, do not delete
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request


def _remote_repo() -> tuple[str, str] | None:
    try:
        out = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    # https://github.com/owner/repo.git or git@github.com:owner/repo.git
    m = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", out)
    if not m:
        return None
    return m.group(1), m.group(2).rstrip("/")


def _request(method: str, url: str, token: str, data: bytes | None = None) -> tuple[int, dict | list | None]:
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode()
            if not body:
                return resp.status, None
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as e:
        err = e.read().decode()[:500]
        print(f"HTTP {e.code} {url}\n{err}", file=sys.stderr)
        raise SystemExit(1) from e


def _delete(url: str, token: str) -> None:
    req = urllib.request.Request(url, method="DELETE")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status not in (204, 200):
                print(f"Unexpected DELETE status {resp.status}", file=sys.stderr)
                raise SystemExit(1)
    except urllib.error.HTTPError as e:
        err = e.read().decode()[:500]
        print(f"HTTP {e.code} {url}\n{err}", file=sys.stderr)
        raise SystemExit(1) from e


def main() -> None:
    token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
    if not token:
        print(
            "Missing GITHUB_TOKEN (or GH_TOKEN). Create a PAT with permission to delete workflow runs.\n"
            "Classic: repo scope. Fine-grained: Actions read + write on this repository.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    env_repo = (os.environ.get("GITHUB_REPOSITORY") or "").strip()
    if env_repo and "/" in env_repo:
        owner, repo = env_repo.split("/", 1)
    else:
        parsed = _remote_repo()
        if not parsed:
            print("Could not parse owner/repo from git remote; set GITHUB_REPOSITORY=owner/repo", file=sys.stderr)
            raise SystemExit(1)
        owner, repo = parsed

    workflow_file = (os.environ.get("WORKFLOW_FILE") or "deploy-modal.yml").strip()
    dry = os.environ.get("DRY_RUN", "").strip() in ("1", "true", "yes")

    base = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_file}/runs"
    to_delete: list[int] = []
    page = 1
    while True:
        url = f"{base}?per_page=100&page={page}"
        status, payload = _request("GET", url, token)
        if status != 200 or not isinstance(payload, dict):
            print("Unexpected list response", file=sys.stderr)
            raise SystemExit(1)
        runs = payload.get("workflow_runs") or []
        if not runs:
            break
        for r in runs:
            if r.get("conclusion") == "failure":
                to_delete.append(int(r["id"]))
        if len(runs) < 100:
            break
        page += 1

    if not to_delete:
        print(f"No failed runs found for workflow {workflow_file} on {owner}/{repo}.")
        return

    print(f"Found {len(to_delete)} failed run(s) for {owner}/{repo} ({workflow_file}).")
    if dry:
        print("DRY_RUN: would delete:", to_delete[:20], ("..." if len(to_delete) > 20 else ""))
        return

    deleted = 0
    for run_id in to_delete:
        del_url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}"
        _delete(del_url, token)
        deleted += 1
        print(f"Deleted run {run_id} ({deleted}/{len(to_delete)})")

    print("Done. Refresh GitHub — commit status icons may take a short time to update.")


if __name__ == "__main__":
    main()
