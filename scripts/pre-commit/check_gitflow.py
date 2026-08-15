#!/usr/bin/env python3
"""Validate branch names and commit messages against Gitflow conventions.

The script is designed to be invoked from pre-commit hooks to ensure branches
and commit messages align with the agreed Gitflow workflow before code is
shared.

"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import typing as tp
from pathlib import Path

ALLOWED_BRANCHES: tp.Final[set[str]] = {"main", "master", "develop"}
ALLOWED_PREFIXES: tp.Final[tuple[str, ...]] = (
    "feature",
    "bugfix",
    "hotfix",
    "release",
    "support",
    "task",
    "chore",
)
BRANCH_SEGMENT_PATTERN: tp.Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SUMMARY_MAX_LEN: tp.Final[int] = 72
BODY_MAX_LEN: tp.Final[int] = 100


def _parse_args(argv: tp.Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Gitflow validation.

    Args:
        argv: Optional sequence of arguments for testing.

    Returns:
        Namespace containing parsed arguments.

    """
    parser = argparse.ArgumentParser(
        description="Validate branch names and commit messages for Gitflow compatibility",
    )
    parser.add_argument(
        "--commit-msg-file",
        type=Path,
        help="Path to the commit message file provided by Git.",
    )
    parser.add_argument(
        "--skip-branch-check",
        action="store_true",
        help="Skip branch name validation.",
    )
    parser.add_argument(
        "--skip-commit-check",
        action="store_true",
        help="Skip commit message validation.",
    )
    return parser.parse_args(argv)


def _run_git_command(args: tp.Sequence[str]) -> str | None:
    """Return stdout from a git command, or None if the command fails.

    Args:
        args: Arguments passed after ``git``.

    Returns:
        Command output stripped of whitespace, or ``None`` if the command fails.

    """
    try:
        return subprocess.check_output(
            ["git", *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _get_current_branch() -> str | None:
    """Return the current branch name.

    Returns:
        Branch name, or ``None`` when HEAD is detached or unknown.

    """
    branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    if not branch or branch == "HEAD":
        return None
    return branch


def _validate_branch_name(branch: str) -> list[str]:
    """Return validation errors for the provided branch name.

    Args:
        branch: Branch name to inspect.

    Returns:
        List of human-readable validation errors. Empty when valid.

    """
    if branch in ALLOWED_BRANCHES:
        return []

    for prefix in ALLOWED_PREFIXES:
        if branch.startswith(f"{prefix}/"):
            suffix = branch[len(prefix) + 1 :]
            if not suffix:
                return [f"branch '{branch}' must include a slug after '{prefix}/'"]
            segments = suffix.split("/")
            invalid_segments = [
                segment
                for segment in segments
                if not BRANCH_SEGMENT_PATTERN.match(segment) or segment.lower() != segment
            ]
            if invalid_segments:
                segments_list = ", ".join(invalid_segments)
                return [
                    (
                        f"branch '{branch}' uses invalid segment(s): {segments_list}. "
                        "Use lowercase letters, digits, dots, underscores, or hyphens."
                    )
                ]
            return []

    allowed_prefix_list = ", ".join(ALLOWED_PREFIXES)
    return [
        (
            f"branch '{branch}' must be one of {sorted(ALLOWED_BRANCHES)} "
            f"or start with one of: {allowed_prefix_list}."
        )
    ]


def _validate_commit_message(path: Path) -> list[str]:
    """Return validation errors for the commit message stored at path.

    Args:
        path: Path to the temporary commit message file.

    Returns:
        List of human-readable validation errors. Empty when valid.

    """
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [f"commit message file '{path}' does not exist"]

    lines = content.replace("\r\n", "\n").splitlines()
    if not lines:
        return ["commit message is empty"]

    summary = lines[0].strip()
    errors: list[str] = []

    if not summary:
        errors.append("commit summary (first line) must not be empty")
    if len(summary) > SUMMARY_MAX_LEN:
        errors.append(f"commit summary exceeds {SUMMARY_MAX_LEN} characters (len={len(summary)})")
    if summary and not summary[0].isalpha():
        errors.append("commit summary should start with a letter")

    if len(lines) > 1:
        separator = lines[1]
        if separator.strip():
            errors.append("commit summary must be followed by a blank line before body")
        for idx, body_line in enumerate(lines[2:], start=3):
            if len(body_line) > BODY_MAX_LEN:
                errors.append(f"line {idx} in commit body exceeds {BODY_MAX_LEN} characters")

    return errors


def main(argv: tp.Sequence[str] | None = None) -> int:
    """Program entrypoint responsible for coordinating validations.

    Args:
        argv: Optional sequence of CLI arguments for testing.

    Returns:
        Zero when Gitflow validation succeeds, otherwise 1.

    """
    args = _parse_args(argv)
    failures: list[str] = []

    if not args.skip_branch_check:
        branch = _get_current_branch()
        if branch is None:
            failures.append("unable to determine current branch (detached HEAD?)")
        else:
            failures.extend(_validate_branch_name(branch))

    if not args.skip_commit_check:
        if args.commit_msg_file is None:
            failures.append("commit message validation requested but no file supplied")
        else:
            failures.extend(_validate_commit_message(args.commit_msg_file))

    if failures:
        print("Gitflow convention violations detected:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
