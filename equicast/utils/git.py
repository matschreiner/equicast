"""Git utilities for tracking code versions in experiments."""

import subprocess
from pathlib import Path
from typing import Optional


def get_git_hash(short: bool = False, repo_path: Optional[str] = None) -> Optional[str]:
    """
    Get the current git commit hash.

    Args:
        short: If True, return short hash (7 chars). If False, return full hash.
        repo_path: Path to git repository. If None, uses current directory.

    Returns:
        The git commit hash, or None if not in a git repository or git command fails.

    Example:
        >>> hash = get_git_hash()
        >>> if hash:
        ...     print(f"Running experiment at commit: {hash}")
    """
    try:
        cmd = ["git", "rev-parse"]
        if short:
            cmd.append("--short")
        cmd.append("HEAD")

        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def is_git_dirty(repo_path: Optional[str] = None) -> bool:
    """
    Check if the git repository has uncommitted changes.

    Args:
        repo_path: Path to git repository. If None, uses current directory.

    Returns:
        True if there are uncommitted changes, False if working tree is clean,
        or None if not in a git repository.

    Example:
        >>> if is_git_dirty():
        ...     print("Warning: Uncommitted changes detected!")
    """
    try:
        result = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=repo_path,
            capture_output=True,
            check=False,
        )
        # Exit code 0 means clean, 1 means dirty
        return result.returncode != 0
    except FileNotFoundError:
        return None


def get_git_info(repo_path: Optional[str] = None) -> dict:
    """
    Get comprehensive git information for experiment tracking.

    Args:
        repo_path: Path to git repository. If None, uses current directory.

    Returns:
        Dictionary with git information:
        - hash: Full commit hash
        - short_hash: Short (7 char) commit hash
        - dirty: Whether there are uncommitted changes
        - branch: Current branch name

    Example:
        >>> info = get_git_info()
        >>> logger.log_hyperparams({"git_hash": info["hash"], "git_dirty": info["dirty"]})
    """
    info = {
        "hash": get_git_hash(short=False, repo_path=repo_path),
        "short_hash": get_git_hash(short=True, repo_path=repo_path),
        "dirty": is_git_dirty(repo_path=repo_path),
        "branch": None,
    }

    # Get current branch name
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        info["branch"] = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return info
