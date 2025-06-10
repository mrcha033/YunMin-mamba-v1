import subprocess
import time


def quick_test(mode: str = "baseline") -> bool:
    """Minimal CLI wrapper used in tests."""
    cmd = ["echo", mode]
    if mode == "ia3":
        cmd.append("--ia3")
    result = subprocess.run(cmd)
    time.sleep(0.1)
    return result.returncode == 0

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    success = quick_test(mode)
    raise SystemExit(0 if success else 1)
