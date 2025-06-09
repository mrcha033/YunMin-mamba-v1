import subprocess
import time

def quick_test(model_name: str) -> bool:
    """Run a minimal command to simulate a quick test."""
    result = subprocess.run(["echo", model_name], timeout=30)
    time.sleep(0.1)
    return result.returncode == 0
