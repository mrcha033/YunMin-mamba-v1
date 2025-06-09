import subprocess
import time

def quick_test(mode: str) -> bool:
    """Run a quick training session for the specified mode.

    Parameters
    ----------
    mode : str
        A configuration name passed to the training script.

    Returns
    -------
    bool
        ``True`` if the subprocess exits successfully, ``False`` otherwise.
    """
    cmd = ["python", "train.py", "--mode", mode]
    completed = subprocess.run(cmd, timeout=60)
    time.sleep(0.1)
    return completed.returncode == 0
