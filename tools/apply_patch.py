# tools/apply_patch.py
"""
Apply required patch to biolearns lmQCM to fix empty neighborWeights bug.

The biolearns lmQCM implementation raises ValueError: max() arg is an empty
sequence when a gene has no neighbors above the gamma threshold. This patch
adds a guard before the max() call.

Safe to run multiple times -- checks if patch is already applied.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_lmqcm_file() -> str:
    """
    Find the biolearns lmQCM source file in the current environment.

    Returns
    -------
    str
        Absolute path to _lmQCM.py.

    Raises
    ------
    FileNotFoundError
        If biolearns is not installed or file cannot be found.
    """
    try:
        import biolearns
        biolearns_dir = os.path.dirname(biolearns.__file__)
        lmqcm_path = os.path.join(
            biolearns_dir, "coexpression", "_lmQCM.py"
        )
        if not os.path.exists(lmqcm_path):
            raise FileNotFoundError(
                f"_lmQCM.py not found at {lmqcm_path}"
            )
        return lmqcm_path
    except ImportError:
        raise FileNotFoundError(
            "biolearns is not installed. Run: pip install biolearns==0.0.62"
        )


BUGGY_LINE = "                        maxNeighborWeight = max(neighborWeights)"

PATCHED_LINES = (
    "                        if len(neighborWeights) == 0:\n"
    "                            break\n"
    "                        maxNeighborWeight = max(neighborWeights)"
)


def is_already_patched(content: str) -> bool:
    """
    Check if the patch has already been applied.

    Parameters
    ----------
    content : str
        Full file content.

    Returns
    -------
    bool
        True if patch is already present.
    """
    return "if len(neighborWeights) == 0:" in content


def apply_patch(lmqcm_path: str) -> None:
    """
    Apply the empty neighborWeights guard patch to _lmQCM.py.

    Parameters
    ----------
    lmqcm_path : str
        Absolute path to _lmQCM.py.

    Raises
    ------
    ValueError
        If the buggy line is not found in the file -- version mismatch.
    """
    with open(lmqcm_path, "r") as f:
        content = f.read()

    if is_already_patched(content):
        logger.info("Patch already applied -- nothing to do.")
        return

    if BUGGY_LINE not in content:
        raise ValueError(
            f"Could not find target line in {lmqcm_path}.\n"
            f"Expected: {BUGGY_LINE!r}\n"
            "biolearns version may differ from 0.0.62. "
            "Apply the patch manually -- see README.md."
        )

    # Create backup before patching
    backup_path = lmqcm_path + ".bak"
    if not os.path.exists(backup_path):
        with open(backup_path, "w") as f:
            f.write(content)
        logger.info(f"Backup created at {backup_path}")
    else:
        logger.info(f"Backup already exists at {backup_path}")

    # Apply patch
    patched_content = content.replace(BUGGY_LINE, PATCHED_LINES)

    with open(lmqcm_path, "w") as f:
        f.write(patched_content)

    logger.info(f"Patch applied successfully to {lmqcm_path}")


def verify_patch(lmqcm_path: str) -> bool:
    """
    Verify the patch was applied correctly.

    Parameters
    ----------
    lmqcm_path : str
        Absolute path to _lmQCM.py.

    Returns
    -------
    bool
        True if patch is present and correct -- guard exists before max().
    """
    with open(lmqcm_path, "r") as f:
        content = f.read()

    if not is_already_patched(content):
        logger.error("Patch verification FAILED -- guard not found in file.")
        return False

    # Verify guard appears BEFORE max() call -- not just anywhere in file
    guard_pos = content.find("if len(neighborWeights) == 0:")
    max_pos = content.find("maxNeighborWeight = max(neighborWeights)")

    if guard_pos == -1 or max_pos == -1:
        logger.error("Patch verification FAILED -- expected lines not found.")
        return False

    if guard_pos > max_pos:
        logger.error(
            "Patch verification FAILED -- guard appears AFTER max() call."
        )
        return False

    logger.info("Patch verification PASSED -- guard precedes max() call.")
    return True