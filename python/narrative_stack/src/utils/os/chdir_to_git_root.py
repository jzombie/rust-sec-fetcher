import logging
import subprocess
import os


# TODO: Use proper type
def chdir_to_git_root(relative_path: str = ""):
    """
    Change the working directory to the root of the git project.
    Optionally, append a relative subdirectory under the git root.

    :param relative_path: Optional relative path from the git root.
    """
    logging.info("Original cwd: %s", os.getcwd())

    # Get the top-level Git directory
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    project_root = result.stdout.strip()

    # If a relative path is provided, join it to the root
    target_path = (
        os.path.join(project_root, relative_path) if relative_path else project_root
    )

    os.chdir(target_path)
    logging.info("Updated cwd: %s", os.getcwd())
