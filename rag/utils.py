import sys
from pathlib import Path


def get_default_data_dir(app_name: str) -> Path:
    """
    Get the user data directory for the current system platform.

    Linux: ~/.local/share/<app_name>
    macOS: ~/Library/Application Support/<app_name>
    Windows: C:/Users/<USER>/AppData/Roaming/<app_name>

    :param app_name: Application Name will be used to specify directory
    :type app_name: str
    :return: User Data Path
    :rtype: Path
    """
    home = Path.home()

    system_paths = {
        "win32": home / f"AppData/Roaming/{app_name}",
        "linux": home / f".local/share/{app_name}",
        "darwin": home / f"Library/Application Support/{app_name}",
    }

    data_path = system_paths[sys.platform]
    return data_path
