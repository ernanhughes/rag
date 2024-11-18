import hashlib
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


def get_default_data_dir(app_name: str) -> Path:
    """
    Get the user data directory for the current system platform.

    Linux: ~/.local/share/<app_name>
    macOS: ~/Library/Application Support/<app_name>
    Windows: C:/Users/<USER>/AppData/Roaming/<app_name>

    :param app_name: Application Name will be used to specify directory
    :type app_name: str
    :return: User Data Directory
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


def sanitize_filename(title):
    """Sanitizes a string to be used as a filename."""
    # Remove any characters that are not alphanumeric, spaces, hyphens, or underscores
    return re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")


def get_filename_from_url(url):
    # Parse the URL to get the path component
    parsed_url = urlparse(url)
    # Get the base name from the URL's path
    filename = os.path.basename(parsed_url.path)
    return filename


def compute_file_hash(file_path, algorithm="sha256"):
    """Compute the hash of a file using the specified algorithm."""
    hash_func = hashlib.new(algorithm)

    with open(file_path, "rb") as file:
        # Read the file in chunks of 8192 bytes
        while chunk := file.read(8192):
            hash_func.update(chunk)

    return hash_func.hexdigest()
