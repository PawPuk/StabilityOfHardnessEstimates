"""Utility functions for downloading and extracting datasets."""

import os
import urllib.request
import zipfile


def download_url(url: str, output_path: str):
    """Download file from URL with simple progress reporting."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, output_path)


def extract_archive(archive_path: str, extract_path: str):
    """Extract zip or tar archive."""
    print(f"Extracting {archive_path} to {extract_path}...")

    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
