from zipfile import ZipFile
import os
import splitfolders

def unzip_file(zip_path, extract_to):
    """Extracts a ZIP file."""
    with ZipFile(zip_path,
                 'r') as zip_ref:
        zip_ref.extractall(extract_to)

unzip_file("export.zip",
           "export")

splitfolders.ratio(
    input="export",
    output="dataset",
    seed=42,
    ratio=(0.7, 0.15, 0.15)
)