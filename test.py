from gdown import download

file_id = "1ylLRWqxjH-bpeFFI3YAT7kC0nBL3K2gx"  # Replace with the actual file ID
output_filename = "docu.docx"  # Replace with desired filename

download(f"https://drive.google.com/uc?export=download&id={file_id}")
