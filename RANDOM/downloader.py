import gdown
import tarfile
import os

def acquire_search_space():

    current_directory = os.path.dirname(os.path.realpath(__file__))

    tss_file_path = os.path.join(current_directory, 'NATS-tss-v1_0-3ffb9-simple')

    if os.path.exists(tss_file_path) and os.path.isdir(tss_file_path):
        print("The search space directory already exists, skipping download.")
    else:
        print("The search space directory is not downloaded yet. Downloading now...")
        file_id = '17_saCsj_krKjlCBLOJEpNtzPXArMCqxU'
        url = f'https://drive.google.com/uc?id={file_id}'

        gdown.download(url, 'file.tar.gz', quiet=False)

        def untar_file(file_path, path_to_extract):
            with tarfile.open(file_path) as file:
                file.extractall(path=path_to_extract)

        untar_file('file.tar.gz', current_directory)
        print("Removing tar.gz file...")
        os.remove('file.tar.gz')
        print("File removed.")

        print("File downloaded and untarred successfully.")