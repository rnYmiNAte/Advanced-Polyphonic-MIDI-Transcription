import os
import urllib.request
import tarfile

def ensure_crepe_model():
    model_path = "models/crepe/full.pth"
    url = "https://github.com/marl/crepe/raw/master/crepe/model/full.pth"

    if not os.path.exists(model_path):
        os.makedirs("models/crepe", exist_ok=True)
        print("Downloading CREPE full model...")
        urllib.request.urlretrieve(url, model_path)
        print("CREPE model downloaded.")

def ensure_spleeter_model():
    model_dir = "models/spleeter/4stems"
    tar_url = "https://github.com/deezer/spleeter/releases/download/v2.1.0/4stems.tar.gz"
    tar_path = "models/spleeter/4stems.tar.gz"

    if not os.path.exists(model_dir):
        os.makedirs("models/spleeter", exist_ok=True)
        print("Downloading Spleeter 4-stem model...")
        urllib.request.urlretrieve(tar_url, tar_path)

        print("Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall("models/spleeter")
        
        os.remove(tar_path)
        print("Spleeter model ready.")
