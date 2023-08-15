class confServer:
    dataset={
        "spectrogram": "/data/conan/spectrogram_dataset",
        "audio": "/data/conan/mp3_dataset",
        "mid": None,
    }
    model={
        "model_path": "/data/conan/model",
    }
    device="cuda"

class confLocal:
    dataset={
        "spectrogram": "/Users/conan/Desktop/SDSC/audio_style_transfer/datasets/spectrogram_dataset",
        "audio": "/Users/conan/Desktop/SDSC/audio_style_transfer/datasets/audio_dataset",
        "mid": "/Users/conan/Desktop/SDSC/audio_style_transfer/datasets/mid_small",
    }
    model={
        "model_path": "/Users/conan/Desktop/SDSC/audio_style_transfer/model/",
    }
    device="cpu"
    

config = confServer()