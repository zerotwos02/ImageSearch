from PIL import Image
from pathlib import Path
import numpy as np
from extractor_manager import getExtractor


if __name__ == '__main__':
    fe, extractor_value  = getExtractor()

    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        img_path_str = str(img_path)
        feature = fe.extract(img_path=img_path_str)
        feature_path = Path("./static/feature/",extractor_value) / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)
