import io
import zipfile
from pathlib import Path
from PIL import Image
from .common import read_json, sha

class FrameStore:
    def __init__(self, root, video_id, verify=False):
        root = Path(root)
        self.meta = read_json(root/f'{video_id}.json')
        path = root/f'{video_id}.zip'
        if verify and sha(path) != self.meta['zip_sha256']:
            raise ValueError(f'Corrupt frame store {video_id}')
        self.count = self.meta['frames']
        self.zip = zipfile.ZipFile(path)

    def image(self, t):
        if int(t) != t or not 0 <= t < self.count:
            raise ValueError(f'Invalid frame {t}')
        return Image.open(io.BytesIO(self.zip.read(f'f/{int(t):06d}.jpg'))).convert('RGB')

    def surgformer_input(self, t):
        return self.image(t).resize((320,320),Image.Resampling.BILINEAR)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.zip.close()
