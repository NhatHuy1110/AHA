"""Encoder extraction contract on a tiny randomly initialised ViT (no pretrained weights)."""
import io
import zipfile
import numpy as np
import pytest
from PIL import Image
from aha.common import read_json,write_json,sha
from aha.features import encoder

def test_label_free_encoder_cache_shape_resume_and_provenance(tmp_path):
    transformers=pytest.importorskip('transformers')
    root=tmp_path; n=5
    model=transformers.Dinov2Model(transformers.Dinov2Config(hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
                                   image_size=28,patch_size=14))
    source=root/'assets/encoders/tiny'; model.save_pretrained(source)
    write_json(source/'source.json',dict(id='random-init-test',revision='0'*40))
    write_json(root/'data/manifests/splits.json',dict(train=['001'],val=[],test=[]))
    path=root/'data/frames/001.zip'; path.parent.mkdir(parents=True)
    rng=np.random.default_rng(0)
    with zipfile.ZipFile(path,'w') as z:
        for t in range(n):
            b=io.BytesIO(); Image.fromarray(rng.integers(0,255,(36,64,3),dtype=np.uint8)).save(b,'JPEG')
            z.writestr(f'f/{t:06d}.jpg',b.getvalue())
    write_json(root/'data/frames/001.json',dict(frames=n,zip_sha256=sha(path)))
    encoder(root,'tiny',device='cpu',batch=2,height=28,width=56)
    out=root/'artifacts/features/tiny'
    with np.load(out/'001.npz') as z:
        assert z['hidden'].shape==(n,32) and z['hidden'].dtype==np.float16 and np.isfinite(z['hidden']).all()
        assert np.array_equal(z['times'],np.arange(n))
    meta=read_json(out/'001.json'); stamp=(out/'001.npz').stat().st_mtime_ns
    assert meta['complete'] and meta['frames']==n and read_json(out/'spec.json')['revision']=='0'*40
    encoder(root,'tiny',device='cpu',batch=2,height=28,width=56)  # resume: nothing recomputed
    assert (out/'001.npz').stat().st_mtime_ns==stamp
    with pytest.raises(ValueError,match='different provenance'):
        encoder(root,'tiny',device='cpu',batch=3,height=28,width=56)
