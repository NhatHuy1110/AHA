from pathlib import Path
import numpy as np
import pytest
from aha.common import read_json,qa_files,PHASE_NAMES,TOOL_NAMES
from aha.data import Study

ROOT=Path(__file__).resolve().parents[1]

def test_staged_split_and_qa_counts():
    splits=read_json(ROOT/'data/manifests/splits.json')
    assert [len(splits[k]) for k in ('train','val','test')]==[30,11,16]
    assert not set(splits['train'])&(set(splits['val'])|set(splits['test']))
    assert not set(splits['val'])&set(splits['test'])
    for split,n in [('train',106),('val',44),('test',61)]:
        q=read_json(ROOT/f'data/questions/{split}.json')
        assert len(q)==n
        assert all(not set(x)&{'answer','onsets','supported_onsets'} for x in q)
        assert {x['video_id'] for x in q}<=set(splits[split])

def test_test_cannot_be_training_dataset():
    with pytest.raises(ValueError,match='Training may only'):
        Study(ROOT,read_json(ROOT/'configs/full.json'),'test',training=True)

def test_offset_alignment_and_train_unknown_count():
    inv=read_json(ROOT/'data/manifests/inventory.json')
    train=read_json(ROOT/'data/manifests/splits.json')['train']
    unknown=0
    for v in inv:
        if v['video_id'] not in train:
            continue
        with np.load(ROOT/f'data/labels/train/{v["video_id"]}.npz') as z:
            off=v['label_offset_seconds']
            assert (z['phase'][:off]==-1).all()
            assert (z['phase'][off:]>=0).sum()==v['num_frames']
            unknown+=(~z['tool_mask'][off:]).sum()
    assert unknown==1

def test_derived_v2_provenance_support_and_language():
    splits=read_json(ROOT/'data/manifests/splits.json')
    names=set(PHASE_NAMES.values()); tools=set(TOOL_NAMES.values())
    for split,n in [('train',208),('val',62),('test',100)]:
        qp,gp=qa_files(ROOT,split,'derived')
        qs=read_json(qp); gs=read_json(gp)
        assert len(qs)==len(gs)==n and [q['qid'] for q in qs]==[g['qid'] for g in gs]
        assert ('private' in gp.parts)==(split=='test')
        assert all(q['video_id'] in splits[split] and q['origin']=='derived_v2_labels' for q in qs)
        assert all(not set(q)&{'answer','onsets','supported_onsets'} for q in qs)
        assert all(g['supported_onsets']==g['onsets'] and g['onsets'] for g in gs)
        assert all(any(name in q['question'] for name in names) for q in qs)
        assert all(o[3:] in tools for q in qs for o in q['options'])
        for q,g in zip(qs,gs):
            assert q['option_tools'][ord(g['answer'])-65]==g['answer_tool']

def test_every_question_and_option_has_a_text_embedding():
    texts=set(read_json(ROOT/'artifacts/text/embeddings.json')['texts'])
    for split in ('train','val','test'):
        for kind in ('original','derived'):
            for q in read_json(qa_files(ROOT,split,kind)[0]):
                assert q['question'] in texts and all(o[3:] in texts for o in q['options'])

def test_inference_has_no_label_reader_access():
    # Deliberately bypass initialization so this access-control test needs no GPU cache.
    study=object.__new__(Study)
    study.training=False; study.ids=['053']
    with pytest.raises(ValueError,match='Inference dataset'):
        study.labels('053')
