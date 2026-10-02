"""One-time staging from the historical workspace; runtime never uses that workspace."""
import os
import shutil
from pathlib import Path
import numpy as np
from .common import PHASES, TOOLS, read_json, write_json, sha, atomic_npz

def stage(src, dst, hardlink=False):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if src.stat().st_size != dst.stat().st_size or sha(src) != sha(dst):
            raise ValueError(f'Existing staged file differs: {dst}')
        return
    if hardlink:
        os.link(src, dst)  # Fail explicitly across volumes; never silently fill the disk.
    else:
        shutil.copy2(src, dst)

def labels(phase_path, tool_path, offset, count):
    pl = [s.split() for s in Path(phase_path).read_text().splitlines()[1:] if s.strip()]
    tl = [s.split() for s in Path(tool_path).read_text().splitlines()[1:] if s.strip()]
    assert len(pl) == len(tl) and [r[0] for r in pl] == [r[0] for r in tl]
    p = np.full(count, -1, np.int64)
    t = np.full((count, 8), -1, np.float32)
    p[offset:offset+len(pl)] = [PHASES.index(r[1]) for r in pl]
    t[offset:offset+len(tl)] = np.asarray([r[1:] for r in tl], np.float32)
    # Unknown labels can be anything except 0/1; preserve their mask.
    mask = (t == 0) | (t == 1)
    onset = np.zeros((count, 12), np.float32)
    transitions = np.flatnonzero((p[1:] != p[:-1]) & (p[1:] >= 0) & (p[:-1] >= 0)) + 1
    onset[transitions, p[transitions]] = 1
    boundary_mask = p >= 0
    boundary_mask[max(0, offset-5):offset+6] = False  # left-censored start
    return dict(phase=p, tool=np.where(mask, t, 0), tool_mask=mask,
                onset=onset, boundary_mask=boundary_mask)

def derive(video_id, lab):
    """Conservative single-choice phase-onset QA; distractors absent at EVERY onset.

    An answer must be present at every fully observed occurrence. Unknowns or
    mixed instrument identities cannot become false single-choice labels.
    """
    qs, gs = [], []
    for pi, phase in enumerate(PHASES):
        ts = np.flatnonzero(lab['onset'][:, pi])
        if not len(ts):
            continue
        known = lab['tool_mask'][ts].all(0)
        present = np.flatnonzero(known & (lab['tool'][ts] == 1).all(0))
        absent = np.flatnonzero(known & (lab['tool'][ts] == 0).all(0))
        if len(absent) < 3:
            continue
        for a in present:
            rng = np.random.default_rng(int(video_id)*1000 + pi*10 + int(a))
            opts = np.r_[a, rng.choice(absent, 3, replace=False)]
            rng.shuffle(opts)
            answer = int(np.flatnonzero(opts == a)[0])
            qid = f'derived_{video_id}_{pi}_{a}'
            qs.append(dict(qid=qid, video_id=video_id, target_phase=phase,
                question=f'At a fully observed onset of {phase}, which listed instrument is visible?',
                options=[f'{chr(65+i)}. {TOOLS[o].replace("_", " ")}' for i,o in enumerate(opts)],
                option_tools=[TOOLS[o] for o in opts], origin='derived_train_labels'))
            gs.append(dict(qid=qid, video_id=video_id, answer=chr(65+answer),
                answer_tool=TOOLS[a], target_phase=phase, onsets=ts.tolist(),
                supported_onsets=ts.tolist(), primary_eligible=True, flags=[],
                gold_level='annotation_derived', origin='derived_train_labels'))
    return qs, gs

def prepare(workspace, root, hardlink=True):
    workspace, root = Path(workspace), Path(root)
    src = workspace / 'IPL/manifests/study_v2'
    old = read_json(src / 'splits.json')
    splits = dict(train=old['surgformer_train'], val=old['dev'], test=old['test'])
    sets = [set(x) for x in splits.values()]
    assert not any(a & b for i,a in enumerate(sets) for b in sets[i+1:])
    assert tuple(map(len, sets)) == (30,11,16)
    inv = read_json(src / 'inventory.json')
    questions = read_json(src / 'questions.json')
    gold = read_json(src / 'private/gold.json')
    for g in gold:
        g['supported_onsets'] = [o['second'] for o in g['per_onset'] if g['answer'] in o['options_labelled_present']]
    write_json(root/'data/manifests/splits.json', splits)
    write_json(root/'data/manifests/inventory.json', inv)
    write_json(root/'data/manifests/classes.json', dict(phases=PHASES,tools=TOOLS))
    frames = workspace / 'data/ipl_frames_v1'
    fm = read_json(frames/'frames_manifest.json')['videos']
    summary, assets = {}, []
    derived_q, derived_g = [], []
    for split, ids in splits.items():
        sq = [q for q in questions if q['video_id'] in ids]
        for q in sq:
            q.pop('video_path', None)
            q['origin'] = 'MedHorizon_TMVP_M1'
        sg = [g for g in gold if g['video_id'] in ids]
        write_json(root/f'data/questions/{split}.json', sq)
        gp = f'data/private/{split}_gold.json' if split == 'test' else f'data/targets/{split}.json'
        write_json(root/gp, sg)
        rows = 0
        for vid in ids:
            item = next(v for v in inv if v['video_id']==vid)
            meta = fm[vid]
            for ext in ('zip','json'):
                source = frames/meta['part']/f'{vid}.{ext}'
                dst = root/f'data/frames/{vid}.{ext}'
                stage(source, dst, hardlink=hardlink and ext=='zip')
            actual = sha(root/f'data/frames/{vid}.zip')
            assert actual == meta['zip_sha256'], f'Frame hash mismatch: {vid}'
            assets.append(dict(path=f'data/frames/{vid}.zip', sha256=actual, bytes=meta['zip_bytes']))
            lp = root/('data/private/test_labels' if split=='test' else f'data/labels/{split}')
            for kind in ('phase','instrument'):
                source = workspace/f'data/medhorizon_tmvp/{kind}_annotations/{vid}.txt'
                stage(source, lp/f'{vid}_{kind}.txt')
                assert sha(source) == item[f'{kind}_sha256']
            lab = labels(lp/f'{vid}_phase.txt',lp/f'{vid}_instrument.txt',item['label_offset_seconds'],meta['frames'])
            atomic_npz(lp/f'{vid}.npz', **lab)
            rows += int((lab['phase']>=0).sum())
            if split=='train':
                dq,dg = derive(vid,lab)
                derived_q.extend(dq); derived_g.extend(dg)
            print(f'staged {split}/{vid}',flush=True)
        summary[split] = dict(cases=len(ids),original_qa=len(sq),primary=sum(g['primary_eligible'] for g in sg),label_rows=rows)
    write_json(root/'data/questions/train_derived.json',derived_q)
    write_json(root/'data/targets/train_derived.json',derived_g)
    summary['derived_train_qa'] = len(derived_q)
    summary['frames'] = sum(fm[v]['frames'] for ids in splits.values() for v in ids)
    summary['frame_bytes'] = sum(a['bytes'] for a in assets)
    summary['storage'] = 'NTFS hard links; immutable; copy/rsync materializes ordinary files' if hardlink else 'copies'
    write_json(root/'data/manifests/dataset_report.json',summary)
    stage(workspace/'SurgFormer/checkpoint.pth',root/'assets/surgformer_checkpoint.pth',hardlink)
    assets.append(dict(path='assets/surgformer_checkpoint.pth',sha256=sha(root/'assets/surgformer_checkpoint.pth'),bytes=(root/'assets/surgformer_checkpoint.pth').stat().st_size))
    write_json(root/'data/manifests/assets.json',assets)
    stage(workspace/'reference/TRAINABLE_JOINT_VQA_RESEARCH_20261002.md',root/'references/research_20261002.md')
    return summary
