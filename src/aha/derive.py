"""Label-derived phase-onset QA (v2) for train, validation and test.

Same conservative single-choice rule as v1, but (a) written with the natural
MedHorizon phrasings instead of raw class names and (b) also produced for the
held-out splits, where it is an evaluation-only secondary benchmark. Nothing
here reads model outputs; held-out gold is written next to the other gold.
"""
from pathlib import Path
import numpy as np
from .common import PHASES, TOOLS, PHASE_NAMES, TOOL_NAMES, read_json, write_json, qa_files, label_dir

TEMPLATES = [
    'At phase onset for {p}, which instrument is present?',
    'At the beginning of {p}, which surgical instrument appears in view?',
    'At the onset of the {p} phase, which surgical instrument is visible?',
    'For the start of the {p} phase, which instrument is observed?',
    'What surgical instrument is visible when {p} first begins?',
    'When the {p} phase begins, which instrument is present in the operative field?',
    'Which instrument can be identified at the start of the {p} phase?',
    'Which instrument is in the field at the transition into the {p} phase?',
    'As the procedure enters the {p} phase, what instrument is seen?',
    'Which instrument is present during the opening moments of the {p} phase?']

def derive_video(video_id, lab):
    """An answer is present at EVERY fully observed onset; distractors absent at every one."""
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
            template = TEMPLATES[int(rng.integers(len(TEMPLATES)))]
            qid = f'derived2_{video_id}_{pi}_{a}'
            qs.append(dict(qid=qid, video_id=video_id, target_phase=phase,
                question=template.format(p=PHASE_NAMES[phase]),
                options=[f'{chr(65+i)}. {TOOL_NAMES[TOOLS[o]]}' for i, o in enumerate(opts)],
                option_tools=[TOOLS[o] for o in opts], origin='derived_v2_labels'))
            flags = ['repeated_phase'] if len(ts) > 1 else []
            gs.append(dict(qid=qid, video_id=video_id, answer=chr(65+answer),
                answer_tool=TOOLS[a], target_phase=phase, onsets=ts.tolist(),
                supported_onsets=ts.tolist(), primary_eligible=True, flags=flags,
                gold_level='annotation_derived', origin='derived_v2_labels'))
    return qs, gs

def derive(root):
    root = Path(root)
    splits = read_json(root/'data/manifests/splits.json')
    report = {}
    for split, ids in splits.items():
        qs, gs = [], []
        for vid in ids:
            with np.load(label_dir(root, split)/f'{vid}.npz') as z:
                q, g = derive_video(vid, {k: z[k] for k in z.files})
            qs += q; gs += g
        qp, gp = qa_files(root, split, 'derived')
        write_json(qp, qs); write_json(gp, gs)
        report[split] = dict(qa=len(qs), cases=len({q['video_id'] for q in qs}),
                             repeated=sum('repeated_phase' in g['flags'] for g in gs))
    write_json(root/'data/manifests/derived_v2_report.json', report)
    return report
