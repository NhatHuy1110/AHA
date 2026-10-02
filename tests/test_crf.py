"""Workflow CRF: the parallel scan must equal brute-force enumeration."""
import itertools
import torch
from aha.crf import WorkflowCRF

def brute(crf, emit):
    t, k = emit.shape
    scores = {}
    for path in itertools.product(range(k), repeat=t):
        s = crf.start[path[0]] + emit[0, path[0]]
        for i in range(1, t):
            s = s + crf.transition[path[i-1], path[i]] + emit[i, path[i]]
        scores[path] = s
    return scores

def make(t=5, k=3):
    torch.manual_seed(0)
    crf = WorkflowCRF(k)
    with torch.no_grad():
        crf.transition.normal_(); crf.start.normal_()
    return crf, torch.randn(t, k)

def test_partition_and_posteriors_match_enumeration():
    crf, emit = make()
    scores = brute(crf, emit)
    log_z = torch.logsumexp(torch.stack(list(scores.values())), 0)
    assert torch.allclose(crf.log_partition(emit), log_z, atol=1e-4)
    phase, onset = crf.posteriors(emit)
    prob = {p: (s-log_z).exp() for p, s in scores.items()}
    for t in range(5):
        for j in range(3):
            assert torch.isclose(phase[t, j].exp(), sum(v for p, v in prob.items() if p[t] == j), atol=1e-4)
            if t:
                expected = sum(v for p, v in prob.items() if p[t] == j and p[t-1] != j)
                assert torch.isclose(onset[t, j].exp(), expected, atol=1e-4)
    assert (onset[0].exp() < 1e-6).all()

def test_nll_matches_enumeration_with_unlabelled_seconds():
    crf, emit = make()
    labels = torch.tensor([-1, 0, 0, 2, -1])
    scores = brute(crf, emit)
    log_z = torch.logsumexp(torch.stack(list(scores.values())), 0)
    ok = [s for p, s in scores.items() if p[1] == 0 and p[2] == 0 and p[3] == 2]
    expected = (log_z - torch.logsumexp(torch.stack(ok), 0)) / 5
    assert torch.isclose(crf.nll(emit, labels), expected, atol=1e-4)

def test_odd_lengths_single_second_and_gradients():
    crf, _ = make()
    for t in (1, 2, 7, 33):
        emit = torch.randn(t, 3, requires_grad=True)
        phase, onset = crf.posteriors(emit)
        assert phase.shape == onset.shape == (t, 3)
        assert torch.allclose(phase.exp().sum(1), torch.ones(t), atol=1e-4)
        loss = crf.nll(emit, torch.zeros(t, dtype=torch.long)) + onset.sum()*1e-3
        loss.backward()
        assert torch.isfinite(emit.grad).all()

def test_onset_posterior_peaks_at_the_transition():
    crf = WorkflowCRF(2)
    emit = torch.full((40, 2), -6.); emit[:20, 0] = 0; emit[20:, 1] = 0
    _, onset = crf.posteriors(emit)
    assert int(onset[:, 1].argmax()) == 20 and onset[20, 1].exp() > 0.9
