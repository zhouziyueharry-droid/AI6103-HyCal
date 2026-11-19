"""Numerical and protocol contracts; independent dense and upstream references."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import hycal_core as core


def test_gaussian_normalization_is_shift_invariant_and_finite_for_ties():
    x = torch.tensor([[1., 2., 3.], [4., 4., 4.]])
    expected = torch.tensor([[0.15865525, .5, .84134475], [.5, .5, .5]])
    torch.testing.assert_close(core.normalize_scores(x), expected)
    torch.testing.assert_close(core.normalize_scores(x + 53), expected)


def test_gate_control_separates_activation_from_covariance_changes():
    cosine = torch.tensor([[.8, .2]])
    md = torch.tensor([[.1, .9]])
    counts = torch.tensor([5, 5])
    torch.testing.assert_close(core.fuse_scores(cosine, md, counts, 'H0'), cosine)
    weight = 1 / (1 + math.exp(1))
    expected = (1-weight)*cosine + weight*md
    torch.testing.assert_close(core.fuse_scores(cosine, md, counts, 'H1'), expected)
    torch.testing.assert_close(core.fuse_scores(cosine, md, torch.tensor([1,1]), 'H2'), cosine)


def test_covariance_center_is_normalized_prototype_and_rho_zero_is_baseline():
    x = torch.tensor([[[1., 0.], [0., 1.]]])
    mu, cov = core.class_statistics(x)
    expected_mu = torch.tensor([[2**-.5, 2**-.5]])
    # Official code centers at normalized mean, not the arithmetic mean.
    expected_cov = torch.tensor([[[2-2**.5, 1-2**.5], [1-2**.5, 2-2**.5]]])
    torch.testing.assert_close(mu, expected_mu)
    torch.testing.assert_close(cov, expected_cov)
    torch.testing.assert_close(core.regularize_covariance(cov, 0, .1), expected_cov+.1*torch.eye(2))
    diagonal = core.regularize_covariance(cov, 1, .1)
    assert diagonal[0,0,1].item() == 0
    assert torch.linalg.eigvalsh(diagonal).min() > 0


def test_candidate_query_fusion_and_distance_match_dense_reference():
    torch.manual_seed(22)
    images = torch.randn(7, 4)
    texts = torch.randn(3, 4)
    support = torch.randn(3, 5, 4)
    texts = core.normalize_embeddings(texts)
    images = core.normalize_embeddings(images)
    support = core.normalize_embeddings(support + texts[:,None,:])
    mu, cov = core.class_statistics(support)
    precision = torch.linalg.pinv(core.regularize_covariance(cov, .5, .01))
    cos, md = core.score_features(images, texts, mu, precision, batch_size=2)
    queries = torch.stack([torch.stack([(v+t)/(v+t).norm() for t in texts]) for v in images])
    cos_raw = torch.stack([torch.stack([q@p for q,p in zip(row,mu)]) for row in queries])
    md_raw = torch.stack([torch.stack([-((q-m)@p@(q-m)) for q,m,p in zip(row,mu,precision)]) for row in queries])
    torch.testing.assert_close(cos, core.normalize_scores(cos_raw), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(md, core.normalize_scores(md_raw), atol=2e-6, rtol=2e-6)


def test_macro_accuracy_weights_classes_equally():
    m = core.classification_metrics(torch.tensor([0,0,0,0]), torch.tensor([0,0,0,1]), 2)
    assert m['top1'] == 75.0
    assert m['macro'] == 50.0


def test_single_support_has_finite_precision_and_one_class_normalization():
    mu,cov = core.class_statistics(torch.tensor([[[1.,0.]], [[0.,1.]]]))
    torch.testing.assert_close(cov, .001*torch.eye(2).expand(2,2,2))
    assert torch.isfinite(torch.linalg.pinv(core.regularize_covariance(cov, .5, .0001))).all()
    torch.testing.assert_close(core.normalize_scores(torch.tensor([[3.]])),torch.tensor([[.5]]))


def test_cpu_statistics_match_immutable_upstream_precision_exactly():
    # Catches replacing the official per-class matmul with bmm: a tiny change in
    # covariance can be amplified by inversion in a rank-deficient 5-shot class.
    upstream=Path(__file__).resolve().parents[1]/'external/HyCal-CIL'
    sys.path.insert(0,str(upstream))
    from hycal.hycal_model import HyCalModel
    torch.set_num_threads(8)
    torch.manual_seed(0)
    x=core.normalize_embeddings(torch.randn(3,5,512))
    model=HyCalModel.__new__(HyCalModel)
    model.device='cpu';model.lambda_reg=1e-4;model.real=False;model.gamma=1.
    official=model.compute_class_statistics(x.reshape(-1,512),['a']*5+['b']*5+['c']*5)
    mu,cov=core.class_statistics(x)
    p=torch.stack([torch.pinverse(v) for v in core.regularize_covariance(cov,0,1e-4)])
    for i,name in enumerate(['a','b','c']):
        torch.testing.assert_close(mu[i],official[name][0],atol=0,rtol=0)
        torch.testing.assert_close(p[i],official[name][1],atol=0,rtol=0)
