"""Covariance delta method: independent Jacobian and centering identities."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import propagation_core as pc


def test_projected_covariance_matches_dense_normalization_jacobian():
    generator=torch.Generator().manual_seed(14)
    a=torch.randn(7,7,generator=generator,dtype=torch.float64)
    covariance=a@a.T
    centers=torch.randn(3,7,generator=generator,dtype=torch.float64)
    got=pc.push_covariance(covariance,centers,'projected')
    expected=[]
    for v in centers:
        u=v/v.norm()
        j=(torch.eye(7,dtype=v.dtype)-u[:,None]*u[None,:])/v.norm()
        expected.append(j@covariance@j.T)
    torch.testing.assert_close(got,torch.stack(expected))


def test_push_covariance_annihilates_radial_direction_and_keeps_psd():
    covariance=torch.tensor([[2.,.5],[.5,1.]],dtype=torch.float64)
    centers=torch.tensor([[3.,4.],[-2.,5.]],dtype=torch.float64)
    got=pc.push_covariance(covariance,centers,'projected')
    torch.testing.assert_close(torch.einsum('cij,cj->ci',got,centers),torch.zeros_like(centers),atol=1e-14,rtol=1e-12)
    assert torch.linalg.eigvalsh(got).min()>-1e-12


def test_support_covariance_centering_offset_is_accounted_for():
    raw=torch.tensor([[[1.,0.],[0.,1.]],[[.8,.6],[.6,.8]]],dtype=torch.float64)
    texts=torch.tensor([[1.,0.],[0.,1.]],dtype=raw.dtype)
    prior,radial=pc.propagated_prior(raw,texts,'raw')
    means=raw.mean(1,keepdim=True)
    residual=(raw-means).flatten(0,1)
    pooled=residual.T@residual/(2*(2-1))
    torch.testing.assert_close(prior-radial,pooled.expand_as(prior))
    fused=raw+texts[:,None,:]
    fused=fused/(fused.norm(dim=-1,keepdim=True)+1e-10)
    m=fused.mean(1);u=m/(m.norm(dim=-1,keepdim=True)+1e-10)
    delta=m-u
    torch.testing.assert_close(radial,2*delta[:,:,None]*delta[:,None,:])


def test_jacobian_matches_finite_difference_of_normalize():
    v=torch.tensor([.3,.7,-.2],dtype=torch.float64)
    h=torch.tensor([.4,-.2,.1],dtype=torch.float64)
    eps=1e-6
    finite=((v+eps*h)/(v+eps*h).norm()-(v-eps*h)/(v-eps*h).norm())/(2*eps)
    unit=v/v.norm()
    analytical=(h-unit*torch.dot(unit,h))/v.norm()
    torch.testing.assert_close(finite,analytical,atol=1e-9,rtol=1e-8)


def test_positive_definite_precision_matches_dense_solve():
    g=torch.Generator().manual_seed(7)
    x=torch.randn(3,8,8,generator=g,dtype=torch.float64)
    cov=x@x.transpose(-1,-2)+1e-3*torch.eye(8,dtype=x.dtype)
    expected=torch.linalg.solve(cov,torch.eye(8,dtype=x.dtype).expand_as(cov))
    torch.testing.assert_close(pc.positive_precision(cov),expected)
