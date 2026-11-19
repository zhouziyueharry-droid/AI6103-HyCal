"""Round2 numerical contracts: independent closed-form references, no mocked models."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import round2_core as r2


def test_shared_covariance_removes_between_class_location():
    x=torch.tensor([[[0.,0.],[2.,0.]],[[100.,3.],[102.,3.]]],dtype=torch.float64)
    mu,cov=r2.gda_parameters(x)
    torch.testing.assert_close(mu,torch.tensor([[1.,0.],[101.,3.]],dtype=x.dtype))
    residual=torch.tensor([[-1.,0.],[1.,0.],[-1.,0.],[1.,0.]],dtype=x.dtype)
    s=residual.T@residual/3
    p=2*torch.linalg.pinv(3*s+torch.trace(s)*torch.eye(2,dtype=x.dtype))
    torch.testing.assert_close(cov,p)


def test_semantic_neighbor_prior_excludes_self_and_follows_permutation():
    cov=torch.stack([torch.eye(2)*v for v in [1.,3.,9.]])
    texts=torch.tensor([[1.,0.],[.9,.1],[-1.,0.]])
    prior=r2.neighbor_prior(cov,texts,1)
    torch.testing.assert_close(prior[0],cov[1])
    torch.testing.assert_close(prior[1],cov[0])
    torch.testing.assert_close(prior[2],cov[1])
    perm=torch.tensor([2,0,1])
    torch.testing.assert_close(r2.neighbor_prior(cov[perm],texts[perm],1),prior[perm])


def test_official_proker_scale_matches_independent_dense_solution():
    x=torch.eye(3,dtype=torch.float64)
    text=torch.tensor([[1.,0.,0.],[0.,1.,0.]],dtype=x.dtype)
    y=torch.tensor([0,1,0])
    beta,lam=2.,.3
    coeff=r2.proker_fit(x,y,text,beta,lam)
    k=torch.exp(-beta*(1-x@x.T))
    residual=torch.nn.functional.one_hot(y,2).to(x)-x@text.T
    expected=lam*torch.linalg.solve(k+lam*torch.eye(3,dtype=x.dtype),residual)
    torch.testing.assert_close(coeff,expected)


def test_tip_adapter_zero_alpha_is_clip_logits():
    x=torch.eye(2)
    y=torch.tensor([0,1])
    q=torch.tensor([[.8,.6]])
    torch.testing.assert_close(r2.tip_scores(q,x,y,x,beta=5.,alpha=0.),100*q@x.T)


def test_rda_zero_pool_matches_local_ridge_and_full_pool_is_shared():
    cov=torch.stack([torch.eye(3),2*torch.eye(3)])
    text=torch.eye(2,3)
    local=r2.rda_covariance(cov,text,alpha=0.,neighbors=0,ridge=.1)
    torch.testing.assert_close(local,cov+.1*torch.eye(3))
    shared=r2.rda_covariance(cov,text,alpha=1.,neighbors=0,ridge=.1)
    torch.testing.assert_close(shared,torch.ones(2,1,1)*(1.6*torch.eye(3)))


def test_candidate_conditioned_quadratic_keeps_query_norm_term():
    q=torch.tensor([[1.,2.],[3.,1.]],dtype=torch.float64)
    proto=torch.tensor([[.2,.8],[.7,.1]],dtype=q.dtype)
    texts=torch.tensor([[.5,0.],[0.,.5]],dtype=q.dtype)
    p=torch.stack([torch.diag(torch.tensor([2.,1.],dtype=q.dtype))]*2)
    cos,md=r2.quadratic_scores(q,texts,proto,p)
    z=q[:,None,:]+texts[None,:,:]
    z=z/(z.norm(dim=-1,keepdim=True)+1e-10)
    expected=torch.stack([-torch.sum((z[:,c]-proto[c])@p[c]*(z[:,c]-proto[c]),dim=-1) for c in range(2)],dim=1)
    torch.testing.assert_close(md,expected)
    torch.testing.assert_close(cos,(z*proto[None,:,:]).sum(-1))


def test_sphere_rotation_maps_source_to_target_and_preserves_gram_matrix():
    a=torch.tensor([1.,0.,0.],dtype=torch.float64)
    b=torch.tensor([0.,1.,0.],dtype=torch.float64)
    values=torch.tensor([[1.,0.,0.],[.5,.7,.2],[-.1,.3,.9]],dtype=torch.float64)
    rotated=r2.rotate_residuals(values,a,b)
    torch.testing.assert_close(rotated[0],b)
    torch.testing.assert_close(rotated@rotated.T,values@values.T)
    torch.testing.assert_close(r2.rotate_residuals(values,a,a),values)


def test_transport_pool_matches_explicit_rotated_covariance():
    prototypes=torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float64)
    residual=torch.tensor([[[.1,.2],[-.1,-.2]],[[.3,.1],[-.3,-.1]]],dtype=torch.float64)
    texts=prototypes.clone()
    prior=r2.transported_prior(residual,prototypes,texts,1)
    rotation=torch.tensor([[0.,1.],[-1.,0.]],dtype=torch.float64)
    s=residual[1].T@residual[1]
    torch.testing.assert_close(prior[0],rotation@s@rotation.T)
