"""Delta-method covariance propagation for candidate-conditioned normalization.

This is an explicitly exploratory extension, not a claim of novel mathematics.
"""
import torch
from hycal_core import normalize_embeddings


def positive_precision(covariance):
    """Ridge makes each covariance SPD; use its inverse via Cholesky."""
    return torch.cholesky_inverse(torch.linalg.cholesky(covariance))


def push_covariance(covariance,centers,mode='projected'):
    c,d=centers.shape
    if mode=='raw': return covariance.expand(c,d,d)
    norm=centers.norm(dim=-1).clamp_min(1e-8)
    if mode=='scale_only': return covariance[None,:,:]/norm[:,None,None].square()
    if mode!='projected': raise ValueError(mode)
    u=centers/norm[:,None]
    su=u@covariance
    radial=(su*u).sum(-1)
    projected=(covariance[None,:,:]-u[:,:,None]*su[:,None,:]-su[:,:,None]*u[:,None,:]
               +radial[:,None,None]*u[:,:,None]*u[:,None,:])
    return projected/norm[:,None,None].square()


def propagated_prior(raw_support,texts,mode='projected'):
    c,k,d=raw_support.shape
    mean=raw_support.mean(1)
    residual=(raw_support-mean[:,None,:]).reshape(c*k,d)
    pooled=residual.T@residual/(c*(k-1))
    prior=push_covariance(pooled,mean+texts,mode)
    fused=normalize_embeddings(raw_support+texts[:,None,:])
    mf=fused.mean(1);prototype=normalize_embeddings(mf)
    offset=mf-prototype
    radial=k/(k-1)*offset[:,:,None]*offset[:,None,:]
    return prior+radial,radial
