"""Analytical few-shot controls and cross-class covariance transfer.

No query labels are accepted. Existing methods are named explicitly; this module
does not imply novelty for GDA, Tip-Adapter, ProKeR or regularized discriminants.
"""
import torch
import torch.nn.functional as F
from hycal_core import normalize_embeddings


def gda_parameters(support):
    """ICLR24 GDA estimator, with ordinary class means and pooled residuals."""
    c,k,d=support.shape
    mean=support.mean(1)
    residual=(support-mean[:,None,:]).reshape(c*k,d)
    covariance=torch.cov(residual.T)
    precision=d*torch.linalg.pinv((c*k-1)*covariance+covariance.trace()*torch.eye(d,device=support.device,dtype=support.dtype))
    return mean,precision


def gda_scores(query,mean,precision):
    weights=precision@mean.T
    bias=-.5*(mean.T*weights).sum(0)
    return query@weights+bias


def neighbor_prior(cov,texts,neighbors,mode='semantic',random_seed=20261116):
    c=len(cov)
    if neighbors==0:
        return cov.mean(0,keepdim=True).expand_as(cov)
    k=min(neighbors,c-1)
    if mode=='semantic':
        affinity=texts@texts.T
    elif mode=='random':
        generator=torch.Generator(device='cpu').manual_seed(random_seed)
        affinity=torch.rand((c,c),generator=generator,dtype=texts.dtype).to(texts.device)
    else:
        raise ValueError(mode)
    affinity=affinity.clone()
    affinity.fill_diagonal_(-torch.inf)
    indices=affinity.topk(k,dim=1).indices
    # Uniform top-k weights avoid adding a separately tuned softmax temperature.
    weights=torch.zeros((c,c),device=cov.device,dtype=cov.dtype)
    weights.scatter_(1,indices,1/k)
    return (weights@cov.reshape(c,-1)).reshape_as(cov)


def rda_covariance(cov,texts,alpha,neighbors,ridge,mode='semantic'):
    prior=neighbor_prior(cov,texts,neighbors,mode)
    return (1-alpha)*cov+alpha*prior+ridge*torch.eye(cov.shape[-1],device=cov.device,dtype=cov.dtype)


def rotate_residuals(values,source,target):
    """Minimal rotation source->target; tangent vectors follow sphere transport.

    Values may include radial residuals: an orthogonal rotation preserves them.
    An antipodal pair has no unique minimal transport and is rejected explicitly.
    """
    a=source.unsqueeze(-2);b=target.unsqueeze(-2)
    cosine=(a*b).sum(-1,keepdim=True)
    if torch.any(cosine < -1+1e-6): raise ValueError('Antipodal prototypes have ambiguous transport')
    av=(values*a).sum(-1,keepdim=True)
    bv=(values*b).sum(-1,keepdim=True)
    return values+av*(b-a)-(a+b)*(bv-cosine*av)/(1+cosine)


def transported_prior(residuals,prototypes,texts,neighbors):
    c,k,d=residuals.shape
    if neighbors==0:
        indices=torch.arange(c,device=texts.device).expand(c,c)
    else:
        affinity=texts@texts.T;affinity.fill_diagonal_(-torch.inf)
        indices=affinity.topk(min(neighbors,c-1),dim=1).indices
    priors=[]
    for target,ids in enumerate(indices):
        rotated=rotate_residuals(residuals[ids],prototypes[ids],prototypes[target].expand(len(ids),d))
        z=rotated.reshape(-1,d)
        priors.append(z.T@z/(len(ids)*(k-1)))
    return torch.stack(priors)


def kernel(x,y,beta):
    return torch.exp(-beta*(1-x@y.T))


def proker_fit(support,labels,texts,beta,lam):
    k=kernel(support,support,beta)
    target=F.one_hot(labels,len(texts)).to(support)-support@texts.T
    return torch.linalg.solve(k/lam+torch.eye(len(support),device=support.device,dtype=support.dtype),target)


def proker_scores(query,support,texts,coeff,beta):
    return query@texts.T+kernel(query,support,beta)@coeff


def tip_scores(query,support,labels,texts,beta,alpha):
    values=F.one_hot(labels,len(texts)).to(support)
    return 100*(query@texts.T)+alpha*(kernel(query,support,beta)@values)


def quadratic_scores(images,texts,prototypes,precisions,batch_size=128,logdet=None):
    """Keep the full class-conditioned quadratic; shared P does not make this LDA."""
    cosines,distances=[],[]
    for start in range(0,len(images),batch_size):
        query=normalize_embeddings(images[start:start+batch_size,None,:]+texts[None,:,:])
        cosines.append((query*prototypes[None,:,:]).sum(-1))
        diff=(query-prototypes[None,:,:]).transpose(0,1).contiguous()
        md=-(torch.bmm(diff,precisions)*diff).sum(-1).T
        if logdet is not None:
            md=md-logdet[None,:]
        distances.append(md)
    return torch.cat(cosines),torch.cat(distances)
