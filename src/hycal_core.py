"""Cache-based HyCal; all candidate classes must have learned statistics.

Preserves upstream normalized-prototype centering and sample std correction.
Candidate-conditioned query embeddings never consume query labels.
"""
import math
import torch


def normalize_embeddings(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-10)


def normalize_scores(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return torch.full_like(x, .5)
    std = x.std(dim=1, keepdim=True, correction=1).clamp_min(1e-6)
    z = (x - x.mean(dim=1, keepdim=True)) / std
    return .5 * (1 + torch.erf(z / math.sqrt(2)))


def class_statistics(support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Input [classes, K, D] contains normalized image-text fused features."""
    if support.ndim != 3 or support.shape[1] < 1:
        raise ValueError('support must have shape [classes,K,D] with K>=1')
    mu = normalize_embeddings(support.mean(dim=1))
    if support.shape[1] == 1:
        cov = .001 * torch.eye(support.shape[-1], device=support.device, dtype=support.dtype)
        return mu, cov.expand(support.shape[0], -1, -1).clone()
    centered = support - mu[:, None, :]
    # Retain upstream per-class GEMM reduction order; bmm roundoff can be
    # amplified by a nearly singular low-shot covariance's pseudoinverse.
    cov = torch.stack([z.T @ z / (support.shape[1]-1) for z in centered])
    return mu, cov


def regularize_covariance(cov: torch.Tensor, rho: float, lambda_reg: float) -> torch.Tensor:
    if not 0 <= rho <= 1 or lambda_reg <= 0:
        raise ValueError('rho must be in [0,1] and lambda_reg positive')
    diag = torch.diag_embed(cov.diagonal(dim1=-2,dim2=-1))
    eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
    return (1-rho)*cov + rho*diag + lambda_reg*eye


def fuse_scores(cosine: torch.Tensor, md: torch.Tensor, counts: torch.Tensor,
                method: str, n_min_p: int=10, scale: float=5) -> torch.Tensor:
    if method == 'C0':
        return cosine.clone()
    if method == 'MD':
        return md.clone()
    if method not in {'H0','H1','H2'}:
        raise ValueError(method)
    weights = torch.sigmoid((counts.to(cosine)-n_min_p)/scale)
    enabled = counts >= n_min_p if method == 'H0' else counts > 1
    weights = weights * enabled.to(weights)
    return (1-weights[None,:])*cosine + weights[None,:]*md


@torch.no_grad()
def score_features(images: torch.Tensor, texts: torch.Tensor, prototypes: torch.Tensor,
                   precisions: torch.Tensor, batch_size: int=128,
                   background: tuple[torch.Tensor,torch.Tensor] | None=None
                   ) -> tuple[torch.Tensor,torch.Tensor]:
    cosines, distances = [], []
    for start in range(0,len(images),batch_size):
        query = normalize_embeddings(images[start:start+batch_size,None,:] + texts[None,:,:])
        cos_raw = torch.einsum('bcd,cd->bc',query,prototypes)
        diff = (query-prototypes[None,:,:]).transpose(0,1).contiguous()
        md_raw = -(torch.bmm(diff,precisions)*diff).sum(dim=-1).T
        if background is not None:
            bg_mu,bg_precision = background
            bg_diff = query[:,0,:]-bg_mu
            md_raw = md_raw + ((bg_diff@bg_precision)*bg_diff).sum(-1,keepdim=True)
        cosines.append(normalize_scores(cos_raw).cpu())
        distances.append(normalize_scores(md_raw).cpu())
    return torch.cat(cosines),torch.cat(distances)


def classification_metrics(predictions: torch.Tensor, labels: torch.Tensor,
                           n_classes: int) -> dict:
    p,y = predictions.cpu().long(), labels.cpu().long()
    if len(p) != len(y) or len(y)==0:
        raise ValueError('predictions and labels must be non-empty and equal length')
    confusion = torch.bincount(y*n_classes+p,minlength=n_classes*n_classes).reshape(n_classes,n_classes)
    counts = confusion.sum(1)
    correct = confusion.diag()
    per_class = 100*correct.double()/counts.clamp_min(1)
    return {'top1':100*correct.sum().item()/len(y),
            'macro':per_class[counts>0].mean().item(),
            'n':len(y),'per_class':per_class.tolist(),
            'class_counts':counts.tolist(),'confusion':confusion.tolist()}
