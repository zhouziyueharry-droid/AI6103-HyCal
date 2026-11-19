"""HyCal on cached CLIP features.

A line-by-line port of the official scoring logic (third_party/HyCal-CIL/hycal/
hycal_model.py + hycal_evaluator.py) that works on pre-extracted, L2-normalised image
and text embeddings instead of re-encoding images.  Supports incremental sessions
(class statistics are kept; background statistics are updated recursively), so the same
module can be used for the merged three-domain experiment.

Fusion 'sum' (paper setting):
  support:  e = norm(img + text[label])                         (label is a Support label)
  query:    e_c = norm(img + text[c]) for EVERY candidate c     (no test label used)
Scores (per query, normalised across candidate classes with z-score -> Gaussian CDF):
  cosine:   <e_c, mu_c>
  md:       RMD_c = d_bg(e_ref) - d_c(e_c),  Mahalanobis with regularised precision
  weighted: (1-w_c)*cos + w_c*md, w_c = sigmoid((N_c - n_min_p)/scale), only if N_c >= n_min_p
  avg:      0.5*(cos+md), only if N_c >= n_min_p
"""
import math
from typing import Dict, List, Optional

import torch


def _norm(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-10)


def _cdf_norm(s):
    """Per-sample z-score across classes, then Gaussian CDF (official _normalize_rmd_scores)."""
    z = (s - s.mean(dim=1, keepdim=True)) / torch.clamp(s.std(dim=1, keepdim=True), min=1e-6)
    return 0.5 * (1 + torch.erf(z / math.sqrt(2)))


class HyCalCached:
    def __init__(self, fusion: Optional[str] = "sum", lambda_reg: float = 1e-4,
                 lambda_bg: float = 1e-4, device: str = "cpu", shrink_alpha: float = 0.0):
        """shrink_alpha is NOT part of official HyCal (0 = official behaviour).  With alpha > 0 each
        class covariance is replaced by (1-alpha)*cov_c + alpha*cov_pooled, where cov_pooled is the
        within-class covariance pooled over the classes of the same session (needs K >= 2)."""
        assert fusion in (None, "sum", "concat")
        self.fusion, self.lambda_reg, self.lambda_bg, self.device = fusion, lambda_reg, lambda_bg, device
        self.shrink_alpha = shrink_alpha
        self.class_ids: List[str] = []           # global ids in arrival order, e.g. "eurosat:3"
        self.mu: Dict[str, torch.Tensor] = {}
        self.P: Dict[str, torch.Tensor] = {}
        self.N: Dict[str, int] = {}
        self.text: Dict[str, torch.Tensor] = {}
        self.bg = {"mu": None, "P": None, "N": 0}

    # ---------------------------------------------------------------- statistics
    def _precision(self, cov, lam):
        d = cov.shape[0]
        reg = cov + lam * torch.eye(d, device=self.device)
        try:
            return torch.pinverse(reg)             # official code path
        except RuntimeError:
            # SVD can fail to converge when many singular values coincide (strong regularisation).
            # reg is symmetric positive definite, so the plain inverse is the same matrix.
            return torch.linalg.inv(reg.double()).float()

    def add_session(self, img_feats, labels: List[str], text_feats: Dict[str, torch.Tensor]):
        """Learn one session. labels are global class ids; text_feats maps id -> [D]."""
        img_feats = img_feats.to(self.device).float()
        for cid in dict.fromkeys(labels):
            self.text[cid] = text_feats[cid].to(self.device).float()
        if self.fusion == "sum":
            emb = _norm(img_feats + torch.stack([self.text[c] for c in labels]))
        elif self.fusion == "concat":
            emb = _norm(torch.cat([img_feats, torch.stack([self.text[c] for c in labels])], dim=-1))
        else:
            emb = img_feats

        stats, scatter, dof = {}, 0.0, 0
        for cid in sorted(set(labels)):           # official: np.unique order
            e = emb[torch.tensor([l == cid for l in labels], device=self.device)]
            n = e.shape[0]
            mu = _norm(e.mean(dim=0))
            if n > 1:
                centered = e - mu.unsqueeze(0)    # centred on the *normalised* prototype, as official
                cov = centered.T @ centered / (n - 1)
                scatter, dof = scatter + centered.T @ centered, dof + n - 1
            else:
                cov = torch.eye(mu.shape[0], device=self.device) * 1e-3   # official K=1 fallback
            stats[cid] = (mu, cov, n)
        for cid, (mu, cov, n) in stats.items():
            if self.shrink_alpha > 0 and n > 1 and dof > 0:
                cov = (1 - self.shrink_alpha) * cov + self.shrink_alpha * (scatter / dof)
            self.mu[cid], self.P[cid], self.N[cid] = mu, self._precision(cov, self.lambda_reg), n
            if cid not in self.class_ids:
                self.class_ids.append(cid)
        self._update_bg(emb)

    def _update_bg(self, emb):
        n_new = emb.shape[0]
        if n_new == 0:
            return
        if self.bg["mu"] is None:
            mu = emb.mean(dim=0)
            P = None
            if n_new > 1:
                c = emb - mu.unsqueeze(0)
                P = self._precision(c.T @ c / (n_new - 1), self.lambda_bg)
            self.bg = {"mu": mu, "P": P, "N": n_new}
            return
        n_prev, mu_prev = self.bg["N"], self.bg["mu"]
        mu_new = emb.mean(dim=0)
        mu_upd = (n_prev * mu_prev + n_new * mu_new) / (n_prev + n_new)
        if n_new > 1 and n_prev > 0:
            if self.bg["P"] is not None:
                cov_prev = torch.inverse(self.bg["P"])
            else:
                cov_prev = torch.eye(mu_upd.shape[0], device=self.device) * self.lambda_bg
            c = emb - mu_new.unsqueeze(0)
            cov_new = c.T @ c / (n_new - 1)
            diff = (mu_prev - mu_new).unsqueeze(1)
            cross = (n_prev * n_new) / (n_prev + n_new) * (diff @ diff.T)
            tot = n_prev + n_new - 1
            cov_upd = (n_prev - 1) / tot * cov_prev + (n_new - 1) / tot * cov_new + cross / tot
            P_upd = self._precision(cov_upd, self.lambda_bg)
        else:
            P_upd = self.bg["P"]
        self.bg = {"mu": mu_upd, "P": P_upd, "N": n_prev + n_new}

    # ---------------------------------------------------------------- scoring
    @torch.no_grad()
    def score(self, img_feats, batch_size: int = 1024):
        """Return normalised score matrices over all learned classes (column order = self.class_ids).

        {'cos': [N, C], 'md': [N, C]}  -- both already z-score/CDF normalised per query.
        """
        ids = self.class_ids
        protos = torch.stack([self.mu[c] for c in ids])
        text = torch.stack([self.text[c] for c in ids])
        cos_all, md_all = [], []
        for i in range(0, img_feats.shape[0], batch_size):
            x = img_feats[i:i + batch_size].to(self.device).float()
            if self.fusion == "sum":
                e = _norm(x.unsqueeze(1) + text.unsqueeze(0))            # [B, C, D]
                cos = torch.einsum("bcd,cd->bc", e, protos)
                ref = e[:, 0, :]                                         # bg_ref_class='first'
            elif self.fusion == "concat":
                e = _norm(torch.cat([x.unsqueeze(1).expand(-1, len(ids), -1),
                                     text.unsqueeze(0).expand(x.shape[0], -1, -1)], dim=-1))
                cos = torch.einsum("bcd,cd->bc", e, protos)
                ref = e[:, 0, :]
            else:
                e = x.unsqueeze(1).expand(-1, len(ids), -1)
                cos = x @ protos.T
                ref = x
            cos_all.append(_cdf_norm(cos).cpu())
            if self.bg["P"] is not None:
                d = ref - self.bg["mu"].unsqueeze(0)
                bg = ((d @ self.bg["P"]) * d).sum(dim=1)
                raw = torch.zeros(x.shape[0], len(ids), device=self.device)
                for c, cid in enumerate(ids):
                    dc = e[:, c, :] - protos[c].unsqueeze(0)
                    raw[:, c] = bg - ((dc @ self.P[cid]) * dc).sum(dim=1)
                md_all.append(_cdf_norm(raw).cpu())
        out = {"cos": torch.cat(cos_all)}
        if md_all:
            out["md"] = torch.cat(md_all)
        return out

    def combine(self, s, metric: str, n_min_p: int = 20, scale: float = 5.0):
        """Fuse normalised cosine / Mahalanobis scores exactly as the official evaluator."""
        if metric == "cosine" or "md" not in s:
            return s["cos"]
        if metric == "md":
            return s["md"]
        counts = torch.tensor([self.N[c] for c in self.class_ids], dtype=torch.float32)
        prefer = counts >= n_min_p
        if metric == "avg":
            w = torch.full_like(counts, 0.5)
        elif metric == "weighted":
            w = torch.sigmoid((counts - n_min_p) / scale)
        else:
            raise ValueError(metric)
        w = torch.where(prefer, w, torch.zeros_like(w))
        return (1 - w) * s["cos"] + w * s["md"]


def zero_shot_scores(img_feats, text_feats):
    """Plain CLIP zero-shot: image-text cosine (no Support images)."""
    return img_feats.float() @ text_feats.float().T
