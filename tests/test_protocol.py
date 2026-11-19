"""Tests guard against incomplete seed selection and inconsistent label averaging."""
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_experiments import select


def test_selection_uses_macro_average_and_prefers_smaller_rho_on_tie():
    cfg={'rho_candidates':[0.,.5,1.], 'seeds':[0,1,42,1993],
         'selection_metric':'macro', 'selection_tie_break':'smaller rho'}
    scores={0.:[50,50,50,50],.5:[60,40,60,40],1.:[49,49,49,49]}
    rows=[{'method':'H2','seed':seed,'rho':rho,'macro':value,'top1':99 if rho==1 else 0}
          for rho,values in scores.items() for seed,value in zip(cfg['seeds'],values)]
    result=select(rows,cfg,{})
    assert result['selected_rho']==0.
    assert result['validation_macro_means']['0.5']==50


def test_selection_rejects_missing_or_duplicate_seed():
    cfg={'rho_candidates':[0.], 'seeds':[0,1,42,1993],
         'selection_metric':'macro', 'selection_tie_break':'smaller rho'}
    rows=[{'method':'H2','seed':seed,'rho':0.,'macro':90} for seed in [0,1,42]]
    with pytest.raises(ValueError,match='all pre-registered seeds'):
        select(rows,cfg,{})
    rows.append(dict(rows[-1]))
    with pytest.raises(ValueError,match='all pre-registered seeds'):
        select(rows,cfg,{})
