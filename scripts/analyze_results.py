"""Aggregate frozen-protocol results; creates figures and machine-readable analysis."""
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/part_a'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':130})


def display_method(row):
    if row['method']=='H1' and row['lambda_reg']!=.0001:
        return 'H1 tuned lambda'
    return row['method']


def summarize(rows):
    groups={}
    for row in rows:
        key=(row['shot'],display_method(row))
        groups.setdefault(key,[]).append(row)
    result=[]
    for (shot,method),group in sorted(groups.items()):
        row={'shot':shot,'method':method,'seeds':[g['seed'] for g in group],'n_seeds':len(group)}
        for metric in ['top1','macro']:
            values=[r[metric] for r in group]
            row[metric+'_mean']=statistics.mean(values)
            row[metric+'_sd']=statistics.stdev(values) if len(values)>1 else 0.
        result.append(row)
    return result


def paired(rows,method,reference,shot=5):
    a={r['seed']:r for r in rows if r['shot']==shot and display_method(r)==method}
    b={r['seed']:r for r in rows if r['shot']==shot and display_method(r)==reference}
    assert a.keys()==b.keys()
    result={'method':method,'reference':reference,'shot':shot,'n_seeds':len(a)}
    for metric in ['top1','macro']:
        diffs=[a[s][metric]-b[s][metric] for s in sorted(a)]
        mean=statistics.mean(diffs)
        sd=statistics.stdev(diffs) if len(diffs)>1 else 0
        half=float(t.ppf(.975,len(diffs)-1))*sd/math.sqrt(len(diffs)) if len(diffs)>1 else 0
        result[metric]={'mean_delta_pp':mean,'sd_delta_pp':sd,'ci95_support_seeds':[mean-half,mean+half],
                        'wins':sum(v>1e-10 for v in diffs),'ties':sum(abs(v)<=1e-10 for v in diffs),
                        'losses':sum(v<-1e-10 for v in diffs),'seed_deltas':dict(zip(map(str,sorted(a)),diffs))}
    return result


def predictions(row,directory):
    with (directory/(row['tag']+'_predictions.csv')).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    primary=json.loads((ROOT/'results/part_a/test/summary_k5_1_10.json').read_text())
    replication=json.loads((ROOT/'results/part_a/replication/summary_k5.json').read_text())
    selection=json.loads((ROOT/'results/part_a/selection.json').read_text())
    records=json.loads((ROOT/'data/manifests/flowers102/records.json').read_text())
    labels={r['label']:r['classname'] for r in records}
    byid={r['sample_id']:r for r in records}
    analysis={'primary_summary':summarize(primary),'replication_summary':summarize(replication),
              'primary_paired':[paired(primary,'H2',m) for m in ['H0','H1','H1 tuned lambda']],
              'replication_paired':[paired(replication,'H2',m) for m in ['H0','H1','H1 tuned lambda']],
              'selected_rho':selection['selected_rho'],
              'ci_scope':'Student t interval over support sampling seeds, conditional on this fixed dataset split; not a cross-domain or independent-dataset interval.'}
    for name,rows in [('primary',primary),('replication',replication)]:
        summary=summarize(rows)
        with (OUT/(name+'_summary.csv')).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0]))
            writer.writeheader();writer.writerows(summary)

    # Main result chart: standard deviations across seeds, not confidence intervals.
    methods=['H0','H1','H1 tuned lambda','H2']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,rows,title in zip(axes,[primary,replication],['Primary: 4 seeds','Replication: 16 new seeds']):
        summary={r['method']:r for r in summarize(rows) if r['shot']==5}
        for j,m in enumerate(methods):
            ax.errorbar(j,summary[m]['macro_mean'],yerr=summary[m]['macro_sd'],fmt='o',
                        capsize=5,markersize=8,color=['#8796A5','#4A7F9C','#8CAFC0','#177E72'][j])
            ax.annotate(f"{summary[m]['macro_mean']:.3f}",(j,summary[m]['macro_mean']),
                        xytext=(9,0),textcoords='offset points',va='center',fontsize=9)
        ax.set_xticks(range(4),['H0\nOfficial','H1\nSoft gate','H1\nTuned ridge','H2\nDiagonal shrink'])
        ax.set_xlim(-.5,3.7)
        ax.set_ylim(91.5,94.5)
        ax.set_ylabel('Class-macro accuracy (%)');ax.set_title(title)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Flowers102 5-shot | Frozen CLIP ViT-B/16 | Mean +/- sample SD')
    fig.tight_layout();fig.savefig(OUT/'main_results.png',bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,4))
    combined=primary+replication
    colors={'H0':'#8796A5','H1':'#4A7F9C','H1 tuned lambda':'#D18D42'}
    for j,m in enumerate(['H0','H1','H1 tuned lambda']):
        d=paired(combined,'H2',m)['macro']['seed_deltas']
        ax.scatter(np.arange(len(d))+(j-1)*.18,list(d.values()),label='H2 minus '+m,s=24,color=colors[m])
    ax.axhline(0,color='black',lw=.8)
    ax.set_xticks(range(len(d)),list(d))
    ax.set_ylabel('Class-macro difference (percentage points)');ax.set_xlabel('Support seed')
    ax.set_title('Paired 5-shot differences; all methods share each seed\'s support')
    ax.legend(ncol=3);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(OUT/'paired_differences.png',bbox_inches='tight');plt.close(fig)

    # Aggregate class-wise effects over primary seeds; image repeats across seeds are retained as repeated predictions.
    class_rows=[]
    matrices={}
    for m in ['H0','H1','H2']:
        selected=[r for r in primary if r['shot']==5 and display_method(r)==m]
        raw=[json.loads((ROOT/'results/part_a/test'/(r['tag']+'.json')).read_text()) for r in selected]
        matrices[m]=sum(np.asarray(v['metrics']['confusion']) for v in raw)
    acc={m:100*np.diag(mat)/mat.sum(1) for m,mat in matrices.items()}
    for label in range(102):
        class_rows.append({'label':label,'classname':labels[label],'H0':acc['H0'][label],'H1':acc['H1'][label],
                           'H2':acc['H2'][label],'H2_minus_H0':acc['H2'][label]-acc['H0'][label],
                           'H2_minus_H1':acc['H2'][label]-acc['H1'][label]})
    with (OUT/'per_class.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(class_rows[0]));w.writeheader();w.writerows(class_rows)
    analysis['class_changes']={'vs_H0':{'improved':int(np.sum(acc['H2']>acc['H0'])),
                                      'equal':int(np.sum(acc['H2']==acc['H0'])),
                                      'worse':int(np.sum(acc['H2']<acc['H0']))},
                               'largest_improvements':sorted(class_rows,key=lambda r:r['H2_minus_H0'],reverse=True)[:8],
                               'largest_regressions':sorted(class_rows,key=lambda r:r['H2_minus_H0'])[:8]}
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,m in zip(axes,['H0','H2']):
        mat=matrices[m]/matrices[m].sum(1,keepdims=True)
        im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
        ax.set_title(m+' | row-normalized confusion');ax.set_xlabel('Predicted class ID');ax.set_ylabel('True class ID')
    fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.025,pad=.03)
    fig.savefig(OUT/'confusion_matrices.png',bbox_inches='tight');plt.close(fig)

    # Select representative outcome categories reproducibly, with class diversity.
    seedrows={display_method(r):r for r in primary if r['shot']==5 and r['seed']==0}
    pred={m:{r['sample_id']:r for r in predictions(seedrows[m],ROOT/'results/part_a/test')}
          for m in ['H0','H1','H2']}
    categories=[('H0 wrong -> H2 correct','H0',False,True,2),
                ('H0 correct -> H2 wrong','H0',True,False,2),
                ('H1 wrong -> H2 correct','H1',False,True,2),
                ('H1 correct -> H2 wrong','H1',True,False,2),
                ('Both H0 and H2 wrong','H0',False,False,2)]
    cases=[];used=set()
    for title,base,bc,hc,limit in categories:
        used_classes=set()
        eligible=[sid for sid,r in pred[base].items()
                  if bool(int(r['correct']))==bc and bool(int(pred['H2'][sid]['correct']))==hc and sid not in used]
        for sid in sorted(eligible):
            record=byid[sid]
            if record['label'] in used_classes: continue
            cases.append({'sample_id':sid,'category':title,'true':record['classname'],
                          'H0':labels[int(pred['H0'][sid]['pred_label'])],
                          'H1':labels[int(pred['H1'][sid]['pred_label'])],
                          'H2':labels[int(pred['H2'][sid]['pred_label'])]})
            used.add(sid);used_classes.add(record['label'])
            if len(used_classes)==limit: break
    analysis['cases']=cases
    fig,axes=plt.subplots(math.ceil(len(cases)/2),2,figsize=(12,3.6*math.ceil(len(cases)/2)))
    for ax,case in zip(np.asarray(axes).ravel(),cases):
        with Image.open(byid[case['sample_id']]['path']) as im: ax.imshow(im)
        ax.set_title(case['category']+' | '+case['sample_id'].removeprefix('jpg/')+
                     '\nTrue: '+case['true']+'\nH0: '+case['H0']+' | H1: '+case['H1']+' | H2: '+case['H2'],fontsize=9)
        ax.axis('off')
    for ax in np.asarray(axes).ravel()[len(cases):]: ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'error_cases.png',bbox_inches='tight',dpi=120);plt.close(fig)

    # Paired corrected/broken counts at image level, summed across repeated support runs.
    analysis['prediction_transitions']={}
    for group_name,rows,folder in [('primary',primary,'test'),('replication',replication,'replication')]:
        counts={m:{'corrected':0,'broken':0,'both_wrong':0} for m in ['H0','H1','H1 tuned lambda']}
        for seed in sorted(set(r['seed'] for r in rows)):
            sr={display_method(r):r for r in rows if r['seed']==seed and r['shot']==5}
            hp={r['sample_id']:r for r in predictions(sr['H2'],ROOT/'results/part_a'/folder)}
            for m in counts:
                for r in predictions(sr[m],ROOT/'results/part_a'/folder):
                    b=bool(int(r['correct']));h=bool(int(hp[r['sample_id']]['correct']))
                    if not b and h: counts[m]['corrected']+=1
                    if b and not h: counts[m]['broken']+=1
                    if not b and not h: counts[m]['both_wrong']+=1
        analysis['prediction_transitions'][group_name]=counts
    (OUT/'analysis.json').write_text(json.dumps(analysis,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in analysis.items() if k in ['primary_summary','replication_summary','primary_paired','replication_paired','prediction_transitions']},indent=2))


if __name__=='__main__':
    main()
