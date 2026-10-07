"""Render measured CSVs; matplotlib is optional and not needed to run the app."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=list(csv.DictReader(Path('benchmarks/results-scale.csv').open()))
selected=[r for r in rows if r['k']=='10' and r['efSearch']=='50']
fig, axes=plt.subplots(1,2,figsize=(11,4.3))
n=[int(r['n']) for r in selected]
axes[0].plot(n,[float(r['exact_mean_us']) for r in selected],'o-',label='Exact brute force (full sort)')
axes[0].plot(n,[float(r['hnsw_mean_us']) for r in selected],'o-',label='HNSW (efSearch=50)')
axes[0].set(xscale='log',yscale='log',xlabel='Stored vectors',ylabel='Mean query latency (µs)',title='Synthetic 16D: scale vs latency')
axes[0].legend();axes[0].grid(alpha=.25)
for size in ['1000','10000','50000']:
    points=[r for r in rows if r['n']==size and r['k']=='10']
    axes[1].plot([float(r['hnsw_mean_us']) for r in points],[float(r['recall'])*100 for r in points],'o-',label=f'{int(size):,} vectors')
    # Label only endpoints to avoid overlap at near-perfect recall.
    for r in (points[0], points[-1]):
        axes[1].annotate('ef='+r['efSearch'],(float(r['hnsw_mean_us']),float(r['recall'])*100),xytext=(4,-14 if r is points[-1] else 5),textcoords='offset points',fontsize=8)
axes[1].set(xlabel='Mean HNSW query latency (µs)',ylabel='Recall@10 (%)',title='Search breadth: recall vs latency',ylim=(min(float(r['recall'])*100 for r in rows if r['k']=='10')-2,100.6))
axes[1].legend();axes[1].grid(alpha=.25)
fig.suptitle('Cosine distance · M=16 · efConstruction=200 · 100 held-out queries per configuration',fontsize=10)
fig.tight_layout();fig.savefig('benchmarks/scale-and-recall.png',dpi=170)
parameters=list(csv.DictReader(Path('benchmarks/results-parameters.csv').open()))
fig, ax=plt.subplots(figsize=(7.5,4.5))
for m in ['8','16','32']:
 for build in ['100','200']:
    points=[r for r in parameters if r['M']==m and r['efConstruction']==build and r['k']=='10']
    ax.plot([float(r['hnsw_mean_us']) for r in points],[float(r['recall'])*100 for r in points],'o-',label=f'M={m}, efConstruction={build}')
ax.set(xlabel='Mean HNSW query latency (µs)',ylabel='Recall@10 (%)',title='10,000 synthetic 16D vectors: parameter sweep',ylim=(min(float(r['recall'])*100 for r in parameters if r['k']=='10')-2,100.6));ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig('benchmarks/parameters.png',dpi=170)
