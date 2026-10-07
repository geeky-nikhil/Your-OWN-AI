"""Reproducible synthetic evaluation; each index runs in a fresh process."""
import argparse, csv, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--sizes',nargs='+',type=int,default=[1000,10000,50000]);p.add_argument('--dims',type=int,default=16);p.add_argument('--queries',type=int,default=100);p.add_argument('--metrics',nargs='+',default=['cosine']);p.add_argument('--m',nargs='+',type=int,default=[8,16,32]);p.add_argument('--ef-construction',nargs='+',type=int,default=[100,200]);p.add_argument('--output',default='benchmarks/results.csv');args=p.parse_args()
subprocess.run(['make','build/benchmark'],check=True)
with open(args.output,'w',newline='') as out:
    writer=None
    for n in args.sizes:
        for metric in args.metrics:
            for m in args.m:
                for ef in args.ef_construction:
                    if ef<m: continue
                    print(f'Benchmark n={n} d={args.dims} metric={metric} M={m} efConstruction={ef}',flush=True)
                    result=subprocess.run(['./build/benchmark',str(n),str(args.dims),str(args.queries),str(m),str(ef),metric],capture_output=True,text=True,check=True)
                    reader=csv.DictReader(result.stdout.splitlines())
                    if writer is None:writer=csv.DictWriter(out,fieldnames=reader.fieldnames);writer.writeheader()
                    writer.writerows(reader);out.flush()
