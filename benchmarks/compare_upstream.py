"""Compare a supplied pre-update main.cpp with this fork on identical seeded data."""
import argparse, csv, io, json, subprocess, tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('benchmarks/results-upstream-comparison.csv'));args=p.parse_args()
harness=Path('benchmarks/metric_consistency.cpp').read_text()
# Rename the baseline server's main function while retaining all search classes.
baseline_source=harness.replace('#include "../main.cpp"','#define main upstream_main\n#include '+json.dumps(str(args.baseline.resolve()))+'\n#undef main')
with tempfile.TemporaryDirectory() as folder:
    cpp=Path(folder)/'baseline.cpp';exe=Path(folder)/'baseline';cpp.write_text(baseline_source)
    subprocess.run(['g++','-std=c++17','-O2','-pthread','-I',str(Path.cwd()),str(cpp),'-o',str(exe)],check=True)
    subprocess.run(['make','build/metric-check'],check=True)
    with args.output.open('w',newline='') as out:
        writer=None
        for implementation,command in [('uploaded_upstream',[str(exe)]),('updated_fork',['./build/metric-check'])]:
            result=subprocess.run(command,capture_output=True,text=True,check=True)
            reader=csv.DictReader(io.StringIO(result.stdout))
            if writer is None:writer=csv.DictWriter(out,fieldnames=['implementation']+reader.fieldnames);writer.writeheader()
            for row in reader:writer.writerow(dict(implementation=implementation,**row))
