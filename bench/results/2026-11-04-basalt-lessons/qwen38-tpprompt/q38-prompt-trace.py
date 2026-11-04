import argparse, csv, subprocess, collections
from pathlib import Path
parser = argparse.ArgumentParser(description='Analyze the original Qwen prompt rocprof slice; raw CSVs are required.')
parser.add_argument('trace_dir', type=Path, help='Directory containing qwen_kernel_trace.csv and qwen_memory_copy_trace.csv')
base = parser.parse_args().trace_dir
for name in ('qwen_kernel_trace.csv', 'qwen_memory_copy_trace.csv'):
    if not (base / name).is_file():
        parser.error(f'missing raw trace: {base / name}')
lo=392734115481447
hi=lo+20_000_000_000
# grep discards the unrelated startup/short-request/decode records before CSV parsing.
p=subprocess.Popen(['grep','-a','-E',r',3927(3[2-9]|[45][0-9]|6[0-3])[0-9]{9},',str(base/'qwen_kernel_trace.csv')],stdout=subprocess.PIPE,text=True)
ks=[]
markers=[]
by_stream=collections.defaultdict(lambda:[0,0])
for r in csv.reader(p.stdout):
    s,e=int(r[9]),int(r[10]); agent=int(r[1].split()[-1])-1
    if 'gr_broadcast_kernel' in r[7] or ('embed_rows_kernel' in r[7] and int(r[20])>32):
        markers.append((agent,s,r[7].split('(')[0],r[20]))
    if s>=hi or e<=lo: continue
    s,e=max(s,lo),min(e,hi)
    ks.append((s,e,agent,r[7],int(r[3])))
    by_stream[(agent,r[3])][0]+=e-s; by_stream[(agent,r[3])][1]+=1
p.wait()
cs=[]
copy_streams=collections.defaultdict(lambda:[0,0,0])
with (base/'qwen_memory_copy_trace.csv').open() as f:
    for r in csv.DictReader(f):
        s,e=int(r['Start_Timestamp']),int(r['End_Timestamp'])
        if s>=hi or e<=lo: continue
        direction=r['Direction']; src=r['Source_Agent_Id']; dst=r['Destination_Agent_Id']
        a=int((dst if 'HOST_TO_DEVICE' in direction else src).split()[-1])-1
        copy_streams[(a,direction,r['Stream_Id'])][0]+=min(e,hi)-max(s,lo)
        copy_streams[(a,direction,r['Stream_Id'])][1]+=1
        copy_streams[(a,direction,r['Stream_Id'])][2]=max(copy_streams[(a,direction,r['Stream_Id'])][2],e-s)
        cs.append((max(s,lo),min(e,hi),a,direction))
def segments(intervals):
    events=[]
    for s,e,k in intervals: events.append((s,k,1)); events.append((e,k,-1))
    events.sort()
    acc=[0]*4; last=lo; out=collections.Counter(); spans=[]
    for t,k,d in events:
        if t>last:
            state=tuple(v>0 for v in acc)
            out[state]+=t-last
            spans.append((last,t,state))
        acc[k]+=d; last=t
    if last<hi: out[tuple(v>0 for v in acc)]+=hi-last
    return out,spans
out,spans=segments([(s,e,a) for s,e,a,_,_ in ks]+[(s,e,2+a) for s,e,a,_ in cs if a in (0,1)])
print('WINDOW',lo,hi,(hi-lo)/1e9)
print('State: compute0 compute1 copy0 copy1; seconds')
for state,ns in sorted(out.items()): print(state,round(ns/1e9,6))
for a in range(2):
    idle=sum(ns for state,ns in out.items() if not state[a])
    idle_own_copy=sum(ns for state,ns in out.items() if not state[a] and state[2+a])
    idle_no_own=sum(ns for state,ns in out.items() if not state[a] and not state[2+a])
    print('GPU',a,'idle_s',idle/1e9,'idle_own_copy_s',idle_own_copy/1e9,'idle_no_own_copy_s',idle_no_own/1e9)
    # Merge kernels so gaps are actual SM-idle periods, including short event/launch gaps.
    merged=[]
    for s,e,*_ in sorted(k for k in ks if k[2]==a):
        if merged and s<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],e)
        else: merged.append([s,e])
    gaps=[(merged[i-1][1],merged[i][0]) for i in range(1,len(merged)) if merged[i][0]>merged[i-1][1]]
    for ms in [0.05,0.2,1,5,20,100]:
        subset=[(s,e) for s,e in gaps if e-s>=ms*1e6]
        print('gap>=',ms,'ms count',len(subset),'sum_s',round(sum(e-s for s,e in subset)/1e9,6))
    print('longest gaps',[(round((s-lo)/1e9,6),round((e-s)/1e6,4)) for s,e in sorted(gaps,key=lambda x:x[1]-x[0],reverse=True)[:10]])
print('Kernel streams (GPU,stream): kernel sum seconds,count')
for k,v in sorted(by_stream.items()): print(k,round(v[0]/1e9,6),v[1])
print('Copy streams (GPU,direction,stream): sum seconds,count,max ms')
for k,v in sorted(copy_streams.items()): print(k,round(v[0]/1e9,6),v[1],round(v[2]/1e6,6))
print('Large embedding/chunk markers (seconds relative window start)')
for a,s,n,rows in markers: print(a,round((s-lo)/1e9,6),rows,n)
