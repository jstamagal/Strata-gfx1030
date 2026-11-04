#!/bin/bash
# usage: agab.sh NAME KEY=VAL ...  agent-workload A/B: 6 code / tool-JSON cases (DF2 agent's prompt set), each once per load, greedy, 1200 max tokens
name=$1; shift
D=/tmp/basalt-prof/ab/$name; mkdir -p $D
P=${PROMPTS:-/home/ronald_rump/inbox/agents/claude/strata-q38-dflash2/bench/results/2026-11-04-dflash2-agent-prompts.json}
python3 - "$D" "${CFG:-/tmp/basalt-prof/qwen-ud4.json}" "$@" <<'PY'
import json,sys
D=sys.argv[1]; d=json.load(open(sys.argv[2]))
d['log']=D+'/engine.log'
for kv in sys.argv[3:]:
    k,v=kv.split('=',1)
    if k=='EXE': d['exe']=v
    elif k.startswith('CFG.'): d[k[4:]]=v
    elif k=='ARG':
        parts=v.split(); f=parts[0]; a=d['args']
        if f in a:
            i=a.index(f); n=len(parts)-1
            a[i+1:i+1+n]=parts[1:]
        else: a+=parts
    elif v=='': d['env'].pop(k,None)
    else: d['env'][k]=v
json.dump(d,open(D+'/cfg.json','w'),indent=1)
PY
M=${MODEL:-qwen3.8-flash-next-unsloth-ud-q4_k_xl}
python3 -I - "$P" "$D" "$M" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); D=sys.argv[2]; M=sys.argv[3]
for c in p['cases']:
    body={'model':M,'messages':c['messages'],'max_tokens':1200,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
    json.dump(body,open(f"{D}/req_{c['id']}.json",'w'))
PY
cd /home/ronald_rump/Strata_donttouch
.venv/bin/python serve/server.py --engine strata --config $D/cfg.json --port 8091 > $D/server.out 2>&1 &
SP=$!
for i in $(seq 1 1500); do grep -q '^ready:' $D/server.out && break; kill -0 $SP 2>/dev/null || { echo "$name: server died"; tail -5 $D/server.out; exit 1; }; sleep 1; done
curl -s -m 300 http://127.0.0.1:8091/v1/chat/completions -H 'content-type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":16,\"temperature\":0}" > $D/warm.json
for r in $D/req_*.json; do id=$(basename $r .json); id=${id#req_}; curl -s -m 900 http://127.0.0.1:8091/v1/chat/completions -H 'content-type: application/json' -d @$r > $D/out_$id.json; done
kill $SP; wait $SP 2>/dev/null
python3 -I - "$D" "$name" <<'PY'
import json,sys,glob,hashlib,os
D,name=sys.argv[1],sys.argv[2]; tn=tt=0
for f in sorted(glob.glob(D+'/out_*.json')):
    d=json.load(open(f)); t=d['timings']; txt=d['choices'][0]['message']['content']
    cid=os.path.basename(f)[4:-5]; open(f"{D}/{cid}.txt",'w').write(txt)
    tn+=t['predicted_n']; tt+=t['predicted_n']/t['predicted_per_second']
    print(f"{name} {cid:30s} {t['predicted_n']:5d} tok {t['predicted_per_second']:6.1f} tok/s drafts {t.get('draft_n_accepted')}/{t.get('draft_n')} {hashlib.md5(txt.encode()).hexdigest()[:12]}")
print(f"{name} TOTAL {tn} tok {tn/tt:.1f} tok/s")
PY
