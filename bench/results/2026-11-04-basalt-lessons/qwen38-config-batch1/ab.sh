#!/bin/bash
# usage: ab.sh NAME KEY=VAL ...   (env overrides on top of the Ornith Q3 profiling config)
name=$1; shift
D=/tmp/basalt-prof/ab/$name; mkdir -p $D
python3 - "$D" "${CFG:-/tmp/basalt-prof/ornith-q3-prof.json}" "$@" <<'PY'
import json,sys
D=sys.argv[1]; d=json.load(open(sys.argv[2]))
d['log']=D+'/engine.log'
for kv in sys.argv[3:]:
    k,v=kv.split('=',1)
    if k=='EXE': d['exe']=v
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
cd /home/ronald_rump/Strata_donttouch
.venv/bin/python serve/server.py --engine strata --config $D/cfg.json --port 8091 > $D/server.out 2>&1 &
SP=$!
for i in $(seq 1 1500); do grep -q '^ready:' $D/server.out && break; kill -0 $SP 2>/dev/null || { echo "$name: server died"; tail -5 $D/server.out; exit 1; }; sleep 1; done
req() { curl -s -m 900 http://127.0.0.1:8091/v1/chat/completions -H 'content-type: application/json' -d "{\"model\":\"${MODEL:-ornith-1.5-122b-q3}\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":$2,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}"; }
req "Say hi." 16 > $D/warm.json
for r in $(seq 1 ${RUNS:-2}); do req "Write a detailed 900-word essay on the history of the steam engine, from Hero of Alexandria to Watt and beyond." 1200 > $D/essay$r.json; done
kill $SP; wait $SP 2>/dev/null
for r in $(seq 1 ${RUNS:-2}); do python3 -c "import json,sys; d=json.load(open('$D/essay$r.json')); t=d['timings']; print('$name run$r', t['predicted_n'], 'tok', t['predicted_per_second'], 'tok/s', 'drafts', t['draft_n_accepted'], '/', t['draft_n'])"; python3 -c "import json; print(json.load(open('$D/essay$r.json'))['choices'][0]['message']['content'])" > $D/essay$r.txt; done
grep PROF $D/engine.log | tail -1
md5sum $D/essay*.txt | cut -c1-12
