#!/bin/bash
# usage: pab.sh NAME KEY=VAL ...  prompt-speed A/B: the ~36K-token prompt twice per load, each with a unique first line (no prefix reuse), 32 tokens out
name=$1; shift
D=/tmp/basalt-prof/ab/$name; mkdir -p $D
python3 - "$D" "${CFG:-/tmp/basalt-prof/qwen-ud4.json}" "$@" <<'PY'
import json,sys
D=sys.argv[1]; d=json.load(open(sys.argv[2]))
d['log']=D+'/engine.log'
for kv in sys.argv[3:]:
    k,v=kv.split('=',1)
    if k=='EXE': d['exe']=v
    elif k=='ARG':
        parts=v.split(); f=parts[0]; a=d['args']
        if f in a:
            i=a.index(f); n=len(parts)-1; a[i+1:i+1+n]=parts[1:]
        else: a+=parts
    elif v=='': d['env'].pop(k,None)
    else: d['env'][k]=v
json.dump(d,open(D+'/cfg.json','w'),indent=1)
PY
cd /home/ronald_rump/Strata_donttouch
.venv/bin/python serve/server.py --engine strata --config $D/cfg.json --port 8091 > $D/server.out 2>&1 &
SP=$!
for i in $(seq 1 1500); do grep -q '^ready:' $D/server.out && break; kill -0 $SP 2>/dev/null || { echo "$name: server died"; tail -5 $D/server.out; exit 1; }; sleep 1; done
M=${MODEL:-qwen3.8-flash-next-unsloth-ud-q4_k_xl}
for r in $(seq 1 ${RUNS:-2}); do
  python3 -c "
import json; d=json.load(open('/tmp/basalt-prof/long-prompt.json')); d['messages'][0]['content']='Run $name-$r-$RANDOM. '+d['messages'][0]['content']
d.update(model='$M', max_tokens=32, temperature=0, chat_template_kwargs={'enable_thinking': False}); json.dump(d, open('$D/req$r.json','w'))"
  curl -s -m 1800 http://127.0.0.1:8091/v1/chat/completions -H 'content-type: application/json' -d @$D/req$r.json > $D/long$r.json
  python3 -c "import json; t=json.load(open('$D/long$r.json'))['timings']; print('$name run$r prompt', t['prompt_n'], round(t['prompt_per_second'],1), 'tok/s', round(t['prompt_ms']/1000,2), 's')"
done
kill $SP; wait $SP 2>/dev/null
