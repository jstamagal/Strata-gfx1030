#!/bin/bash
# window 10 (after window 9), IQ3_S, agent + edit prompts: HyperQwen-shaped levers already in Strata
#  en  = --mtp-draft-vocab draft_vocab_en.bin (40.5K rows vs the shared rt's 106K cjk subset; same 2.1% miss on code/JSON)
#  lk12 = STRATA_PL_LOOKUP_MAX_T=12 (opt-in long lookup windows in the pipelined loop; HyperQwen's DFLASH_TOKENS=15 analog)
H=/home/ronald_rump/inbox/agents/claude/strata-ab
until grep -q 'WINDOW9 DONE' $H/window9.out 2>/dev/null; do sleep 15; done
echo "window 10 start $(date +%H:%M)"
cd $H
EN="ARG=--mtp-draft-vocab /home/ronald_rump/Strata_donttouch/data/draft_vocab_en.bin"
{
for P in agent edit; do
  [ $P = edit ] && PF=$H/edit-prompts.json || PF=/home/ronald_rump/inbox/agents/claude/strata-q38-dflash2/bench/results/2026-11-04-dflash2-agent-prompts.json
  env CFG=$H/qwen-iq3s.json MODEL=qwen3.8-abl-iq3_s PROMPTS=$PF ./agab2.sh w10_${P}_base
  env CFG=$H/qwen-iq3s.json MODEL=qwen3.8-abl-iq3_s PROMPTS=$PF ./agab2.sh w10_${P}_en "$EN"
  env CFG=$H/qwen-iq3s.json MODEL=qwen3.8-abl-iq3_s PROMPTS=$PF ./agab2.sh w10_${P}_lk12 STRATA_PL_LOOKUP_MAX_T=12
  env CFG=$H/qwen-iq3s.json MODEL=qwen3.8-abl-iq3_s PROMPTS=$PF ./agab2.sh w10_${P}_en_lk12 "$EN" STRATA_PL_LOOKUP_MAX_T=12
done
} > batch30.out 2>&1
echo WINDOW10 DONE
