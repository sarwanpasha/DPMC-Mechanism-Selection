#!/usr/bin/env bash
# Reproduce every experiment of the paper, then every table and the figure.
#
#   bash scripts/run_all.sh                 all experiments, then the analysis
#   JOBS=28 THREADS=5 bash scripts/run_all.sh
#   SCALING=1 bash scripts/run_all.sh       also the 4000x4000 and 8000x8000 MovieLens blocks
#
# JOBS processes run in parallel, each with THREADS BLAS threads (one thread per process is
# fastest for the small per-row solves of ALS: JOBS=$(nproc) THREADS=1). Results are written under
# results/ and every driver writes JSON atomically, so the script can be interrupted and
# rerun: finished result files are skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
JOBS="${JOBS:-$(( $(nproc) / 5 > 0 ? $(nproc) / 5 : 1 ))}"
THREADS="${THREADS:-5}"
SCALING="${SCALING:-0}"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export ML="$ROOT/data/ml" AM="$ROOT/data/amazon" GR="$ROOT/data/goodreads"
export LAMS=1,3,10,30,100,300,1000,3000,10000,30000,100000
mkdir -p results/grid results/nodebias results/pythia results/loo results/baselines results/logs

bash scripts/fetch_data.sh
python3 scripts/build_blocks.py --ml_dir "$ML" --am_dir "$AM" --gr_dir "$GR" $([ "$SCALING" = 1 ] || echo "--ml_sizes 2000")

run_jobs () {
  grep -v '^[[:space:]]*$' "$1" | xargs -d '\n' -I{} -P "$JOBS" bash -c '{}'
}

: > results/jobs_stage1.txt
dir_of () { case $1 in ml-25m|ml-25m-rand) echo "$ML";; amazon) echo "$AM";; goodreads) echo "$GR";; esac; }
for ds in ml-25m ml-25m-rand amazon goodreads; do
  D=$(dir_of $ds); mkdir -p results/grid/$ds results/pythia/$ds results/nodebias/$ds
  for s in 0 1 2 3 4 5 6 7 8 9; do for e in 0.1 0.25 0.5 1 2 4 8; do
    for mech in shin pm krr geom laplace expo sw huber stair lpopt rrbins hm; do
      # below its switching budget the hybrid is the one-bit randomizer (channels.canonical)
      if [ $mech = hm ]; then case $e in 0.1|0.25|0.5) continue;; esac; fi
      out=results/grid/$ds/${mech}_e${e}_s${s}.json
      [ -s "$out" ] || echo "python3 experiments/run_dpmc.py --dataset $ds --data_dir $D --mech $mech --eps_list $e --seeds $s --lams $LAMS --out $out > results/logs/${ds}_${mech}_e${e}_s${s}.log 2>&1" >> results/jobs_stage1.txt
    done
    for f in 0.05 0.1 0.2; do
      out=results/pythia/$ds/pythia_e${e}_f${f}_s${s}.json
      [ -s "$out" ] || echo "python3 experiments/run_pythia.py --dataset $ds --data_dir $D --eps_list $e --fracs $f --seeds $s --lams $LAMS --out $out > results/logs/${ds}_pythia_e${e}_f${f}_s${s}.log 2>&1" >> results/jobs_stage1.txt
    done
    if [ $ds = ml-25m ]; then
      out=results/nodebias/$ds/krr_e${e}_s${s}.json
      [ -s "$out" ] || echo "python3 experiments/run_dpmc.py --dataset $ds --data_dir $D --mech krr --debias 0 --eps_list $e --seeds $s --lams $LAMS --out $out > results/logs/${ds}_krr-raw_e${e}_s${s}.log 2>&1" >> results/jobs_stage1.txt
    fi
  done; done
done
mkdir -p results/diag/goodreads results/leverage
for ds in ml-25m ml-25m-rand amazon goodreads; do
  D=$(dir_of $ds)
  [ -s results/baselines/reference_${ds}.json ] || echo "python3 experiments/run_reference.py --dataset $ds --data_dir $D --out results/baselines/reference_${ds}.json > results/logs/reference_${ds}.log 2>&1" >> results/jobs_stage1.txt
  [ -s results/leverage/${ds}.json ] || echo "python3 experiments/run_leverage.py --dataset $ds --data_dir $D --out results/leverage/${ds}.json > results/logs/leverage_${ds}.log 2>&1" >> results/jobs_stage1.txt
done
# the one-bit release on Goodreads at eps = 1 refitted at the ridge weight chosen for the other mechanisms
for s in 0 1 2 3 4 5 6 7 8 9; do
  out=results/diag/goodreads/shin_e1_s${s}.json
  [ -s "$out" ] || echo "python3 experiments/run_dpmc.py --dataset goodreads --data_dir $GR --mech shin --eps_list 1 --seeds $s --lams 300 --out $out > results/logs/diag_goodreads_shin_s${s}.log 2>&1" >> results/jobs_stage1.txt
done
for ds in ml-25m amazon; do
  tag=$([ $ds = ml-25m ] && echo ml || echo am); D=$(dir_of $ds)
  [ -s results/baselines/dpsgd_${tag}.json ] || echo "python3 experiments/run_dpsgd.py --dataset $ds --data_dir $D --noise gauss --out results/baselines/dpsgd_${tag}.json > results/logs/dpsgd_${tag}.log 2>&1" >> results/jobs_stage1.txt
  [ -s results/baselines/dpgd_${tag}.json ] || echo "python3 experiments/run_dpsgd.py --dataset $ds --data_dir $D --noise laplace_pure --out results/baselines/dpgd_${tag}.json > results/logs/dpgd_${tag}.log 2>&1" >> results/jobs_stage1.txt
done
for lam in 0.0031 0.0094 0.031; do
  out=results/baselines/nuclear_ml_lam${lam}.json
  [ -s "$out" ] || echo "python3 experiments/run_nuclear.py --dataset ml-25m --data_dir $ML --mech krr --bias 0 --eps_list 0.1 --lam_fixed $lam --out $out > results/logs/nuclear_ml_${lam}.log 2>&1" >> results/jobs_stage1.txt
done
for lam in 0.0006 0.0019 0.0057; do
  out=results/baselines/nuclear_am_lam${lam}.json
  [ -s "$out" ] || echo "python3 experiments/run_nuclear.py --dataset amazon --data_dir $AM --mech krr --bias 0 --eps_list 0.1 --lam_fixed $lam --out $out > results/logs/nuclear_am_${lam}.log 2>&1" >> results/jobs_stage1.txt
done
for k in 3 4 5 7 10 20; do
  out=results/baselines/synthetic_k${k}.json
  [ -s "$out" ] || echo "python3 experiments/run_synthetic.py --k $k --out $out > results/logs/synthetic_k${k}.log 2>&1" >> results/jobs_stage1.txt
done
if [ "$SCALING" = 1 ]; then
  for s in 4000 8000; do for mech in shin krr; do
    out=results/baselines/scaling_ml${s}_${mech}.json
    [ -s "$out" ] || echo "python3 experiments/run_dpmc.py --dataset ml-25m --data_dir $ML --m $s --n $s --mech $mech --lams $LAMS --out $out > results/logs/scaling_${s}_${mech}.log 2>&1" >> results/jobs_stage1.txt
  done; done
fi
run_jobs results/jobs_stage1.txt

: > results/jobs_stage2.txt
for ds in ml-25m amazon; do
  tag=$([ $ds = ml-25m ] && echo ml || echo am); D=$([ $ds = ml-25m ] && echo "$ML" || echo "$AM")
  for mech in krr geom laplace shin pm; do for e in 0.1 0.25 0.5 1 2 4 8; do
    out=results/loo/loo_${tag}_${mech}_e${e}.json
    [ -s "$out" ] || echo "python3 experiments/run_loo_bias.py --dataset $ds --data_dir $D --mech $mech --eps $e --hparams results/grid/$ds --out $out > results/logs/loo_${tag}_${mech}_${e}.log 2>&1" >> results/jobs_stage2.txt
  done; done
done
run_jobs results/jobs_stage2.txt

OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  python3 experiments/run_timing.py --ml_dir "$ML" --am_dir "$AM" \
  --blocks "ml-25m:2000$([ "$SCALING" = 1 ] && echo ',ml-25m:4000,ml-25m:8000'),amazon:2000" \
  --out results/timing.json > results/logs/timing.log 2>&1

cd analysis
python3 channel_tables.py                                   > ../results/table_channels.txt
python3 optgap_table.py                                     > ../results/table_optgap.txt
python3 family_check.py                                     > ../results/table_family.txt
python3 main_table.py --runs ../results/grid --dpsgd ../results/baselines/dpsgd_*.json \
        --dpgd ../results/baselines/dpgd_*.json --reference ../results/baselines/reference_*.json \
                                                            > ../results/table_main.txt
for m in recall ndcg; do
  python3 main_table.py --runs ../results/grid --metric $m  > ../results/table_main_$m.txt
  python3 agreement.py --runs ../results/grid --metric $m   > ../results/table_agreement_$m.txt
done
python3 agreement.py --runs ../results/grid                 > ../results/table_agreement.txt
python3 optimal_endtoend.py --runs ../results/grid          > ../results/table_optimal_endtoend.txt
python3 staircase_check.py --runs ../results/grid           > ../results/table_staircase.txt
python3 pythia_table.py --pythia ../results/pythia/* --runs ../results/grid > ../results/table_pythia.txt
python3 loo_table.py --loo ../results/loo                   > ../results/table_loo.txt
python3 timing_table.py ../results/timing.json              > ../results/table_timing.txt
python3 ablation_table.py --runs ../results/grid --nodebias ../results/nodebias \
        --nuclear ../results/baselines/nuclear_*.json      > ../results/table_ablation.txt
python3 synthetic_table.py ../results/baselines/synthetic_k*.json > ../results/table_synthetic.txt
python3 main_table.py --runs ../results/diag                  > ../results/table_goodreads_fixed_ridge.txt
python3 -c "import glob, json; [print(json.load(open(f))['meta'], json.load(open(f))['records']) for f in sorted(glob.glob('../results/leverage/*.json'))]" > ../results/table_leverage.txt
python3 make_figure.py --runs ../results/grid --out ../results/mechanism_crossover.pdf
echo "all tables written to results/"
