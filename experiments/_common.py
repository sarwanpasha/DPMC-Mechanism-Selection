"""Shared command-line plumbing for the experiment drivers."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LAMS = "1,3,10,30,100,300,1000,3000,10000,30000,100000"
EPS = "0.1,0.25,0.5,1,2,4,8"


def add_data_args(ap):
    """Arguments that locate and shape the evaluation block."""
    ap.add_argument("--dataset", choices=["ml-25m", "amazon"], default="ml-25m")
    ap.add_argument("--data_dir", required=True,
                    help="directory holding ml-25m/ratings.csv or Video_Games.csv")
    ap.add_argument("--cache_dir", default=None,
                    help="directory for cached blocks (default: data_dir)")
    ap.add_argument("--m", type=int, default=2000, help="number of most active users kept")
    ap.add_argument("--n", type=int, default=2000, help="number of most active items kept")
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.10)


def add_fit_args(ap):
    """Arguments of the reconstruction and its cross-validated hyperparameters."""
    ap.add_argument("--ranks", type=str, default="8,16")
    ap.add_argument("--lams", type=str, default=LAMS)
    ap.add_argument("--n_iter", type=int, default=15)
    ap.add_argument("--folds", type=int, default=5)


def log(s):
    """Print and flush, so progress is visible in batch logs."""
    print(s, flush=True)
