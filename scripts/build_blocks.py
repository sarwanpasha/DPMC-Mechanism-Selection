"""Build and cache the evaluation blocks used in the paper.

MovieLens-25M: the 2000 x 2000 top-activity block (main results), the 4000 x 4000 and
8000 x 8000 blocks (scaling) and the 2000 x 2000 random-user block (long tail).
Amazon Video Games and Goodreads comics: the 2000 x 2000 top-activity blocks.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpmc.data import load_cached


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ml_dir", default="data/ml")
    ap.add_argument("--am_dir", default="data/amazon")
    ap.add_argument("--gr_dir", default="data/goodreads")
    ap.add_argument("--ml_sizes", default="2000,4000,8000")
    args = ap.parse_args()
    for s in (int(x) for x in args.ml_sizes.split(",")):
        rows, cols, vals, shape, lo, hi = load_cached("ml-25m", args.ml_dir, s, s, 0, args.ml_dir)
        print("ml-25m %dx%d -> shape %s, %d ratings" % (s, s, shape, vals.size))
    for ds, d in (("ml-25m-rand", args.ml_dir), ("amazon", args.am_dir),
                  ("goodreads", args.gr_dir)):
        rows, cols, vals, shape, lo, hi = load_cached(ds, d, 2000, 2000, 0, d)
        print("%s 2000x2000 -> shape %s, %d ratings" % (ds, shape, vals.size))


if __name__ == "__main__":
    main()
