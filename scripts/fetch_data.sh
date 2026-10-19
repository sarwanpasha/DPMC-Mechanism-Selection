#!/usr/bin/env bash
# Download the two public datasets into data/ml and data/amazon.
#   data/ml/ml-25m/ratings.csv   MovieLens-25M (GroupLens), about 250 MB zipped
#   data/amazon/Video_Games.csv  Amazon Reviews 2023, Video Games 5-core, rating only
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data/ml" "$ROOT/data/amazon"
if [ ! -f "$ROOT/data/ml/ml-25m/ratings.csv" ]; then
  curl -L -o "$ROOT/data/ml/ml-25m.zip" https://files.grouplens.org/datasets/movielens/ml-25m.zip
  (cd "$ROOT/data/ml" && unzip -q ml-25m.zip)
fi
if [ ! -f "$ROOT/data/amazon/Video_Games.csv" ]; then
  curl -L -o "$ROOT/data/amazon/Video_Games.csv.gz" \
    https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz
  gunzip -f "$ROOT/data/amazon/Video_Games.csv.gz"
fi
echo "datasets ready under $ROOT/data"
