#!/usr/bin/env bash
# Download the three public datasets.
#   data/ml/ml-25m/ratings.csv   MovieLens-25M (GroupLens), about 250 MB zipped
#   data/amazon/Video_Games.csv  Amazon Reviews 2023, Video Games 5-core, rating only
#   data/goodreads/goodreads_interactions_comics_graphic.json.gz
#                                UCSD Book Graph, comics and graphic novels, about 390 MB
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data/ml" "$ROOT/data/amazon" "$ROOT/data/goodreads"
if [ ! -f "$ROOT/data/ml/ml-25m/ratings.csv" ]; then
  curl -L -o "$ROOT/data/ml/ml-25m.zip" https://files.grouplens.org/datasets/movielens/ml-25m.zip
  (cd "$ROOT/data/ml" && unzip -q ml-25m.zip)
fi
if [ ! -f "$ROOT/data/amazon/Video_Games.csv" ]; then
  curl -L -o "$ROOT/data/amazon/Video_Games.csv.gz" \
    https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz
  gunzip -f "$ROOT/data/amazon/Video_Games.csv.gz"
fi
GR="$ROOT/data/goodreads/goodreads_interactions_comics_graphic.json.gz"
if [ ! -f "$GR" ]; then
  curl -L -o "$GR" \
    https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_interactions_comics_graphic.json.gz
fi
echo "datasets ready under $ROOT/data"
