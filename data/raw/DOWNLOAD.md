# Obtaining the raw inputs

The repository does not redistribute third party data. Run the commands below
from the repository root to reproduce `data/raw` exactly as it was used.

```bash
mkdir -p data/raw/corpora

# UNSW-NB15, both public partitions.
curl -L -o data/raw/UNSW_hf_training-set.csv \
  https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv
curl -L -o data/raw/unsw_w_train.parquet \
  https://huggingface.co/datasets/wwydmanski/UNSW-NB15/resolve/main/train.parquet

# Real source code used as the leaked repository content.
curl -L -o data/raw/corpora/flask.tar.gz \
  https://github.com/pallets/flask/archive/refs/tags/3.0.3.tar.gz
curl -L -o data/raw/corpora/requests.tar.gz \
  https://github.com/psf/requests/archive/refs/tags/v2.32.3.tar.gz

# Real instruction prompts used as benign assistant traffic.
curl -L -o data/raw/corpora/alpaca.json \
  https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
curl -L -o data/raw/corpora/awesome_prompts.csv \
  https://huggingface.co/datasets/fka/awesome-chatgpt-prompts/resolve/main/prompts.csv

# Real structured business records used as the leaked document content.
curl -L -o data/raw/corpora/biz_26k-consumer-complaints.csv \
  https://raw.githubusercontent.com/plotly/datasets/master/26k-consumer-complaints.csv
curl -L -o data/raw/corpora/biz_2014_apple_stock.csv \
  https://raw.githubusercontent.com/plotly/datasets/master/2014_apple_stock.csv
```
