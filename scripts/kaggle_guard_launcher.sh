#!/bin/sh
exec /usr/bin/python3 /home/pokebot/.local/libexec/pokebot-kaggle-guard.py \
  --real "$0.pokebot-real" -- "$@"
