.PHONY: setup test zeroshot coop tipadapter

setup:
\tpython -m pip install -r requirements.txt

test:
\tpytest -q

zeroshot:
\tbash scripts/run_zeroshot.sh

coop:
\tbash scripts/run_coop.sh

tipadapter:
\tbash scripts/run_tipadapter.sh