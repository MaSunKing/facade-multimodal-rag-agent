"""Re-run fixed questions without altering published historical results.

Preparation audits SHA/canonical binding; the historical freeze manifest is
not rewritten to match another parser. New runs stay in ignored runtime/.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tokenizer-path',type=Path,required=True)
    args=parser.parse_args()
    if not args.tokenizer_path.exists(): raise ValueError('Local tokenizer required; do not label estimates as exact packing')
    subprocess.run([sys.executable,str(ROOT/'scripts/prepare_public_eval.py')],check=True,cwd=ROOT)
    from run_public_eval import offline
    folder=ROOT/'evaluation/public_v1';dataset=json.loads((folder/'public_eval.json').read_text(encoding='utf-8'))
    output=ROOT/'runtime/public_eval/regression_30';output.mkdir(parents=True,exist_ok=True)
    offline(folder,dataset,output,args.tokenizer_path)
    import run_fresh_english_native_eval as fresh
    fresh.OUTPUT=ROOT/'runtime/public_eval/fresh_english_30'
    import run_basic_capabilities_extra_eval as extra
    extra.OUTPUT=ROOT/'runtime/public_eval/basic_extra_15'
    # Both existing runners resolve the standard local models path. Keep
    # execution explicit rather than silently downloading/model-loading.
    import os
    os.environ['PUBLIC_EVAL_TOKENIZER_PATH']=str(args.tokenizer_path.resolve())
    fresh.main()
    original=sys.argv;sys.argv=[original[0],'--prompt-budget','5000']
    try:extra.main()
    finally:sys.argv=original
    print('Finished fixed 75 CPU cases. New results are in runtime/public_eval; published historical results unchanged.')


if __name__=='__main__':main()
