import argparse
from modelscope.hub.snapshot_download import snapshot_download


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="model_name_or_path")
parser.add_argument("--cache_path", help="cache_path", default='/home/xuanming/pre-trained-models')
args = parser.parse_args()

print(f'download {args.model_name} to {args.cache_path}')
snapshot_download(args.model_name, cache_dir=args.cache_path)
print('done')