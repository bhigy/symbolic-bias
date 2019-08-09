import argparse
import torch

import vg.defn.asr as D

# Parse command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('path', metavar='path', help='Model\'s path', nargs='+')
args = parser.parse_args()

for path in args.path:
    net = torch.load(path)
    print(path)
    print(net)
