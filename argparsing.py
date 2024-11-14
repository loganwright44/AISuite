import argparse


def parse_args():
  parser = argparse.ArgumentParser(description='AI environment handler', prog='AI Suite')
  
  parser.add_argument('--version', action='version', version='%(prog)s 1.0')
  parser.add_argument('-c', '--cycles', type=int)
  parser.add_argument('--testloops', type=int)
  parser.add_argument('--trainloops', type=int)
  parser.add_argument('-e', '--epochs', type=int)
  parser.add_argument('-v', '--verbose', action='store_true', default=False)
  parser.add_argument('--compile', action='store_true', default=False)
  
  args = parser.parse_args()
  return args

