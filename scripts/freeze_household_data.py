"""Regenerate synthetic data into a NEW directory; no training."""
import argparse,json
from life_choice.environment import freeze_dataset
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--train',type=int,default=10000)
    p.add_argument('--validation',type=int,default=1000)
    p.add_argument('--test',type=int,default=2000)
    a=p.parse_args()
    m=freeze_dataset(a.output,a.train,a.validation,a.test)
    print(json.dumps({'split_counts':m['split_counts'],'generator_source_sha256':m['source_sha256']}))
if __name__=='__main__':main()
