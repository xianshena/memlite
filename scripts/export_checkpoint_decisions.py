"""Optional CUDA checkpoint audit for a NEW self-trained run; not exercised in preview QA."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from life_choice.train_gpu import cuda_setup,digest,load_data,evaluate,exclusive_gpu
from life_choice.gpu_model import LargeChoiceConfig,LargeChoiceNetwork
from life_choice.environment import generate_episode,execute,VARIANT_NAMES
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('use a new output directory')
    a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.data/'manifest.json').read_text())
    cuda_setup(0)
    with exclusive_gpu(a.output):
        # Load only a checkpoint generated and trusted by the operator (Torch 2.3 pickle format).
        state=torch.load(a.checkpoint,map_location='cuda')
        assert state['data_hash']==digest(a.data/'manifest.json')
        source_root=Path(__file__).resolve().parents[1]/'life_choice'
        current_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_root.glob('*.py')}
        assert state['source_hashes']==current_sources, 'checkpoint source mismatch'
        model=LargeChoiceNetwork(LargeChoiceConfig(**state['config'])).cuda().eval()
        model.load_state_dict(state['model']);del state
        data=load_data(a.data/'test.npz',manifest,'test')
        metrics,prediction=evaluate(model,data)
        selected=prediction['probabilities'].argmax(1).cpu().tolist()
        probabilities=prediction['probabilities'].cpu().tolist()
        seeds=data['seeds'].cpu().tolist();variants=data['variants'].cpu().tolist()
        cf=0
        with (a.output/'decisions.jsonl').open('w') as stream:
            for i,(seed,v) in enumerate(zip(seeds,variants)):
                flip=bool(cf%2) if v==1 else False
                if v==1:cf+=1
                ep=generate_episode(seed,split='test',variant=VARIANT_NAMES[v],history_flip=flip)
                assert np.array_equal(ep.mask,data['mask'][i].cpu().numpy())
                assert np.allclose(ep.features,data['features'][i].cpu().numpy(),atol=2e-6,rtol=0)
                row={'episode_seed':seed,'variant':VARIANT_NAMES[v],'selected_index':selected[i],
                     'probabilities':probabilities[i],'execution':execute(ep,selected[i])}
                stream.write(json.dumps(row)+'\n')
        (a.output/'metrics.json').write_text(json.dumps(metrics,indent=2))
        print(json.dumps({'n':len(selected),'checkpoint_sha256':digest(a.checkpoint),'output':str(a.output)}))
if __name__=='__main__':main()
