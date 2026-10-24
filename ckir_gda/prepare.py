"""Build one recorded episode from explicit cached features and text assets."""
from pathlib import Path
import json
import numpy as np
import torch
from .io import digest


def prepare_episode(index_path,feature_cache,template,cupl,*,phase,seed,label_version,output):
    output=Path(output)
    if output.exists():raise FileExistsError('choose a fresh prepared-input directory')
    index=json.loads(Path(index_path).read_text('utf-8'))
    paths={'feature_cache':Path(feature_cache),'template':Path(template),'cupl':Path(cupl)}
    for name,path in paths.items():
        if digest(path)!=index['assets'][name]['sha256']:
            raise ValueError(f'{name} differs from the recorded experiment asset')
    episode=next((e for e in index['episodes'] if e['phase']==phase and e['support_seed']==seed),None)
    if episode is None or label_version not in range(4):
        raise ValueError('unknown recorded episode or label version')
    with np.load(paths['feature_cache'],allow_pickle=False) as z:
        features=z['image_features'].astype(np.float64)
        ids=[str(v) for v in z['ids']]
        gda_text=z['text_features'].T.astype(np.float32)
    positions={v:i for i,v in enumerate(ids)}
    support_ids=episode['support_ids'];query_ids=index['query_ids']
    raw={name:torch.load(paths[name],map_location='cpu',weights_only=True).detach().cpu().numpy().copy(order='K') for name in ('template','cupl')}
    arrays={'support':features[[positions[v] for v in support_ids]],
            'query':features[[positions[v] for v in query_ids]],
            'observed_labels':np.asarray(episode['observed_label_versions'][label_version],dtype=np.int64),
            'template':raw['template'],'cupl':raw['cupl'],'gda_text':gda_text}
    output.mkdir(parents=True)
    np.savez_compressed(output/'episode.npz',**arrays)
    manifest={'format':'ckir-gda-episode-v1','episode_id':f'{phase}-{seed}-v{label_version}',
        'kind':'recorded-pets-episode','phase':phase,'support_seed':seed,'label_version':label_version,
        'class_names':index['class_names'],'support_ids':support_ids,'query_ids':query_ids,
        'arrays':{'path':'episode.npz','sha256':digest(output/'episode.npz')},
        'input_asset_sha256':{k:index['assets'][k]['sha256'] for k in paths}}
    (output/'episode.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8')
    return {'status':'PREPARED_RECORDED_EPISODE','manifest':str(output/'episode.json'),
            'query_labels_included':False,'training_performed':False}
