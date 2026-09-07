"""Read-only actual-checkpoint probe; cached head features for virtual updates."""
import argparse,io,json,os,time
from pathlib import Path
import sys
import numpy as np

def main(args):
 import torch,pyarrow.parquet as pq,cv2
 from PIL import Image
 from torchvision import transforms
 from torchvision.transforms import InterpolationMode
 torch.set_num_threads(4);sys.path.insert(0,args.xvla_root)
 from models.modeling_xvla import XVLA
 from models.processing_xvla import XVLAProcessor
 args.output.mkdir(parents=True,exist_ok=True);started=time.time();spec=json.loads(args.spec.read_text());samples=list(spec['samples'])
 rng=np.random.default_rng(spec['sampling_seed']);idroot=Path(spec['id_root']);tables={}
 def read(path):
  if path not in tables:tables[path]=pq.read_table(path,columns=['state','actions','image','wrist_image']).to_pydict()
  return tables[path]
 # ID replay is a retention diagnostic, not a new independently held-out ID test.
 id_positions=[]
 for ep in range(108,128):
  path=str(idroot/f'data/chunk-000/episode_{ep:06d}.parquet');t=read(path)
  id_positions.extend((ep,path,i,len(t['state'])) for i in range(len(t['state'])))
 for k in rng.choice(len(id_positions),size=spec['id_samples_to_add'],replace=False):
  ep,path,i,n=id_positions[int(k)];samples.append(dict(group='id_replay_probe',seed=None,split='id',episode=ep,path=path,offset=i,episode_length=n,actual_takeover=None,role='id_replay_probe'))
 for ep in spec.get('id_contiguous_episodes',[]):
  path=str(idroot/f'data/chunk-000/episode_{ep:06d}.parquet');n=len(read(path)['state'])
  for i in range(n):samples.append(dict(group='id_calibration' if ep in spec['id_calibration_episodes'] else 'id_check',seed=None,split='id',episode=ep,episode_id=f'id:{ep}',path=path,offset=i,episode_length=n,actual_takeover=None,role='id_reference'))
 if args.smoke:samples=[next(s for s in samples if s['group']==g) for g in ['ood_reference','ood_probe','id_replay_probe','query_immediate','query_post_lift']]
 specs_file=args.output/'samples.json';specs_file.write_text(json.dumps(samples,indent=2))
 for p in set(s['path'] for s in samples):assert Path(p).exists(),p
 print('INPUT_PREFLIGHT',len(samples),'files',len(set(s['path'] for s in samples)),flush=True)
 model=XVLA.from_pretrained(args.checkpoint,torch_dtype=torch.bfloat16).cuda().eval();model.requires_grad_(False);processor=XVLAProcessor.from_pretrained(args.checkpoint)
 assert model.action_mode=='auto' and model.action_space.real_dim==8 and model.num_actions==10
 cfg=dict(checkpoint=args.checkpoint,action_mode=model.action_mode,real_dim=8,num_actions=10,domain=0,loss_scale=100.,noise_times=[.25,.75],dtype='bf16 trunk, fp32 cached linear head',
  probe_scope='domain0 last action decoder; no checkpoint mutation or saving',device=torch.cuda.get_device_name(),torch_version=torch.__version__)
 (args.output/'contract.json').write_text(json.dumps(cfg,indent=2));head=model.transformer.action_decoder;capture={}
 def hook(module,inputs):capture['head_input']=inputs[0].detach().float()
 handle=head.register_forward_pre_hook(hook)
 transform=transforms.Compose([transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
 language=processor.encode_language(['stack the red cube on the green cube'])['input_ids'].cuda()
 cached=[];features=[];targets=[];masks=[];native_losses=[];native_step_losses=[];free_predictions=[]
 w=head.fc.weight[0].detach().float().reshape(head.input_size,head.output_size)[:,:8].cpu().numpy();b=head.bias.weight[0,:8].detach().float().cpu().numpy()
 example_saved=False
 for index,s in enumerate(samples):
  t=read(s['path']);i=s['offset'];n=len(t['state']);assert n==s['episode_length']
  images=[]
  for name in ['image','wrist_image']:
   cell=t[name][i];im=cv2.imdecode(np.frombuffer(cell['bytes'],dtype=np.uint8),cv2.IMREAD_COLOR);assert im is not None
   # Match historical cv2 BGR bytes interpreted by PIL in the X-VLA handler.
   if not example_saved and name=='image':Image.fromarray(im[:,:,::-1]).save(args.output/'sample_source_rgb.png')
   images.append(transform(Image.fromarray(im)))
  example_saved=True
  image=torch.stack(images).unsqueeze(0).cuda().bfloat16();proprio=torch.zeros((1,20),device='cuda',dtype=torch.bfloat16);proprio[0,:9]=torch.tensor(t['state'][i],device='cuda',dtype=torch.bfloat16)
  length=min(10,n-i);action=np.array(t['actions'][i:i+length],dtype=np.float32);action=np.concatenate([action,np.repeat(action[-1:],10-length,axis=0)],axis=0)
  target=torch.zeros((1,10,20),device='cuda',dtype=torch.bfloat16);target[0,:,:8]=torch.from_numpy(action).cuda().bfloat16();mask=np.arange(10)<length
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   enc=model.forward_vlm(language,image,torch.ones((1,2),device='cuda',dtype=torch.bool))
   bridge=torch.cat([enc['vlm_features'],enc['aux_visual_inputs']],dim=1).float().mean(1)[0]
   features.append(bridge.cpu().numpy());per_noise=[];per_loss=[];step_losses=[]
   for mc,noise_t in enumerate([.25,.75]):
    generator=torch.Generator(device='cuda');generator.manual_seed(260906+index*17+mc)
    noise=torch.randn(target.shape,device='cuda',dtype=torch.bfloat16,generator=generator);noisy=noise*noise_t+target*(1-noise_t)
    pp,aa=model.action_space.preprocess(proprio,noisy)
    pred=model.transformer(domain_id=torch.zeros(1,device='cuda',dtype=torch.long),action_with_noise=aa,proprio=pp,t=torch.tensor([noise_t],device='cuda',dtype=torch.bfloat16),**enc)
    step_error=((pred[0,:,:8].float()-target[0,:,:8].float())**2).mean(-1)*100
    per_loss.append(float(step_error[:length].mean().cpu()));step_losses.append(step_error.cpu().numpy())
    h=capture['head_input'][0].cpu().numpy();per_noise.append(h)
   cached.append(np.stack(per_noise));targets.append(action);masks.append(mask);native_losses.append(per_loss);native_step_losses.append(step_losses)
   if args.free_action_probe:
    predictions=[]
    for mc in range(2):
     generator=torch.Generator(device='cuda');generator.manual_seed(270907+index*17+mc)
     prior=torch.randn(target.shape,device='cuda',dtype=torch.bfloat16,generator=generator);generated=torch.zeros_like(prior)
     for step in range(10,0,-1):
      noise_t=step/10;pp,aa=model.action_space.preprocess(proprio,prior*noise_t+generated*(1-noise_t))
      generated=model.transformer(domain_id=torch.zeros(1,device='cuda',dtype=torch.long),action_with_noise=aa,proprio=pp,t=torch.tensor([noise_t],device='cuda',dtype=torch.bfloat16),**enc)
     predictions.append(model.action_space.postprocess(generated)[0].float().cpu().numpy())
    free_predictions.append(predictions)
  if index%32==0:
   progress=dict(pid=os.getpid(),completed=index+1,total=len(samples),elapsed_seconds=time.time()-started,gpu_memory_gb=torch.cuda.max_memory_allocated()/1e9)
   (args.output/'progress.json').write_text(json.dumps(progress));print('PROGRESS',progress,flush=True)
 handle.remove()
 arrays=dict(head_features=np.asarray(cached),bridge_features=np.asarray(features),targets=np.asarray(targets),valid_mask=np.asarray(masks),native_loss_mc=np.asarray(native_losses),native_loss_per_action_mc=np.asarray(native_step_losses),head_weight=w,head_bias=b)
 if args.free_action_probe:
  arrays['free_predictions']=np.asarray(free_predictions);cfg['free_action_probe']=dict(samples=2,denoising_steps=10,expert_action_conditioning=False,seed_rule='270907+index*17+mc');(args.output/'contract.json').write_text(json.dumps(cfg,indent=2))
 assert all(np.isfinite(v).all() for v in arrays.values());buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);(args.output/'head_probe_cache.npz').write_bytes(buffer.getvalue())
 final=dict(status='FEATURE_EXTRACTION_COMPLETE',samples=len(samples),head_features_shape=list(arrays['head_features'].shape),elapsed_seconds=time.time()-started,
  tail_samples=int(sum(not v.all() for v in masks)),finite=True,actual_model='X-VLA ckpt7500',scope='inference-only original checkpoint; future virtual update restricted to action decoder')
 (args.output/'FEATURE_EXTRACTION_COMPLETE.json').write_text(json.dumps(final,indent=2));print(final,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--xvla-root',default='/root/X-VLA-stackpyramid-clean');p.add_argument('--checkpoint',default='/root/xvla_stage2_inputs_priority/ckpt-7500');p.add_argument('--smoke',action='store_true');p.add_argument('--free-action-probe',action='store_true');main(p.parse_args())
