"""Consolidate candidate tests, paired YCB uncertainty and source-video inspection."""
import itertools,json,subprocess
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr,pearsonr
from PIL import Image,ImageDraw,ImageFont

ROOT=Path('artifacts/ood_supervision_exploration_20260906')

def main():
 a=json.loads((ROOT/'exploration.json').read_text());full=json.loads((ROOT/'full_path_summary.json').read_text());all_trials=list(a['trials'])
 for kernel,f in itertools.product(['hard','soft'],['1','3','6']):
  row=dict(family='full_path',kernel=kernel,factor=int(f),cohorts={},values={})
  for task,result in full.items():
   row['cohorts'][task]={**result['correlations'][kernel][f],'ranking_score':result['correlations'][kernel][f]['spearman']}
   row['values'][task]={m:s[kernel][f] for m,s in result['summaries'].items()}
  row['mean_rank_score']=float(np.mean([v['ranking_score'] for v in row['cohorts'].values()]));all_trials.append(row)
 loto={}
 for held in a['groups']:
  best=max(all_trials,key=lambda t:np.mean([t['cohorts'][g]['ranking_score'] for g in a['groups'] if g!=held]))
  loto[held]=dict(parameters={k:v for k,v in best.items() if k not in ['cohorts','values']},held_out_spearman=best['cohorts'][held]['spearman'])
 # Paired episode bootstrap; each method was evaluated on the same 100 seeds.
 ycb=json.loads(Path('artifacts/pi05_table_tasr_20260903/inputs/utility_evidence_5090.json').read_text())['ycb_object']['files']
 arrays={}
 for item in ycb:
  s=item['summary']
  if s['split']=='ood':arrays[Path(item['relative_path']).parts[0]]={r['seed']:int(r['success']) for r in s['rows']}
 seeds=sorted(arrays['bridge_pca']);assert all(sorted(v)==seeds for v in arrays.values());rng=np.random.default_rng(20260906);indices=rng.integers(0,len(seeds),size=(20000,len(seeds)));paired=[]
 for m in ['offline_oracle','failure_recovery','diffdagger']:
  delta=np.array([arrays['bridge_pca'][s]-arrays[m][s] for s in seeds]);ci=np.quantile(delta[indices].mean(1),[.025,.975])
  paired.append(dict(comparison='bridge_pca minus '+m,delta=float(delta.mean()),bootstrap95=ci.tolist(),pca_only_success=int(np.sum(delta==1)),other_only_success=int(np.sum(delta==-1))))
 # Small inspectable contact sheet from the actual selected suffix videos.
 samples=json.loads(Path('artifacts/cross_asset_overlap_utility_20260903/ycb_unscored_video_manifest.json').read_text());score=json.loads(Path('artifacts/pi05_table_tasr_20260903/tasr_object.json').read_text())
 sheet=Image.new('RGB',(1088,4*300),'white');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',16)
 for k,s in enumerate(samples):
  row=next(r for r in score['rows'] if r['method']==s['method'] and r['seed']==s['seed']);raw=subprocess.check_output(['ffmpeg','-v','error','-i',s['local_video'],'-f','rawvideo','-pix_fmt','rgb24','-'])
  w=s['video_probe']['width'];h=s['video_probe']['height'];frames=np.frombuffer(raw,dtype=np.uint8).reshape(-1,h,w,3)
  assert len(frames)==row['expert_points'];close=row['blocks']['close'][0]
  draw.text((10,k*300+4),f"{s['method']} | seed {s['seed']} | takeover {s['expert_start']} | real training suffix",font=font,fill='black')
  for col,i in enumerate([0,close]):
   sheet.paste(Image.fromarray(frames[i]).resize((544,256)),(col*544,k*300+40));draw.text((col*544+8,k*300+22),f'expert offset {i}',font=font,fill='black')
 sheet.save(ROOT/'ycb_actual_camera_contact_sheet.jpg',quality=92)
 result=dict(total_candidates=len(all_trials),leave_one_cohort_out=loto,ycb_paired_bootstrap=paired,best_in_sample=max(all_trials,key=lambda t:t['mean_rank_score']))
 (ROOT/'combined_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 lines=['# 指标探索结果与建议','','本次尝试60个候选：54种选择权重/阶段扩展/距离核函数组合，以及6种全轨迹匹配设置。全部由已有原始训练轨迹离线计算，ID不进有效分子；正式TASR表未被候选覆盖。','',
  '## 一、确实存在阶段截断问题','',
  'SC原指标排除约41%–43%的OOD后段；把完整OOD夹爪路径纳入匹配后，3倍分数为kNN0.9567、BC0.4954、Diff0.6461、Recovery0.7240。这更符合“绝大多数OOD监督应被保留”的语义，但BC/Recovery/Diff的SR排名仍解释不了。','',
  '## 二、全部候选的关键结果','','| 定义 | SC Spearman | Plane | YCB | OpenDrawer |','|---|---:|---:|---:|---:|']
 for title,trial in [('当前OOD目标段3倍',a['named']['ood_stage_x3']),('只看OOD预算占比',None)]:
  vals=[a['purity'][g]['correlation']['spearman'] for g in a['groups']] if trial is None else [trial['cohorts'][g]['spearman'] for g in a['groups']]
  lines.append('| '+title+' | '+' | '.join('常数，无排序' if v is None else f'{v:.3f}' for v in vals)+' |')
 for f in ['1','3','6']:lines.append('| 完整原始夹爪路径×'+f+' | '+' | '.join(f"{full[g]['correlations']['hard'][f]['spearman']:.3f}" for g in a['groups'])+' |')
 lines+=['','全路径×3的Pearson：'+', '.join(f"{g}={full[g]['correlations']['hard']['3']['pearson']:.3f}" for g in a['groups'])+'。Pearson高不意味着四种方法排序正确。','',
  '把非目标OOD段直接算有效，会破坏OpenDrawer原先的timing表现：其t0包含大量已经掌握的开柜过程，全路径版本奖励这些冗余点。OpenDrawer仍需要局部目标段；SC旧任务允许更宽目标段。任务范围应按任务机制制定，而不是给每行用不同最优半径。','',
  '## 三、留一任务检验','']
 for task,v in loto.items():lines.append(f"- 留出{task}：{v['parameters']}，留出Spearman={v['held_out_spearman']:.3f}。")
 lines+=['','这是已见任务上的交叉检验，不是全新独立验证。当前未找到在所有任务保持强排序相关的统一参数。','',
  '## 四、YCB“差很多”的来源','',
  'YCB Recovery几乎全是OOD（预算占比0.9906），但旧目标段只占其OOD后缀约0.3271，而BC约0.4271；局部形态不相容进一步压低分数。全路径×3后，BC/PCA/Recovery均约0.97–1.00，Diff约0.54，说明此前差距主要来自阶段和轨迹匹配惩罚。','',
  'YCB的SR为48/52/49/42，每组100条。不能要求一个轨迹指标精确拟合这些小差异。按相同测试seed进行配对bootstrap：','']
 for v in paired:lines.append(f"- {v['comparison']}: 差值{100*v['delta']:.1f}个百分点，95%区间[{100*v['bootstrap95'][0]:.1f}, {100*v['bootstrap95'][1]:.1f}]个百分点。")
 lines+=['','相机核对：源视频显示固定外部视角与腕部视角，BC/PCA/Diff/Recovery没有在本指标计算时切换相机。任务合同仅改变物体身份。当前指标完全不读取图像，镜头不能直接导致其分数变化；但policy学习依赖视觉，TASR也无法替代视觉覆盖检查。',
  f"\n![真实YCB训练后缀相机核对]({(ROOT/'ycb_actual_camera_contact_sheet.jpg').resolve()})",'',
  'Grab Plane还存在学习层面的异常：BC的ID ever-grasped仅41/100，PCA92/100、Diff87/100、Recovery93/100；BC并非只在OOD较差。该证据提示需检查其训练与表征适配；不能把全部SR差距归因于轨迹内容。','',
  '## 五、建议保留的单指标定义','',
  'Q = Σ_{OOD expert points} w_task(s,a) / 全部新增专家点。w_task在[0,1]内，ID贡献为0。其含义仍是有限预算里有多少监督落在当前任务待学习的部分；P（OOD预算占比）已经自然包含在这个比率中。',
  'SC候选w范围应覆盖完整任务相关OOD路径，再扣除明确恢复段；OpenDrawer Grasp应保留抓取局部段；Plane/YCB需要在成功参考的可接受动作多样性内判断，而非强行贴近单一范例。下一步若推进新正式定义，应先做可解释的片段标记及多参考校准，并检查不同运行种子；本轮候选分数不能直接改写为已验证有效的TASR。','']
 (ROOT/'最终讨论报告.md').write_text('\n'.join(lines));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
