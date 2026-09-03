#!/usr/bin/env python3
"""Verified pi05 table. Never attribute historical Deep-kNN to Bridge-PCA."""
import argparse,json
from pathlib import Path

SR={'object':{'offline_oracle':48,'failure_recovery':49,'diffdagger':42,'bridge_pca':52},
 'airplane':{'offline_oracle':0,'failure_recovery':63,'diffdagger':84,'bridge_pca':81},
 'stackcube':{'offline_oracle':49,'failure_recovery':29,'diffdagger':34,'bridge_knn':81}}
METHODS=['offline_oracle','failure_recovery','diffdagger','bridge_pca']

def main(root):
 data={t:json.loads((root/f'tasr_{t}.json').read_text()) for t in SR};rows=[]
 evidence={}
 for file in (root/'inputs').glob('utility_evidence_*.json'):
  for cohort,group in json.loads(file.read_text()).items():
   for item in group['files']:
    s=item['summary']
    if s.get('split')!='ood':continue
    if cohort=='ycb_object':task='object';method=Path(item['relative_path']).parts[0]
    elif cohort.startswith('pi05_plane_'):task='airplane';method=Path(item['relative_path']).parts[0]
    elif cohort.startswith('pi05_stack_'):
     task='stackcube';method={'knn':'bridge_knn','offline':'offline_oracle','diff':'diffdagger','recovery':'failure_recovery'}[cohort.removeprefix('pi05_stack_')]
    else:continue
    endpoint='ever_grasped' if task=='airplane' else 'success';n=s['episodes'];count=sum(bool(r[endpoint]) for r in s['rows'])
    assert n==len(s['rows'])==100 and count==SR[task][method]
    evidence[task,method]=dict(summary=item['path'],episodes=n,successes=count,endpoint=endpoint)
 for task,analysis in data.items():
  for method,s in analysis['summary'].items():rows.append(dict(task=task,method=method,SR_percent=SR[task][method],sr_evidence=evidence[task,method],**s))
 complete=all(r['missing_expert_points']==0 for r in rows)
 (root/'pi05_tasr_table_values.json').write_text(json.dumps(dict(status='COMPLETE' if complete else 'PARTIAL',radius_factor=3,rows=rows),indent=2,allow_nan=False))
 def score(task,m):
  value=data[task]['summary'].get(m,{}).get('TASR',{}).get('3');return '--' if value is None else f'{value:.4f}'
 def table_row(task,label):
  cells=[label]
  for m in METHODS:
   value=SR[task].get(m)
   sr='--' if value is None else str(value)
   if task=='airplane' and m=='diffdagger':sr=r'\textbf{84}'
   if task=='object' and m=='bridge_pca':sr=r'\textbf{52}'
   if task=='object' and m=='diffdagger':sr=r'$42^{\ddagger}$'
   cells.extend([sr,score(task,m)])
  return '        '+' & '.join(cells)+r' \\'
 tex=r'''% Computed TASR uses the fixed factor-three tolerance, not a fitted SR model.
\begin{table}[t]
    \centering
    \caption{Active robot-gated learning results on $\pi_{0.5}$.
    SR denotes unassisted OOD success after fine-tuning.
    TASR measures the target alignment of collected expert supervision.
    Higher is better; ``--'' denotes an unavailable verified result.}
    \label{tab:active_learning_results}
    \footnotesize
    \setlength{\tabcolsep}{1pt}
    \renewcommand{\arraystretch}{1.06}
    \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l*{4}{rr}@{}}
        \toprule
        & \multicolumn{2}{c}{BC} & \multicolumn{2}{c}{HG-DAgger}
        & \multicolumn{2}{c}{Diff-DAgger} & \multicolumn{2}{c}{Ours} \\
        \cmidrule(lr){2-3}\cmidrule(lr){4-5}
        \cmidrule(lr){6-7}\cmidrule(lr){8-9}
        Task & SR & TASR & SR & TASR & SR & TASR & SR & TASR \\
        \midrule
        \multicolumn{9}{l}{\textit{Position OOD}} \\
'''+table_row('stackcube',r'SC--Green$^{\dagger}$')+r'''
        \midrule
        \multicolumn{9}{l}{\textit{Pose OOD}} \\
'''+table_row('airplane','PickPlane')+r'''
        \midrule
        \multicolumn{9}{l}{\textit{Object OOD}} \\
'''+table_row('object','YCB--Object')+r'''
        \bottomrule
    \end{tabular*}
    \vspace{3pt}
    \begin{minipage}{\columnwidth}
        \scriptsize
        SR is in percent; TASR is on $[0,1]$, using the fixed $3\times$ tolerance.
        PickPlane uses ever-grasped success; other tasks use task success.
        HG-DAgger denotes the recorded failure-recovery baseline.
        $\dagger$ SC--Green refers to the legacy red/green configuration,
        not an isolated green-target shift. The internal Deep-kNN run achieved
        SR=81 and TASR=KNN_VALUE; it is not a Bridge-PCA result under Ours.
        $\ddagger$ Low-threshold Diff-DAgger diagnostic setting (0.05).
    \end{minipage}
\end{table}
'''
 tex=tex.replace('KNN_VALUE',score('stackcube','bridge_knn'));(root/'active_learning_results_filled.tex').write_text(tex)
 actual=tex.replace(r'\multicolumn{2}{c}{Ours}',r'\multicolumn{2}{c}{Internal gate}')
 actual=actual.replace(table_row('stackcube',r'SC--Green$^{\dagger}$'),
  table_row('stackcube',r'SC--Green$^{\dagger}$').replace(r' & -- & -- \\',r' & \textbf{81} & '+score('stackcube','bridge_knn')+' '+r'\\'))
 actual=actual.replace('it is not a Bridge-PCA result under Ours.','this row uses Deep-kNN, while the other internal gates use Bridge-PCA.')
 (root/'active_learning_all_actual_gates.tex').write_text(actual)
 lines=['# π0.5 TASR 与 Downstream Utility','',f"状态：{'12个真实方法/任务结果已计算完整' if complete else '仍有不满足回放一致性要求的输入，以下空格不是0'}。统一主报告使用3倍半径，未按SR挑选参数。",'',
  '| 任务 | 方法 | SR (%) | TASR ×1 | TASR ×2 | TASR ×3 | 有效/总专家点 |','|---|---|---:|---:|---:|---:|---:|']
 for r in rows:
  vals=['--' if r['TASR'][str(f)] is None else f"{r['TASR'][str(f)]:.4f}" for f in [1,2,3]]
  lines.append(f"| {r['task']} | {r['method']} | {r['SR_percent']} | "+' | '.join(vals)+f" | {r['scored_expert_points']}/{r['total_expert_points']} |")
 lines+=['','## 保留的科学边界','',
  '- StackCube的81来自Deep-kNN，不是Bridge-PCA；全部12个真实结果包含这一额外方法，但不伪填Ours-PCA的空格。',
  '- Plane Diff SR为84，不是74。Plane BC的TASR可以高于其他方法，但SR为0；本批结果不能证明TASR普遍预测下游成功。',
  '- YCB Object BC与PCA的TASR也不完全按SR排序。指标只衡量所定义目标片段的监督构成，不能代替模型优化、数据覆盖和前置技能验证。',
  '- 每组分母来自该模型实际使用的新增专家轨迹；SC与Plane不同方法预算不完全一致，YCB每组严格5954点。不跨任务合并相关系数。',
  '- 参考为同任务成功OOD离线专家；seed mod 5划分参考/校准/检查，查询seed从参考和校准中移除。位置/姿态/开合单位维持2cm/15度/1cm；0.925校准分位数与1/2/3倍半径均保留。',
  '- 所有实际计分的TCP姿态和开合来自原训练qpos；补回物体位置与接触模式时使用原动作、原种子及旧采集流程。未通过目标片段重建核对的轨迹不伪计为0或丢出分母。',
  '- 数值审计不等于指标有效性假设通过；本分析属于已知SR后的离线探索。','',
  f"[Bridge-PCA归属严格版LaTeX]({(root/'active_learning_results_filled.tex').resolve()})",f"[全部真实内部gate结果LaTeX]({(root/'active_learning_all_actual_gates.tex').resolve()})",f"[完整精度数据]({(root/'pi05_tasr_table_values.json').resolve()})",f"[独立审计]({(root/'tasr_independent_audit.json').resolve()})",'']
 (root/'TASR结果与填表说明.md').write_text('\n'.join(lines));print('TABLE',complete,len(rows))
 if complete:
  audit=json.loads((root/'tasr_independent_audit.json').read_text());assert all(audit[t]['status']=='PASS' and audit[t]['unscored']==0 for t in SR)
  (root/'PI05_TASR_COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',actual_method_task_groups=12,selected_expert_trajectories=sum(r['total_selected_episodes'] for r in rows),
   expert_points=sum(r['total_expert_points'] for r in rows),missing_points=sum(r['missing_expert_points'] for r in rows),
   independent_audit='tasr_independent_audit.json',table='active_learning_results_filled.tex',scientific_caveat='No claim that TASR is universally correlated with SR.'),indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
