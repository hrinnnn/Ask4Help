#!/usr/bin/env python3
"""Offline, expert-only diagnostic. Does not alter any experiment controller.

Legacy length-normalized phase alignment is compared with a clock-free monotone
tracker. Only the latter can score an observed takeover prefix consistently.
Reference selection uses recorded reset geometry, explicitly not a deployed gate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_xvla_erd_pose import _quat_inverse, _quat_multiply, _quat_rotvec

ANCHORS = [0, 50, 80, 120, 160, 220]
MODES = ['legacy_phase_band', 'clockfree_monotone', 'open_end_monotone_dp']


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_episode(directory, metadata):
    timeline = json.loads((directory / 'task_state_timeline.json').read_text())
    rows = timeline['rows']
    states = np.load(directory / 'states.npy')
    assert len(rows) == len(states) and states.shape[1] == 9
    assert [r['step'] for r in rows] == list(range(len(rows)))
    reset = json.loads((directory / 'reset_metadata.json').read_text())
    assert reset['split'] == 'grasp_ood'
    pose = dict(position=np.array([r['tcp_position'] for r in rows]),
                quaternion=np.array([r['tcp_quaternion'] for r in rows]),
                width=states[:, -2:].sum(axis=1)[:, None])
    assert all(np.isfinite(x).all() for x in pose.values())
    context = np.array(reset['object_pose']['p'] + reset['target_pose']['p'])
    return dict(seed=int(metadata['seed']), meta=metadata, pose=pose,
                context=context, rows=rows, directory=str(directory), reset=reset)


def residual_matrix(query, ref):
    p = query['position'][:, None] - ref['position'][None, :]
    rot = _quat_rotvec(_quat_multiply(query['quaternion'][:, None],
                                    _quat_inverse(ref['quaternion'])[None, :]))
    g = query['width'][:, None] - ref['width'][None, :]
    return np.concatenate([p, rot, g], axis=-1)


def global_map(cost):
    n, m = cost.shape
    accum = cost.copy()
    parent = np.full((n, m), -1, dtype=np.int8)
    for i in range(n):
        for j in range(m):
            if i == j == 0:
                continue
            options = [(accum[i-1, j] if i else np.inf),
                       (accum[i, j-1] if j else np.inf),
                       (accum[i-1, j-1] if i and j else np.inf)]
            k = int(np.argmin(options))
            accum[i, j] += options[k]
            parent[i, j] = k
    mapping = np.zeros(n, dtype=int)
    i, j = n-1, m-1
    while True:
        mapping[i] = max(mapping[i], j)
        if i == j == 0:
            break
        k = parent[i, j]
        i -= int(k in (0, 2))
        j -= int(k in (1, 2))
    return mapping


def align(cost, mode, ahead=0, behind=0, jump=5):
    n, m = cost.shape
    if mode == 'open_end_monotone_dp':
        # Offline Viterbi path: one query observation per transition, reference
        # increment 0..jump. Endpoint is free, so failure need not reach success.
        # Backtracking uses future query evidence; NOT an online gate.
        parent = np.zeros((n, m), dtype=int)
        previous_cost = np.full(m, np.inf)
        previous_cost[:min(m,jump+1)] = cost[0,:min(m,jump+1)]
        for i in range(1,n):
            shifted = np.full((jump+1,m),np.inf)
            shifted[0] = previous_cost
            for k in range(1,min(jump+1,m)):
                shifted[k,k:] = previous_cost[:-k]
            increments = np.argmin(shifted,axis=0)
            parent[i] = np.arange(m)-increments
            previous_cost = cost[i]+np.min(shifted,axis=0)
        mapping = np.zeros(n,dtype=int)
        mapping[-1] = int(np.argmin(previous_cost))
        for i in range(n-1,0,-1):
            mapping[i-1] = parent[i,mapping[i]]
        return mapping
    mapping, previous = np.zeros(n, dtype=int), 0
    for i in range(n):
        lo, hi = previous, min(m-1, previous+jump)
        if mode == 'legacy_phase_band':
            u = i / max(1, n-1)
            lo = max(previous, int(np.floor(max(0, u-behind)*(m-1))))
            hi = min(hi, int(np.ceil(min(1, u+ahead)*(m-1))))
            if lo > hi:  # Preserve historical fallback, disclose its limitation.
                lo, hi = previous, max(previous, min(m-1, int(np.ceil(min(1, u+ahead)*(m-1)))))
        previous = lo + int(np.argmin(cost[i, lo:hi+1]))
        mapping[i] = previous
    return mapping


def onset(values, threshold):
    checks = np.arange(0, len(values), 5)
    for a, b in zip(checks[:-1], checks[1:]):
        if values[a] > threshold and values[b] > threshold:
            return int(a), int(b)
    return None, None


def qstats(values):
    values = [x for x in values if x is not None]
    return dict(n=len(values), p25=float(np.quantile(values, .25)) if values else None,
                median=float(np.median(values)) if values else None,
                p75=float(np.quantile(values, .75)) if values else None)


def main(root):
    inputs = root / 'inputs'
    experts = [load_episode(inputs/'formal/anchor_0/accepted'/f"episode_{r['episode_index']:06d}", r)
               for r in read_jsonl(inputs/'formal/anchor_0/accepted_experts.jsonl')]
    # Frozen by seed before looking at policy scores; 18 fit, 6 calibration, 6 test.
    refs = [e for e in experts if e['seed'] % 5 < 3]
    cal = [e for e in experts if e['seed'] % 5 == 3]
    test = [e for e in experts if e['seed'] % 5 == 4]
    policy_meta = json.loads((inputs/'policy20/summary.json').read_text())['rows']
    policies = [load_episode(inputs/'policy20/episodes'/f"episode_{r['episode_index']:06d}", r) for r in policy_meta]
    contexts = np.array([e['context'] for e in refs])
    cscale = np.maximum(1.4826*np.median(abs(contexts-np.median(contexts, axis=0)), axis=0), .001)

    def reference(item):
        ds = [np.linalg.norm((item['context']-e['context'])/cscale)
              if item['seed'] != e['seed'] else np.inf for e in refs]
        return refs[int(np.argmin(ds))], float(min(ds))

    # Bootstrap metric with physical units, then fit matching scales on refs only.
    floors = np.array([.001]*3 + [.01]*3 + [.001])
    initial_scale = np.array([.01]*3 + [.1]*3 + [.01])
    fit_residuals, pairs = [], []
    for e in refs:
        peer, distance = reference(e)
        residual = residual_matrix(e['pose'], peer['pose'])
        mapping = global_map(np.linalg.norm(residual/initial_scale, axis=2))
        fit_residuals.extend(residual[np.arange(len(mapping)), mapping])
        pairs.append(dict(seed=e['seed'], reference=peer['seed'], context_distance=distance))
    feature_scale = np.maximum(1.4826*np.median(abs(np.array(fit_residuals)), axis=0), floors)
    offsets = []
    for e in refs:
        peer, _ = reference(e)
        residual = residual_matrix(e['pose'], peer['pose'])
        mapping = global_map(np.linalg.norm(residual/feature_scale, axis=2))
        offsets.extend(mapping/max(1, len(peer['pose']['position'])-1)-np.arange(len(mapping))/max(1, len(mapping)-1))
    ahead = float(np.quantile([v for v in offsets if v > 0], .975))
    behind = float(np.quantile([-v for v in offsets if v < 0], .975))
    out = dict(diagnostic_only=True, protocol=dict(
        reference_seeds=[e['seed'] for e in refs], calibration_seeds=[e['seed'] for e in cal],
        heldout_expert_seeds=[e['seed'] for e in test], policy_seeds=[e['seed'] for e in policies],
        threshold_quantile=.925, threshold_unit='pooled expert points sampled every 5 environment steps',
        decision_stride=5, persistence=2, max_reference_jump_per_environment_step=5,
        reference_selection='nearest reset object+target position; simulator-context diagnostic, not proprioception-only deployment',
        phase_band=dict(ahead=ahead, behind=behind), feature_scale=feature_scale.tolist(), reference_pairs=pairs,
        N_star_status='no exact same-reset full expert for takeover seeds; E remains unavailable'), modes={})

    for mode in MODES:
        def matched(item):
            peer, context_distance = reference(item)
            residual = residual_matrix(item['pose'], peer['pose'])
            cost = np.linalg.norm(residual/feature_scale, axis=2)
            mapping = align(cost, mode, ahead, behind)
            return residual[np.arange(len(mapping)), mapping, :3], mapping, peer, context_distance

        rp = np.concatenate([matched(e)[0] for e in refs])
        path_scale = np.maximum(1.4826*np.median(abs(rp-np.median(rp, axis=0)), axis=0), .001)
        calibration_values = np.concatenate([np.linalg.norm(matched(e)[0]/path_scale, axis=1)[::5] for e in cal])
        tau = float(np.quantile(calibration_values, .925))

        def score(item, group):
            residual, mapping, peer, context_distance = matched(item)
            values = np.linalg.norm(residual/path_scale, axis=1)
            t, confirmation = onset(values, tau)
            events = {name: next((r['step'] for r in item['rows'] if r.get(name)), None)
                      for name in ['ever_drawer_opened', 'ever_grasped', 'ever_lifted', 'success']}
            return dict(seed=item['seed'], group=group, reference_seed=peer['seed'],
                        context_distance=context_distance, reference_actions=len(peer['rows'])-1,
                        steps=len(values)-1, Tref=t, Tconfirm=confirmation,
                        onset_phase=item['rows'][t].get('phase') if t is not None else None,
                        events=events, D=values.tolist(), D_over_tau=(values/tau).tolist(),
                        aligned_reference_steps=mapping.tolist(), video=item['meta'].get('video'))

        rows = [score(e, group) for group, items in [('expert_calibration',cal), ('expert_heldout',test), ('policy_ood',policies)] for e in items]
        stats = {}
        for group in ['expert_calibration', 'expert_heldout', 'policy_ood']:
            gr = [r for r in rows if r['group'] == group]
            stats[group] = dict(episodes=len(gr), crossed=sum(r['Tref'] is not None for r in gr),
                               onset=qstats([r['Tref'] for r in gr]),
                               before_drawer_open=sum(r['Tref'] is not None and r['events']['ever_drawer_opened'] is not None and r['Tref'] < r['events']['ever_drawer_opened'] for r in gr))
        sensitivity = []
        for quantile in [.90, .925, .95, .975]:
            threshold = float(np.quantile(calibration_values, quantile))
            sensitivity.append(dict(q=quantile,tau=threshold,groups={g:dict(
                n=len([r for r in rows if r['group']==g]),
                onset=qstats([onset(r['D'], threshold)[0] for r in rows if r['group']==g]))
                for g in ['expert_heldout','policy_ood']}))
        result = dict(path_scale=path_scale.tolist(), tau=tau, calibration_points=len(calibration_values),
                      rows=rows, statistics=stats, sensitivity=sensitivity)
        maxima = []
        for row in rows:
            if row['group'] == 'expert_calibration':
                sampled = np.array(row['D'][::5])
                maxima.append(float(np.max(np.minimum(sampled[:-1],sampled[1:]))))
        trajectory_tau = float(np.quantile(maxima,.925))
        result['trajectory_level_threshold_sensitivity'] = dict(
            diagnostic_only=True, tau=trajectory_tau, calibration_maxima=maxima,
            groups={g:dict(n=len([r for r in rows if r['group']==g]),
                          onset=qstats([onset(r['D'],trajectory_tau)[0] for r in rows if r['group']==g]))
                    for g in ['expert_heldout','policy_ood']})

        if mode == 'clockfree_monotone':
            takeover_rows = []
            for anchor in ANCHORS:
                folder = inputs/'formal'/f'anchor_{anchor}'
                for meta in read_jsonl(folder/'raw_attempts.jsonl'):
                    relative = Path(meta['evidence_dir']).name
                    directory = folder/'raw_attempts'/relative
                    if not (directory/'task_state_timeline.json').exists():
                        takeover_rows.append(dict(anchor=anchor, seed=meta['seed'], status='missing_evidence'))
                        continue
                    e = load_episode(directory, meta)
                    take = int(meta['actual_takeover_step'])
                    e['pose'] = {k:v[:take+1] for k,v in e['pose'].items()}
                    e['rows'] = e['rows'][:take+1]
                    scored = score(e, 'actual_takeover_prefix')
                    late = scored['Tref'] is not None
                    takeover_rows.append(dict(anchor=anchor, seed=e['seed'], accepted=meta['accepted'],
                        success=meta['success'], actual_takeover_step=take,
                        expert_actions=meta['expert_action_steps'], D_at_take=scored['D_over_tau'][-1],
                        Tref_observed=scored['Tref'], E=None,
                        D_take=scored['D_over_tau'][-1] if late else None,
                        status='late_confirmed_in_prefix' if late else 'future_autonomous_Tref_unknown',
                        reference_seed=scored['reference_seed']))
            result['takeover_rows'] = takeover_rows
            result['takeover_summary'] = {str(a):dict(
                attempts=len([r for r in takeover_rows if r['anchor']==a]),
                late_confirmed=sum(r['status']=='late_confirmed_in_prefix' for r in takeover_rows if r['anchor']==a),
                D_take=qstats([r.get('D_take') for r in takeover_rows if r['anchor']==a]),
                accepted_mean_expert_actions=float(np.mean([r['expert_actions'] for r in takeover_rows if r['anchor']==a and r.get('accepted')]))
                ) for a in ANCHORS}
        out['modes'][mode] = result
        print(mode, json.dumps(stats), flush=True)

    previous_output = root/'analysis.json'
    if previous_output.exists() and not (root/'analysis_initial_two_modes.json').exists():
        (root/'analysis_initial_two_modes.json').write_text(previous_output.read_text())
    previous_output.write_text(json.dumps(out, indent=2, allow_nan=False))
    write_report(root,out)
    plot(root, out)
    print('ANALYSIS_COMPLETE', root, flush=True)


def plot(root, out):
    fig, axes = plt.subplots(len(out['modes']), 2, figsize=(13, 4*len(out['modes'])), constrained_layout=True)
    for index, (mode, result) in enumerate(out['modes'].items()):
        ax, hist = axes[index]
        for group, color, label in [('expert_heldout','#2676b8','Held-out expert (6)'), ('policy_ood','#d65b39','OOD autonomous (20)')]:
            rows = [r for r in result['rows'] if r['group']==group]
            size = max(len(r['D']) for r in rows)
            matrix = np.full((len(rows),size),np.nan)
            for i,r in enumerate(rows):
                matrix[i,:len(r['D'])] = r['D_over_tau']
            lo, mid, hi = np.nanquantile(matrix,[.25,.5,.75],axis=0)
            ax.plot(mid,color=color,label=label)
            ax.fill_between(np.arange(size),lo,hi,color=color,alpha=.18)
            ts = [r['Tref'] for r in rows if r['Tref'] is not None]
            hist.hist(ts,bins=np.arange(0,421,20),alpha=.55,color=color,label=f'{label}: {len(ts)}/{len(rows)} crossed')
        ax.axhline(1,color='black',ls='--',lw=1)
        ax.set(title=mode, xlabel='Environment step',ylabel='D / expert q92.5 threshold',yscale='symlog')
        ax.set_ylim(bottom=0)
        if mode == 'clockfree_monotone':
            ax.set_yscale('linear')
        ax.legend(fontsize=8)
        hist.set(title='First persistent crossing (not optimal timing)',xlabel='Tref (environment steps)',ylabel='Episodes')
        hist.legend(fontsize=8)
    fig.savefig(root/'D_Tref_overview.png',dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(4,5,figsize=(17,10),sharex=True,sharey=True,constrained_layout=True)
    for ax, seed in zip(axes.flat,out['protocol']['policy_seeds']):
        for mode,color in zip(MODES,['#888888','#d65b39','#238450']):
            row = next(r for r in out['modes'][mode]['rows'] if r['group']=='policy_ood' and r['seed']==seed)
            ax.plot(row['D_over_tau'],color=color,lw=1,label=mode)
            if row['Tref'] is not None:
                ax.axvline(row['Tref'],color=color,ls='--',lw=.8)
        if row['events']['ever_drawer_opened'] is not None:
            ax.axvline(row['events']['ever_drawer_opened'],color='#2676b8',ls=':',lw=1)
        ax.axhline(1,color='black',ls='--',lw=.6)
        ax.set(title=f'seed {seed}',yscale='symlog',ylim=(0,1000))
    fig.suptitle('Grey: legacy | Orange: greedy | Green: open-end DP | Blue dotted: drawer opened | dashed vertical: Tref')
    fig.savefig(root/'all_policy_curves.png',dpi=150)
    plt.close(fig)


def write_report(root, out):
    lines = ['# OpenDrawer 分段 E/D：第一阶段离线审计', '',
        '状态：第一阶段诊断完成；参考时刻可靠性未通过，未启动新配对采集或训练。', '',
        '## 资产与冻结口径', '',
        '- 新 direct-grasp Oracle：180 条 accepted / 262 次 attempts，六个 anchor=0/50/80/120/160/220。',
        '- 完整自主分支：20 条 Grasp-OOD，每条400动作。旧100条自主资产已盘点，但本次未混入。',
        '- t=0 的30条专家按 seed mod 5 拆分：余数0/1/2共18条参考及尺度拟合；余数3共6条阈值校准；余数4共6条专家留出检查。',
        '- 最终 D 只取TCP位置；姿态和夹爪宽度参与匹配。宽度来自9D Panda状态的末两维之和。',
        '- 参考用reset object+target position选择；这是已披露的仿真context诊断，不是仅依赖本体感知的部署方法。初始机器人位姿差异尚未纳入参考条件。',
        '- q=.925，校准对象为6条专家轨迹每5个environment steps采样的逐点D；连续2个检查点超过阈值，首点为Tref，第二点为确认时刻。',
        '- 每条轨迹保留真实长度，不延长专家曲线，不把未越阈填0或400。无校准policy、无SR参与拟合。', '',
        '## 对齐算法审计', '',
        '1. legacy_phase_band：重用旧长度归一化相位带逻辑；不是历史原分数的复刻，因为专家、拆分和尺度重新拟合。',
        '2. clockfree_monotone：去掉总时长约束，贪心参考索引单调、每环境步最多前进5个参考样本；可在自主前缀上计算，但会卡在局部最优。',
        '3. open_end_monotone_dp：保留所有单调候选的累计匹配代价并回溯，参考终点自由；修复贪心卡点，但使用未来query证据，明确仅作离线Tref对照，不声称在线报警。',
        '三者在相同参考、校准、留出划分上分别校准。第3个算法是看到对齐失败后的工程诊断；这6条留出专家现在不是可反复挑方案的最终测试集。', '',
        '| 对齐 | 专家留出持续越阈 | OOD持续越阈 | OOD Tref中位数 | OOD在开抽屉前越阈 |',
        '|---|---:|---:|---:|---:|']
    for mode,m in out['modes'].items():
        e=m['statistics']['expert_heldout'];p=m['statistics']['policy_ood']
        lines.append(f"| {mode} | {e['crossed']}/{e['episodes']} | {p['crossed']}/{p['episodes']} | {p['onset']['median']} | {p['before_drawer_open']}/{p['episodes']} |")
    lines += ['',
        '这不是三种算法择优榜：贪心版较少误报同时伴随错误匹配抬高阈值，不能据此把其85步中位数当成可信最佳时间。逐点q92.5也不保证92.5%的完整专家轨迹都不越阈。', '',
        '### 已定位的贪心错误', '',
        '成功专家78324相对参考78312：t=75被贪心配到j=45，TCP距离0.364918m；独立全时序DTW配到j=75，距离0.004606m。专家78303有同类错误（0.365563m对0.003221m）。这是阶段错配，不是失败恢复。',
        '不依赖总时长不能独自保证匹配正确。DP对照去掉贪心陷阱后，起始自然位姿变化/参考覆盖和逐点阈值问题仍在，留出专家5/6越阈，故本轮不冻结Tref。', '',
        '## 已有接管数据的可计算范围', '',
        '所有E均为缺失，不是0：没有同自主轨迹的未来Tref，也没有逐reset完整专家N*。不把t=0自动归为early。',
        '下表仅保留贪心版本的prefix诊断。D是已在前缀内确认late的条件分布，不能作为最终论文指标。', '',
        '| anchor | raw attempts | prefix内已确认late | late组D/tau中位数 | accepted平均专家动作 |',
        '|---:|---:|---:|---:|---:|']
    for anchor,s in out['modes']['clockfree_monotone']['takeover_summary'].items():
        median=s['D_take']['median']
        lines.append(f"| {anchor} | {s['attempts']} | {s['late_confirmed']} | {median if median is not None else '--'} | {s['accepted_mean_expert_actions']:.2f} |")
    lines += ['',
        '- late的当前D可以回落到阈值以下；它不是累计偏离，不能强制D>=1。',
        '- 30条accepted的专家动作均值不代表采集总成本：failed尝试也消耗专家动作。逐attempt字段保存在analysis.json。',
        '- 新Oracle下游SR尚无完整矩阵；没有将Oracle成功率当作训练模型SR，没有将一个模型SR复制到各采集episode当独立样本。',
        '- 原六组使用不同reset池；旧同seed自主轨迹抽查的动作/状态前缀不一致，因此不作为反事实配对。', '',
        '## 下一步的最小配对补采清单', '',
        '在参考检查通过之前，不启动新的多anchor采集/训练。优先解决同reset参考及校准统计单位，避免围绕错误Tref采集。',
        '候选配对池固定为78700–78719（现有20条自主分支）。逐条核验base/norm/控制与reset，不能只核对seed；若不能重现旧前缀，则新根下成套生成自主与专家分支，禁止混合。',
        '每个reset需要完整新Oracle示范得到N*、可重现自主轨迹/完整环境快照，以及从该自主轨迹不同时刻分叉的expert continuation。actions.npy/states.npy不等于完整仿真快照。',
        '先检查同reset成功参考可消除多少初始差异，同时补独立成功示范校准集。阈值统计单位（逐点、逐轨迹持续最大值或相位条件）必须在最终测试前冻结；不能用目标Tref或SR选阈值。',
        '保留原六anchor并查看相对Tref覆盖；覆盖不足时新版本预注册参考前/附近/后条件。后续独立标注恢复动作，与E/专家成本及D/恢复分别比较；匹配训练预算、步数、eval seeds后才能接SR。', '',
        '## 文件', '',
        '- analysis.json：完整参数、所有D曲线、Tref、确认时间、删失、q敏感性及262条原始前缀诊断。',
        '- D_Tref_overview.png / all_policy_curves.png：算法与全部20条自主曲线对照。',
        '- videos/：初版贪心匹配诊断，4个案例；videos_open_end_monotone_dp/：DP诊断，4个同案例。',
        '- video_audit*.json：帧数、状态步偏移、参考视频、选取帧和输出路径。',
        '- 不将任何一版视频里的Tref标为最佳接管时刻。', '']
    (root/'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
