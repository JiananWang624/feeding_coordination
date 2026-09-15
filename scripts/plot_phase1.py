from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/phase1')
if __name__=='__main__':
 a=p.parse_args(); out=Path(a.output); m=json.loads((out/'manifest.json').read_text()); ds=[np.load(out/x['file']) for x in m['records']]; d=ds[0]
 fig=plt.figure(); ax=fig.add_subplot(projection='3d'); [ax.plot(d[k][:,0],d[k][:,1],d[k][:,2],label=k.replace('_xyz','')) for k in ('shoulder_xyz','elbow_xyz','wrist_xyz')]; ax.set(title=f"Arm trajectory: {m['records'][0]['record_id']}",xlabel='B x (m)',ylabel='B y (m)',zlabel='B z (m)'); ax.legend(); fig.tight_layout(); fig.savefig(out/'arm_3d.png',dpi=160); plt.close(fig)
 fig,ax=plt.subplots(); [ax.plot(x['time'],x['psi_unwrapped'],label=m['records'][i]['record_id']) for i,x in enumerate(ds[:4])]; ax.set(title='Stereo-SEW on selected demonstrations',xlabel='event time (s)',ylabel='unwrapped psi (rad)'); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(out/'psi.png',dpi=160); plt.close(fig)
 fig,ax=plt.subplots(); ax.plot(d['time'],np.linalg.norm(d['shoulder_displacement'],axis=1)); ax.set(title=f"Shoulder displacement: {m['records'][0]['record_id']}",xlabel='event time (s)',ylabel='displacement (m)'); fig.tight_layout(); fig.savefig(out/'shoulder_displacement.png',dpi=160); plt.close(fig)
 fig,ax=plt.subplots(); valid=m['summary']['psi_valid_samples']; invalid=m['summary']['psi_invalid_samples']; tool=m['summary']['tool_pose_valid_samples']; ax.bar(['psi valid','psi invalid','tool valid','tip calibration missing'],[valid,invalid,tool,m['summary']['tool_tip_calibration_missing_demonstrations']]); ax.set(title='Phase 1.5 validity overview',ylabel='frames (tip calibration: demos)'); fig.tight_layout(); fig.savefig(out/'validity.png',dpi=160); plt.close(fig)
 fig=plt.figure(); ax=fig.add_subplot(projection='3d'); tool=d['tool_pose_valid']; ax.plot(d['tool_position'][tool,0],d['tool_position'][tool,1],d['tool_position'][tool,2]); ax.set(title=f"Fork trajectory: {m['records'][0]['record_id']}",xlabel='B x (m)',ylabel='B y (m)',zlabel='B z (m)'); fig.tight_layout(); fig.savefig(out/'fork_trajectory.png',dpi=160); plt.close(fig)
 fig,ax=plt.subplots(); takes=list(m['matching']['per_take']); coverage=[m['matching']['per_take'][x]['valid_percent'] for x in takes]; ax.bar(takes,coverage); ax.set(title='Fork pose validity by take',xlabel='take',ylabel='valid (%)',ylim=(0,100)); ax.tick_params(axis='x',rotation=45); fig.tight_layout(); fig.savefig(out/'tool_validity_by_take.png',dpi=160); plt.close(fig)
