"""Compose report figures from current exports and report-only application diagnostics.
Run from any directory after export_evidence.cpp has been compiled/run at project root.
This script does not run or modify the application pipeline.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
A=HERE/'assets'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def read(path): return np.array(Image.open(path))
def panel(ax,path,title):
    ax.imshow(read(path),cmap='gray',vmin=0,vmax=255)
    ax.set_title(title,fontsize=8,pad=3)
    ax.axis('off')
def save(fig,name):
    fig.savefig(A/name,bbox_inches='tight',pad_inches=.04,dpi=220)
    plt.close(fig)

fig,axs=plt.subplots(1,4,figsize=(7,1.35),width_ratios=[2.5,1,1,1],layout='constrained')
paths=[ROOT/'final_output/detection/AllBalls/filtered_circles.png',A/'detection_grey_blur.png',ROOT/'final_output/detection/AllBalls/raw_circles.png',ROOT/'final_output/detection/AllBalls/filtered_circles.png']
for j,(ax,path,title) in enumerate(zip(axs,paths,['Detected circles on masked table','Smoothed detail','Raw pocket circle','Pocket rejected'])):
 img=read(path)
 if j==1:img=img[120:225,1120:1260]
 if j>=2:img=img[0:75,920:1010]
 ax.imshow(img,cmap='gray',vmin=0,vmax=255);ax.set_title(title,fontsize=8,pad=3);ax.axis('off')
save(fig,'detection_stages.pdf')

fig,axs=plt.subplots(3,4,figsize=(6.8,2.1),layout='constrained')
for row,(prefix,label) in enumerate([('Table2_candidate_12','Solid 1'),('Table2_candidate_5','Stripe 9'),('Table1_candidate_6','Cue')]):
 for col,(suffix,title) in enumerate([('original','Original'),('mask','Circular mask'),('kmeans','K=2 clusters'),('body','Histogram body pixels')]):
  panel(axs[row,col],A/f'{prefix}_{suffix}.png',f'{label}: {title}' if col==0 else title)
save(fig,'recognition_stages.pdf')

w=np.linspace(0,1,501)
fig,ax=plt.subplots(figsize=(3.4,1.7),layout='constrained')
for low,high,label in [(0,.3,'Solid'),(.3,.85,'Stripe'),(.85,1,'Cue')]:
 d=np.maximum.reduce([np.zeros_like(w),low-w,w-high]);ax.plot(w,np.exp(-.5*(d/.1)**2),label=label)
ax.set(xlabel='Visible light fraction w',ylabel='Type compatibility',ylim=(-.02,1.08))
ax.legend(loc='lower center',ncol=3,fontsize=7,frameon=False)
save(fig,'type_support.pdf')

rows=list(csv.DictReader((A/'colour_scores.csv').open()))
selected=[next(r for r in rows if r['image']=='Table2' and r['trueNumber']==n) for n in ['1','9','8']]
fig,ax=plt.subplots(figsize=(3.6,1.7),layout='constrained')
cols=['yellow','blue','red','purple','orange','green','maroon','black']
values=np.array([[float(r[c]) for c in cols] for r in selected])
ax.imshow(values,vmin=0,vmax=1,cmap='Blues',aspect='auto')
ax.set_xticks(range(8),cols,rotation=40,ha='right',fontsize=7)
ax.set_yticks(range(3),['Solid 1','Stripe 9','Black 8'])
for (y,x),v in np.ndenumerate(values):ax.text(x,y,f'{v:.2f}',ha='center',va='center',fontsize=6,color='white' if v>.55 else 'black')
save(fig,'colour_scores.pdf')

fig,axs=plt.subplots(2,3,figsize=(4.4,1.7),layout='constrained')
evidence=list(csv.DictReader((A/'candidate_evidence.csv').open()))
raw=read(ROOT/'Media/Table3.jpeg')
for row,(prefix,candidate,label) in enumerate([('Table3_candidate_2','2','Full cue'),('Table3_candidate_4','4','Ball 12 patch')]):
 record=next(r for r in evidence if r['image']=='Table3' and r['candidate']==candidate)
 x,y,r=[float(record[k]) for k in ['x','y','radius']]
 left,top=round(x)-45,round(y)-45
 axs[row,0].imshow(raw[top:top+90,left:left+90])
 axs[row,0].add_patch(plt.Circle((x-left,y-top),r,fill=False,color='#f4ae27',linewidth=.8))
 axs[row,0].set_title(f'{label}: {r:.2f} px',fontsize=9,pad=3)
 axs[row,0].axis('off')
 for col,suffix,title in [(1,'original','Original crop'),(2,'padded','Radius +1 pixel')]:
  crop=read(A/f'{prefix}_{suffix}.png');height,width=crop.shape[:2]
  axs[row,col].imshow(crop,extent=(-width/2,width/2,height/2,-height/2))
  axs[row,col].set(xlim=(-32,32),ylim=(32,-32))
  axs[row,col].axis('off');axs[row,col].set_title(title,fontsize=9,pad=3)
save(fig,'white_padding.pdf')

fig,axs=plt.subplots(3,2,figsize=(6.9,3.1),layout='constrained')
counts=[(11,14,2),(16,16,0),(15,15,1),(14,14,1),(13,14,0),(11,13,0)]
for i,ax in enumerate(axs.flat,1):
 c,m,miss=counts[i-1]
 panel(ax,ROOT/f'final_output/final/Table{i}/original_numbered.png',f'Table{i}: {c} correct identities / {m+miss} annotated balls')
save(fig,'final_gallery.pdf')

fig,axs=plt.subplots(2,3,figsize=(6.9,1.85),layout='constrained')
for col,(prefix,title,diagnostic) in enumerate([
 ('Table1_candidate_7','Table1: true 14 → Unknown','body'),
 ('Table6_candidate_13','Table6: true 10 → 4','body'),
 ('Table6_candidate_11','Table6: true 3 → 11','kmeans')]):
 panel(axs[0,col],A/f'{prefix}_original.png',title)
 panel(axs[1,col],A/f'{prefix}_{diagnostic}.png','Retained histogram pixels' if diagnostic=='body' else 'K=2: 66.5% counted as light')
save(fig,'failure_details.pdf')

fig,axs=plt.subplots(2,3,figsize=(6.7,1.65),layout='constrained')
for row,(prefix,label) in enumerate([('Table2_candidate_3','True cue'),('Table4_candidate_9','True solid')]):
 for col,(suffix,title) in enumerate([('original',label),('kmeans','K=2: correct'),('cca_light','Light mask: CCA → stripe')]):panel(axs[row,col],A/f'{prefix}_{suffix}.png',title)
save(fig,'cca_examples.pdf')

# Same pixel window for each detection variant; no resizing or repositioning of circles.
fig,axs=plt.subplots(1,3,figsize=(6.9,1.65),layout='constrained')
for ax,suffix,title in zip(axs,['baseline','local','watershed'],['Baseline: 15 TP / 0 FP','Local Hough: 16 TP / 1 FP','Watershed: 15 TP / 1 FP']):
 img=read(A/f'guided_Table3_{suffix}.png')
 ax.imshow(img[70:290,1270:1590]);ax.axis('off');ax.set_title(title,fontsize=8)
save(fig,'guided_details.pdf')

# Diagnostic region generated by the actual simplified watershed experiment.
paths=sorted(A.glob('guided_Table3_diagnostic_*.png'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
if paths:
 fig,axs=plt.subplots(1,3,figsize=(6.9,1.35),layout='constrained')
 for ax,path,title in zip(axs,paths[:3],['Crowded search crop','Foreground and marker cores','Watershed basins']):panel(ax,path,title)
 save(fig,'guided_stages.pdf')
 # The unchanged diagnostic search crop is exactly x=1421..1577, y=77..200
 # in Media/Table3.jpeg. Use that same window for all detection variants.
 fig,axs=plt.subplots(1,5,figsize=(7.1,1.2),layout='constrained')
 for ax,suffix,title in zip(axs[:3],['baseline','local','watershed'],['Baseline','Guided Hough','After watershed']):
  ax.imshow(read(A/f'guided_Table3_{suffix}.png')[77:201,1421:1578])
  ax.axis('off');ax.set_title(title,fontsize=8,pad=3)
 for ax,path,title in zip(axs[3:],paths[1:3],['Foreground / seeds','Watershed regions']):panel(ax,path,title)
 save(fig,'guided_comparison.pdf')

# Per-identity counts from saved predictions and the same application matching.
truth={}
with (ROOT/'data/ground_truth/BallLocationsNew.csv').open(newline='') as f:
 for r in csv.reader(f):
  if r and r[0].startswith('Table'):
   truth[r[0].split('.')[0]]=[0 if i==15 else i+1 for i in range(16) if 2*i+2<len(r) and r[2*i+1] and r[2*i+2]]
evidence=list(csv.DictReader((A/'candidate_evidence.csv').open()))
counts={n:[0,0,0] for n in range(16)}
for image,ids in truth.items():
 for n in ids:counts[n][0]+=1
for r in evidence:
 if r['image']=='AllBalls' or int(r['trueNumber'])<0:continue
 n=int(r['trueNumber']);counts[n][1]+=1;counts[n][2]+=int(r['predictedNumber'])==n
fig,ax=plt.subplots(figsize=(6.8,2.05),layout='constrained')
x=np.arange(16);matched=[counts[n][1] for n in range(16)];correct=[counts[n][2] for n in range(16)]
ax.axvspan(-.5,.5,color='#f2f2f2',zorder=0)
ax.axvspan(8.5,15.5,color='#eef4f7',zorder=0)
ax.bar(x-.25,[counts[n][0] for n in range(16)],width=.24,color='#cbd3db',label='Annotated')
ax.bar(x,matched,width=.24,color='#87b7d3',label='Detected')
ax.bar(x+.25,correct,width=.24,color='#255775',label='Correct identity')
ax.set_xticks(x,list(map(str,range(16))))
ax.set(xlabel='True ball number',ylabel='Number of balls\n(Table1–6)',xlim=(-.6,15.6),ylim=(0,7.3),yticks=list(range(7)))
for position,label in [(0,'Cue'),(4.5,'Solids (1–8)'),(12,'Stripes (9–15)')]:
 ax.text(position,6.8,label,ha='center',va='center',fontsize=8)
for boundary in [.5,8.5]:ax.axvline(boundary,color='#aeb8bf',linewidth=.6)
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax.legend(loc='lower center',bbox_to_anchor=(.5,1.04),ncol=3,frameon=False,fontsize=8)
save(fig,'identity_counts.pdf')
with (A/'group_counts.csv').open('w') as f:
 f.write('identity,annotated,matched,correct\n')
 for n,c in counts.items():f.write(f'{n},{c[0]},{c[1]},{c[2]}\n')
print('Report figures generated from current exports and diagnostic evidence.')
