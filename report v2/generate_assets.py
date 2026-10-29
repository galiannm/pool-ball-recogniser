"""Rebuild report graphics/tables from saved experiment outputs; does not tune the system."""
import csv
import json
from pathlib import Path
import shutil
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
TABLES = HERE / "tables"
SELECTED = "pad1_row_largest_cue"
BASE = ROOT / "output/experiments/row_cue_size"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
ASSETS.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(ASSETS / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(ASSETS / (name + ".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def image(name):
    a = cv.imread(str(ROOT / "Media" / name))
    if a is None:
        raise ValueError(name)
    return a


def circles(name):
    return np.array([[float(r[k]) for k in ("originalX", "originalY", "originalRadiusPixels")]
                     for r in read(BASE / "final_results" / SELECTED / Path(name).stem / "balls.csv")])


def roi(a, c, padding=0):
    x, y, r = (round(float(v)) for v in (c[0], c[1], c[2] + padding))
    x0, y0 = max(0, x-r), max(0, y-r)
    crop = a[y0:min(a.shape[0], y+r+1), x0:min(a.shape[1], x+r+1)].copy()
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv.circle(mask, (x-x0, y-y0), r, 255, -1)
    return crop, mask


def show(ax, a, title):
    ax.imshow(cv.cvtColor(a, cv.COLOR_BGR2RGB) if a.ndim == 3 else a, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


# Pipeline: training, inference, and localisation have explicitly separate paths.
fig, ax = plt.subplots(figsize=(10.5, 3.0))
ax.set(xlim=(0, 12), ylim=(0, 4.3)); ax.axis("off")
def box(x, y, text, colour="#e5eef6", w=3.5):
    ax.add_patch(FancyBboxPatch((x,y), w,.7,boxstyle="round,pad=0.06",fc=colour,ec="#395269",lw=1))
    ax.text(x+w/2,y+.35,text,ha="center",va="center",fontsize=10)
def arrow(a, b, **kwargs):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=11,color="#395269",lw=1,**kwargs))
for x,t in [(0.15,"Input: table mask\nand smoothing"),(4.2,"Hough circles\nand pocket rejection"),(8.25,"Circular ROI; K=2\ninitial type evidence")]: box(x,3.1,t)
arrow((3.7,3.45),(4.15,3.45));arrow((7.75,3.45),(8.2,3.45))
for x,t in [(8.25,"White candidates: +1 pixel;\nrecompute type evidence"),(4.2,"HSV histogram comparison\n+ type/identity scores"),(0.15,"Reserve cue if ambiguous;\nunique identity assignment")]: box(x,1.65,t)
arrow((10,3.05),(10,2.4));arrow((8.2,2),(7.75,2));arrow((4.15,2),(3.7,2))
box(.15,.15,"AllBalls training: colour models + row/radius fit",colour="#ecf3e9",w=7.55)
box(8.25,.15,"Final identities and\ntable coordinates",colour="#f7edda")
arrow((1.9,1.6),(1.9,1.1));arrow((1.9,1.1),(10,1.1));arrow((10,1.1),(10,.9))
arrow((5.95,3.05),(5.95,2.85));arrow((5.95,2.85),(11.95,2.85));arrow((11.95,2.85),(11.95,.5));arrow((11.95,.5),(11.8,.5))
ax.text(8.2,2.86,"centre transformation",fontsize=8,va="bottom")
arrow((5.95,.9),(5.95,1.6),linestyle="--");arrow((.6,.9),(.6,1.6),linestyle="--")
save(fig,"pipeline")

# Original, table mask, preprocessing and retained detections.
a = image("AllBalls.jpeg")
corners = np.array([[337,35],[1570,21],[1888,451],[38,472]],np.int32)
mask = np.zeros(a.shape[:2],np.uint8);cv.fillConvexPoly(mask,corners,255)
masked = cv.bitwise_and(a,a,mask=mask)
smooth = cv.GaussianBlur(cv.cvtColor(masked,cv.COLOR_BGR2GRAY),(5,5),1.5)
detected = a.copy()
for x,y,r in circles("AllBalls.jpeg"):
    cv.circle(detected,(round(x),round(y)),round(r),(0,255,255),2)
for y in (float(circles("AllBalls.jpeg")[:,1].min()),float(circles("AllBalls.jpeg")[:,1].max())):
    cv.line(detected,(337,round(y)),(1570,round(y)),(255,220,40),2)
fig, axes=plt.subplots(2,2,figsize=(10,3.1))
for ax,im,t in zip(axes.flat,[a,masked,smooth,detected],["(a) Original","(b) Table mask applied","(c) Smoothed intensity","(d) Retained Hough circles"]):show(ax,im,t)
fig.tight_layout(pad=.4);save(fig,"detection_stages")

# Recognition panels, reconstructed with the identical deterministic K2 settings.
models=cv.FileStorage(str(ROOT/"output/experiments/ball_colour_histograms/training/models.yml"),cv.FILE_STORAGE_READ)
nodes=models.getNode("models")
hists=None
for i in range(nodes.size()):
    node=nodes.at(i)
    if node.getNode("space").string()=="hsv_histogram":
        entries=node.getNode("histograms");hists=[entries.at(j).getNode("histogram").mat() for j in range(entries.size())]
if hists is None:raise ValueError("Missing frozen models")
models.release()
predictions=read(BASE/"predictions.csv")
fig,axes=plt.subplots(2,5,figsize=(10,3.9))
for row,(name,index,padding) in enumerate([("AllBalls.jpeg",4,0),("AllBalls.jpeg",8,0)]):
    crop,m=roi(image(name),circles(name)[index],padding)
    samples=crop[m>0].astype(np.float32)
    cv.setRNGSeed(12345)
    _,labels,centres=cv.kmeans(samples,2,None,(cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER,50,.2),3,cv.KMEANS_PP_CENTERS)
    colours=np.clip(np.rint(centres),0,255).astype(np.uint8)
    segmented=np.zeros_like(crop);segmented[m>0]=colours[labels[:,0]]
    centreHSV=cv.cvtColor(colours.reshape(1,-1,3),cv.COLOR_BGR2HSV)[0]
    light=np.zeros_like(m)
    for j,c in enumerate(centreHSV):
        if c[2]>=170 and c[1]<=165:
            light[m>0]=np.where(labels[:,0]==j,255,light[m>0])
    expected=float(next(r["lightPercent"] for r in predictions if r["image"]==name and r["method"]=="C_unique" and int(r["candidate"])==index+1))
    actual=100*np.count_nonzero(light)/np.count_nonzero(m)
    if abs(actual-expected)>1e-4:raise ValueError("Illustrative K2 reconstruction disagrees with saved evidence")
    hsv=cv.cvtColor(crop,cv.COLOR_BGR2HSV)
    body=cv.erode(m,cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)),borderType=cv.BORDER_CONSTANT,borderValue=0)
    body[(hsv[:,:,2]>=170)&(hsv[:,:,1]<=165)]=0
    features=hsv.copy();chroma=crop.max(axis=2).astype(int)-crop.min(axis=2).astype(int);features[:,:,0][chroma<15]=0
    h=np.minimum(features[:,:,0].astype(int)*18//180,17);s=features[:,:,1].astype(int)*8//256;v=features[:,:,2].astype(int)*8//256
    # Yellow solid5? actual example at index4 is ball1; index8 is ball9, shared yellow model.
    response=hists[0][h,s,v].astype(float);response[m==0]=0
    response=np.uint8(np.clip(response/max(float(hists[0].max()),1e-12)*255,0,255))
    panels=[crop,segmented,light,body,response]
    for col,(im,title) in enumerate(zip(panels,["Original crop","K=2 colours","Light clusters","Histogram body mask","Yellow response"])):
        show(axes[row,col],im, title if row==0 else "")
    axes[row,0].set_ylabel("Solid 1" if row==0 else "Striped 9")
fig.tight_layout(pad=.45);save(fig,"recognition_stages")

# Coordinate transformation and nominal-scale diagram: copied from the selected run.
folder=BASE/"final_results"/SELECTED/"Table3"
fig=plt.figure(figsize=(9,4.4))
grid=fig.add_gridspec(2,2)
axes=[fig.add_subplot(grid[0,:]),fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1])]
for ax,name,title in zip(axes,["original_numbered.png","table_numbered.png","table_diagram.png"],
    ["(a) Original-image detections and identities","(b) Rectified table coordinates","(c) Nominal 43 mm ball diagram"]):
    show(ax,cv.imread(str(folder/name)),title)
fig.tight_layout(pad=.35);save(fig,"transformation")

# Selected successes and failures with source context, annotated using saved predictions.
cases=[("Table3.jpeg",4,"(a) Purple stripe 12: Cue → 12"),
       ("Table1.jpeg",6,"(b) Actual cue: 14 → Cue"),
       ("Table1.jpeg",7,"(c) Stripe 14: Cue → Unknown"),
       ("Table6.jpeg",11,"(d) Red solid 3 → stripe 11"),
       ("Table5.jpeg",14,"(e) Blue stripe 10 → solid 4"),
       ("Table1.jpeg",15,"(f) False detection assigned 6")]
fig,axes=plt.subplots(2,3,figsize=(9,4.6))
for ax,(name,index,title) in zip(axes.flat,cases):
    a=image(name);c=circles(name)[index-1];x,y,r=c;extent=r+10
    ax.imshow(cv.cvtColor(a,cv.COLOR_BGR2RGB))
    ax.set(xlim=(x-extent,x+extent),ylim=(y+extent,y-extent));ax.axis("off")
    ax.add_patch(Circle((x,y),r,fill=False,ec="#F0E442",lw=1.4))
    ax.set_title(title,fontsize=9)
fig.tight_layout(pad=.5);save(fig,"recognition_cases")

# Detection failure: show both predicted circles and provided ground-truth centres.
gt={r[0]:r[1:] for r in csv.reader((ROOT/"data/ground_truth/BallLocations.csv").open()) if r and r[0].endswith(".jpeg")}
src=corners.astype(np.float32);dst=np.array([[0,0],[1433,0],[1433,682],[0,682]],np.float32)
inverse=np.linalg.inv(cv.getPerspectiveTransform(src,dst))
fig,axes=plt.subplots(1,2,figsize=(8,2.6))
for ax,(name,ids) in zip(axes,[("Table1.jpeg",[6,7]),("Table4.jpeg",[10])]):
    a=image(name);points=[]
    for number in ids:
        values=gt[name];p=np.array([float(values[2*(number-1)+1]),float(values[2*(number-1)]),1.])
        original=inverse@p;points.append(original[:2]/original[2])
    mid=np.mean(points,axis=0);extent=85 if len(ids)>1 else 70
    ax.imshow(cv.cvtColor(a,cv.COLOR_BGR2RGB))
    for x,y,r in circles(name):ax.add_patch(Circle((x,y),r,fill=False,ec="#F0E442",lw=1.2))
    for number,p in zip(ids,points):ax.plot(*p,"rx",ms=8,mew=2);ax.text(p[0]+3,p[1]-4,str(number),color="red",fontsize=10)
    ax.set(xlim=(mid[0]-extent,mid[0]+extent),ylim=(mid[1]+extent,mid[1]-extent));ax.axis("off")
    ax.set_title(name.replace(".jpeg","")+": missed "+", ".join(map(str,ids)),fontsize=10)
fig.tight_layout(pad=.4);save(fig,"detection_failures")

# Alternative segmentation and chamfer evidence, cropped from existing experiment sheets.
source=ROOT/"output/experiments/chamfer_refinement/Table1/candidates/candidate_7.png"
chamfer=cv.imread(str(source))
for column,(title,count) in enumerate(zip(["Original: cue","Padding +1: cue","Minimum cost: cue","Outer supported: 14"],[0,8,0,15])):
    x=column*280
    chamfer[0:40,x:x+280]=25
    chamfer[285:330,x:x+280]=25
    cv.putText(chamfer,title,(x+7,24),cv.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1,cv.LINE_AA)
    cv.putText(chamfer,"Body pixels: "+str(count),(x+7,311),cv.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1,cv.LINE_AA)
fig,ax=plt.subplots(figsize=(9,3.2));show(ax,chamfer,"Chamfer alternatives: original / +1 / minimum cost / outer supported")
fig.tight_layout(pad=.25);save(fig,"chamfer_case")
watershed=cv.imread(str(ROOT/"output/experiments/watershed_detection/Table4/region_1_diagnostics.png"))
fig,axes=plt.subplots(1,3,figsize=(10,3.0),gridspec_kw={"width_ratios":[1.25,1.25,4.3]})
show(axes[0],watershed[32:313,57:239],"(a) Hough cluster")
show(axes[1],watershed[353:634,649:831],"(b) Watershed")
show(axes[2],chamfer,"(c) Chamfer crops and body evidence")
fig.tight_layout(pad=.3);save(fig,"alternatives")
for name in ["row_radius_samples","row_radius_fit","row_radius_estimation"]:
    for ext in ["pdf","png"]:shutil.copy2(ROOT/"report/assets/row_radius"/(name+"."+ext),ASSETS/(name+"."+ext))

# Appendix: preserve all six selected final outputs in one high-resolution overview.
fig,axes=plt.subplots(3,2,figsize=(10,8.1))
for ax,number in zip(axes.flat,range(1,7)):
    folder=BASE/"final_results"/SELECTED/f"Table{number}"
    show(ax,cv.imread(str(folder/"table_numbered.png")),f"Table {number}")
    shutil.copy2(folder/"table_numbered.png",ASSETS/f"final_table{number}.png")
fig.tight_layout(pad=.55,h_pad=.9,w_pad=.7);save(fig,"final_tables")

# Numeric tables are generated directly from the selected run.
metrics=read(BASE/"metrics.csv")
selected=[r for r in metrics if r["method"]==SELECTED and r["image"]!="AllBalls.jpeg"]
lines=[r"\begin{tabular}{lrrrrrrr}",r"\toprule",r"Image & GT & TP & FP & FN & Error & Correct ID & ID acc.\\",r"\midrule"]
for r in selected:
    label=r["image"].replace(".jpeg","")
    lines.append(f"{label} & {r['groundTruth']} & {r['TP']} & {r['FP']} & {r['FN']} & {float(r['meanError']):.2f} & {r['correctNumbers']}/{r['TP']} & {100*float(r['numberAccuracy']):.2f}\\% " + r"\\")
lines.extend([r"\bottomrule",r"\end{tabular}"])
(TABLES/"per_image.tex").write_text("\n".join(lines)+"\n")
(HERE/"provenance.json").write_text(json.dumps({"selectedMethod":SELECTED,"metrics":str(BASE.relative_to(ROOT)/"metrics.csv"),
    "predictions":str(BASE.relative_to(ROOT)/"predictions.csv"),"recognitionFigure":"Reconstructed K2 light fractions verified against saved C++ evidence",
    "figures":"Own visualisations of supplied images and saved experiment outputs; not new evaluation runs"},indent=2)+"\n")
print("Generated report figures and metric table; K2 illustration checks passed.")
