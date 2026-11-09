import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.size":7.5,"font.family":"serif","axes.spines.top":False,"axes.spines.right":False})
fst=["I","II","III","IV","V","VI"]
# light-only ResNet-50, seed-averaged scores (posthoc2.log)
L_s=[.732,.752,.777,.694,.592,.410]; L_sl=[.68,.71,.73,.64,.51,.30]; L_sh=[.78,.79,.82,.74,.67,.54]
L_a=[.925,.919,.936,.946,.901,.810]; L_al=[.91,.90,.92,.93,.87,.75]; L_ah=[.94,.93,.95,.96,.93,.87]
# all-skin-type 5-fold CV (notebook 05)
C_s=[.726,.739,.768,.774,.653,.475]; C_sl=[.683,.706,.727,.724,.573,.355]; C_sh=[.765,.769,.804,.818,.725,.598]
C_a=[.897,.916,.932,.947,.921,.848]; C_al=[.880,.905,.916,.933,.894,.787]; C_ah=[.914,.927,.945,.960,.944,.901]
x=np.arange(6); w=.36
fig,ax=plt.subplots(1,2,figsize=(3.45,1.9))
for a,(Lv,Ll,Lh,Cv,Cl,Ch,t,yl) in zip(ax,[(L_s,L_sl,L_sh,C_s,C_sl,C_sh,"(a) Sensitivity",(0,1)),(L_a,L_al,L_ah,C_a,C_al,C_ah,"(b) AUC",(0.5,1))]):
    for off,v,lo,hi,c,lab,h in [(-w/2,Lv,Ll,Lh,"#4a6fa5","Trained on FST I–III",None),(w/2,Cv,Cl,Ch,"#d08c3c","Trained on all types (CV)",None)]:
        v,lo,hi=map(np.array,(v,lo,hi))
        b=a.bar(x+off,v-yl[0],w,bottom=yl[0],color=c,label=lab,edgecolor="none")
        a.errorbar(x+off,v,yerr=[v-lo,hi-v],fmt="none",ecolor="k",elinewidth=.6,capsize=1.2)
    a.set_xticks(x); a.set_xticklabels(fst); a.set_ylim(*yl); a.set_title(t,fontsize=7.5,loc="left")
    a.set_xlabel("Fitzpatrick type"); a.axvspan(4.5,5.5,color="0.9",zorder=-1)
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,frameon=False,fontsize=6.5,loc="upper center",ncol=2,handlelength=1,bbox_to_anchor=(0.5,1.0))
fig.tight_layout(pad=.3,w_pad=.8,rect=(0,0,1,.9))
fig.savefig("fig_per_fst2.pdf"); fig.savefig("fig_per_fst2.png",dpi=200)
