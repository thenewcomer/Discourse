from collections import *

def evaluate(predict, truth):
  label = []
  for i in truth:
    if i not in label:
       label.append(i)
  result = []
  for i in label:
    tf = tt = ft = ff = 0.0000000000001
    for indice in range(len(predict)):
      if (predict[indice] == truth[indice]):
        if (truth[indice] != i):
          ff += 1
        else:
          tt += 1
      else:
        if (truth[indice] != i):
          ft += 1
        else:
          tf += 1
    p = float(tt)/(tt+ft)
    r = float(tt)/(tt+tf)
    f = float(2*p*r)/(p+r)
    acc = float((tt+ff))/(len(predict))
    result.append({'label':i, 'p':p, 'r':r, 'f':f, 'acc':acc})
  return result
  
#print evaluate(["comparison","expansion","temporal","temporal"], ["comparison","expansion","temporal","temporal"])


import numpy as np
def score(ref,sys, neutralset=set()):
    def div(a,b):
        return float(a)/b if b!=0 else 0
    N_pres=Counter(ref)
    N_pred=Counter(sys)
    N_corr=Counter([sys[i] for i in range(len(sys)) if ref[i]==sys[i]])
    labs=sorted(N_pres.keys())+['Total']
    #labs = set(labs) - neutralset
    N_pres['Total'] = 0
    for lab in N_pres.keys():
        if lab not in neutralset:
            N_corr['Total']+=N_corr[lab]
            N_pres['Total']+=N_pres[lab]
            N_pred['Total']+=N_pred[lab]
    prt=[[lab, N_corr[lab], N_pres[lab], N_pred[lab], div(N_corr[lab],N_pred[lab]), div(N_corr[lab],N_pres[lab]), div(N_corr[lab]*2.0, N_pred[lab]+N_pres[lab])] for lab in labs]
    out=[['%-12s'%'Label']+[('%12s'%lab) for lab in ['N_correct','N_present','N_predict','Precision','Recall','F1']]]
    for row in prt:
        its=[('%-12s'%row[0]) if len(row[0])<=12 else row[0]]+[('%12s'%w) for w in row[1:4]]+['%12s'%('%.4f'%w) for w in row[4:]]
        if len(its[0])>12:
            ex=12-len(its[1].strip())
            its[0]=its[0][0:12+ex]
            its[1]=its[1][ex:]
        out+=[its]
    N_pres.pop("Total")
    #for no_use_label in neutralset:
    #    N_pres.pop(no_use_label)
    out+=[['Average F1: ',str(np.mean([div(N_corr[lab] * 2.0, N_pred[lab] + N_pres[lab]) for lab in N_pres.keys()]))]]
    return '\n'.join([''.join(row) for row in out]), div(N_corr['Total']*2.0, N_pred['Total']+N_pres['Total'])