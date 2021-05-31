
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from read_data import read_data
import json
import glob
from tqdm import tqdm

def _process(line):
    """extract floorplan
    """
    bbox_x1, bbox_y1, bbox_x2, bbox_y2, walls=[], [], [], [], []
    room_type,poly,doors_, walls,out=read_data(line)
    d, all_doors = [], []

    for i in range(len(doors_)):
         if(i%4==0):
                 all_doors.append(d)
                 d=[]
         d.append(doors_[i])  
    kh=0

    for hd in range(len(all_doors)):
        dr_t, dr_in = [], []
        doors=all_doors[hd]
        for dw in range(len(doors)):      
              for nw in range(len(walls)) :
                 if(walls[nw][5]==17):
                     continue
                 if (doors[dw][0]==doors[dw][2])&(abs(doors[dw][0]-walls[nw][0])<=1) & (abs(doors[dw][2]-walls[nw][2])<=1):      
                     l=doors[dw][1]
                     r=doors[dw][3]
                     if(l>r):
                        t=l
                        l=r
                        r=t
                    
                     l_=walls[nw][1]
                     r_=walls[nw][3]
                     if(l_>r_):
                        t=l_
                        l_=r_
                        r_=t
                     if(r<=r_ )& (l>=l_):
                          dr_t.append(walls[nw][5])
                          dr_in.append(nw)
                 if(doors[dw][1]==doors[dw][3])&(abs(doors[dw][1]-walls[nw][1])<=1) & (abs(doors[dw][3]-walls[nw][3])<=1): 
                     l=doors[dw][0]
                     r=doors[dw][2]
                     if(l>r):
                        t=l
                        l=r
                        r=t
                
                     l_=walls[nw][0]
                     r_=walls[nw][2]
                     if(l_>r_):
                        t=l_
                        l_=r_
                        r_=t
                     if(r<=r_ )& (l>=l_):
                          dr_t.append(walls[nw][5])
                          dr_in.append(nw)
        if(len(dr_t)==2):
                walls[dr_in[0]][8]=walls[dr_in[1]][5]
                walls[dr_in[0]][7]=walls[dr_in[1]][6]    
                walls[dr_in[1]][8]=walls[dr_in[0]][5]
                walls[dr_in[1]][7]=walls[dr_in[0]][6]   
    for kw in range(len(walls)):
         for nw in range(len(walls)):
             if(abs(walls[kw][0]-walls[nw][0])<=1) & (abs(walls[kw][2]-walls[nw][2])<=1):
                     l=walls[kw][1]
                     r=walls[kw][3]
                     if(l>r):
                        t=l
                        l=r
                        r=t
                    
                     l_=walls[nw][1]
                     r_=walls[nw][3]
                     if(l_>r_):
                        t=l_
                        l_=r_
                        r_=t
                     if(r>=r_ )& (l<=l_) &( nw!=kw):
                        if(walls[kw][5]==17):
                               walls[kw][8]=walls[nw][5]
                               walls[kw][7]=walls[nw][6]    
                        if(walls[nw][5]==17): 
                              walls[nw][8]=walls[kw][5]
                              walls[nw][7]=walls[kw][6]   

         
             if(abs(walls[kw][1]-walls[nw][1])<=1) & (abs(walls[kw][3]-walls[nw][3])<=1):
                     l=walls[kw][0]
                     r=walls[kw][2]
                     if(l>r):
                        t=l
                        l=r
                        r=t

                     l_=walls[nw][0]
                     r_=walls[nw][2]
                     if(l_>r_):
                        t=l_
                        l_=r_
                        r_=t

                     if(r>=r_ )& (l<=l_) &( nw!=kw):
                         if(walls[kw][5]==17):
                              walls[kw][8]=walls[nw][5]
                              walls[kw][7]=walls[nw][6]    
                         if(walls[nw][5]==17):
                             walls[nw][8]=walls[kw][5]
                             walls[nw][7]=walls[kw][6]   

    for iw in range(len(walls)):
      tp_out, dif_x, dif_y, type_out = -1, 10, 10, 0
      for jw in range(len(walls)):            
          if(walls[iw][0]==walls[iw][2]):
               if (walls[jw][0]!=walls[jw][2]):
                     continue
               if ((walls[iw][0]-walls[jw][0])==(walls[iw][2]- walls[jw][2])):
                          rnp=walls[jw][1]
                          fnp=walls[jw][3]
                          rmp=walls[iw][1]
                          fmp=walls[iw][3]
                          if( rnp<fnp):
                               t=fnp
                               fnp=rnp
                               rnp=t
                          if(rmp<fmp):
                               t=fmp
                               fmp=rmp
                               rmp=t
                          if(abs(rmp)<=abs(rnp))| (abs(fmp)<=abs(fnp)):
                              dif_x_temp=walls[iw][0]-walls[jw][0]
                              if(abs(dif_x)>abs(dif_x_temp)) & (iw!=jw):
                                  dif_x=dif_x_temp
                                  tp_out=walls[jw][6]
                                  type_out=walls[jw][5]
          elif(walls[iw][1]==walls[iw][3]):
               if ((walls[iw][1]-walls[jw][1])==(walls[iw][3]- walls[jw][3])) :           
                          rnp=walls[jw][0]
                          fnp=walls[jw][2]
                          rmp=walls[iw][0]
                          fmp=walls[iw][2]
                          if(rnp<fnp):
                               t=fnp
                               fnp=rnp
                               rnp=t
                          if(rmp<fmp):
                               t=fmp
                               fmp=rmp
                               rmp=t
                          if(abs(rmp)<=abs(rnp))| (abs(fmp)<=abs(fnp)):
                             dif_y_temp=walls[iw][1]-walls[jw][1]
                             if(abs(dif_y)>abs(dif_y_temp))&( iw!=jw ):
                                dif_y=dif_y_temp
                                tp_out=walls[jw][6]
                                type_out=walls[jw][5]   

    km, lenx, leny, min_x, min_y = 0, 1.0, 1.0, 0.0, 0.0
    bboxes, edges, ed_rm = [], [], []  
    info=dict()
    for w_i in range(len(walls)):
         edges.append([((walls[w_i][0]-min_x)/lenx),((walls[w_i][1]-min_y)/leny),((walls[w_i][2]-min_x)/lenx),((walls[w_i][3]-min_y)/leny),walls[w_i][5],walls[w_i][8]])
         if(walls[w_i][6]==-1):
            ed_rm.append([walls[w_i][7]])
         elif(walls[w_i][7]==-1): 
            ed_rm.append([walls[w_i][6]])
         else:
            ed_rm.append([walls[w_i][6],walls[w_i][7]])
    
    for i in range(len(poly)):
        p=poly[i]
        pm=[]
        for p_i in range((p)):
                if(p_i%2==0):
                    pm.append(([edges[km+p_i][0],edges[km+p_i][1]]))
                else:
                    pm.append(([edges[km+p_i][2],edges[km+p_i][3]]))
        km=km+p
    info['room_type'] = room_type
    info['boxes'] = bboxes
    info['edges'] = edges
    info['ed_rm'] = ed_rm
    return info


def parse_args():
    parser = argparse.ArgumentParser(description="Code for preprocessing RPLAN")
    parser.add_argument("--rplan_path", required=True,
                        help="dataset path", metavar="DIR")
    parser.add_argument("--outdir", required=True,
                        help="output dir path", metavar="DIR")
    return parser.parse_args()

def main():
    args = parse_args()
    # create output dir
    os.makedirs(args.outdir, exist_ok=True) 
    files = glob.glob(os.path.join(args.rplan_path, '*.png'))
    for fpath in tqdm(files):
      # process floorplan
      json_data = _process(fpath)
      # save to json
      _basename = os.path.basename(fpath)
      with open(os.path.join(args.outdir, _basename.replace('png', 'json')), "w") as f:
        json.dump(json_data, f)

if __name__ == "__main__":
    main()