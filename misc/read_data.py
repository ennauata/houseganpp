import cv2 
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt


def read_data(line):
	poly=[]
	img = np.asarray(Image.open(line))
	img_room_type=img[:,:,1]
	img_room_number=img[:,:,2]
	room_no=img_room_number.max()
	room_imgs=[]
	rm_types=[]
	for i in range(room_no):

		room_img=np.zeros((256, 256))
	
		for k in range(256):
			for h in range(256):
				if(img_room_number[k][h]==i+1):
					room_img[k][h]=1
					k_=k
					h_=h
		rm_t=img_room_type[k_][h_]
		if(rm_t==0):
			rm_types.append(1)
		elif(rm_t==1):
			rm_types.append(3)
		elif(rm_t==2):
			rm_types.append(2)
		elif(rm_t==3):
			rm_types.append(4)
		elif(rm_t==4):
			rm_types.append(7)
		elif(rm_t==5):
			rm_types.append(3)
		elif(rm_t==6):
			rm_types.append(8)
		elif(rm_t==7):
			rm_types.append(3)
		elif(rm_t==8):
			rm_types.append(3)
		elif(rm_t==9):
			rm_types.append(5)
		elif(rm_t==10):
			rm_types.append(6)
		elif(rm_t==11):
			rm_types.append(10)
		else:
			rm_types.append(16)
		room_imgs.append(room_img)
	walls=[]
	rm_type=rm_types
	for t in range(len(room_imgs)):
		tmp=room_imgs[t]
		for k in range(254):
			for h in range(254):
				if(tmp[k][h]==1) & (tmp[k+1][h]==0) & (tmp[k+2][h]==1):
					tmp[k+1][h] =1
				
		for k in range(254):
			for h in range(254):
				if(tmp[h][k]==1) & (tmp[h][k+1]==0) & (tmp[h][k+2]==1):
					tmp[h][k+1] =1
				
		for k in range(254):
			for h in range(254):
				if(tmp[k][h]==0) & (tmp[k+1][h]==1) & (tmp[k+2][h]==0):
					tmp[k+1][h] =0
				
		for k in range(254):
			for h in range(254):
				if(tmp[h][k]==0) & (tmp[h][k+1]==1) & (tmp[h][k+2]==0):
					tmp[h][k+1] =0
		room_imgs[t]=tmp
		coords=[]
		for k in range(2,254):
			for h in range(2,254):
				if(tmp[k][h]==1):
					if((tmp[k-2][h]==0) & (tmp[k-2][h-2]==0)&(tmp[k][h-2]==0) &(tmp[k-1][h]==0) & (tmp[k-1][h-1]==0)&(tmp[k][h-1]==0)):
						coords.append([h,k,0,0,t,rm_type[t]])
					elif(tmp[k+2][h]==0)&(tmp[k+2][h-2]==0)&(tmp[k][h-2]==0)& (tmp[k+1][h]==0)&(tmp[k+1][h-1]==0)&(tmp[k][h-1]==0):
						coords.append([h,k,0,0,t,rm_type[t]])    
					elif(tmp[k+2][h]==0)&(tmp[k+2][h+2]==0)&(tmp[k][h+2]==0)& (tmp[k+1][h]==0)&(tmp[k+1][h+1]==0)&(tmp[k][h+1]==0): 
						coords.append([h,k,0,0,t,rm_type[t]])   
					elif(tmp[k-2][h]==0)&(tmp[k-2][h+2]==0)&(tmp[k][h+2]==0)& (tmp[k-1][h]==0)&(tmp[k-1][h+1]==0)&(tmp[k][h+1]==0): 
						coords.append([h,k,0,0,t,rm_type[t]])  
					elif(tmp[k+1][h]==1)&(tmp[k+2][h+2]==0)&(tmp[k][h+1]==1)& (tmp[k+1][h+1]==0):
						coords.append([h,k,0,0,t,rm_type[t]])  
					elif(tmp[k-1][h]==1)&(tmp[k-2][h+2]==0)&(tmp[k][h+1]==1)& (tmp[k-1][h+1]==0):
						coords.append([h,k,0,0,t,rm_type[t]])  
					elif(tmp[k+1][h]==1)&(tmp[k+2][h-2]==0)&(tmp[k][h-1]==1)&(tmp[k+1][h-1]==0) : 
						coords.append([h,k,0,0,t,rm_type[t]])  
					elif(tmp[k-1][h]==1) & (tmp[k-2][h-2]==0)&(tmp[k][h-1]==1) & (tmp[k-1][h-1]==0):
						coords.append([h,k,0,0,t,rm_type[t]])  
		p=0
		for c in range(len(coords)):
			for c2 in range(len(coords)):
                                              
				if(c2==c):
					continue
				if(coords[c][0]==coords[c2][0])&(coords[c][2]!=1) &(coords[c2][2]!=1):
					walls.append([coords[c][0],coords[c][1],coords[c2][0],coords[c2][1],-1,coords[c][5],coords[c][4],-1,0])
					p=p+1
					coords[c][2]=1
					coords[c2][2]=1
				
				if(coords[c][1]==coords[c2][1])&(coords[c][3]!=1) &(coords[c2][3]!=1)  :
					walls.append([coords[c][0],coords[c][1],coords[c2][0],coords[c2][1],-1,coords[c][5],coords[c][4],-1,0])  
					coords[c][3]=1
					p=p+1
					coords[c2][3]=1
		poly.append(p)

	tmp=img[:,:,1]
	door_img=np.zeros((256, 256))
		
	for k in range(256):
		for h in range(256):
			if(tmp[k][h]==17)| (tmp[k][h]==15):
				door_img[k][h]=1


	rms_type=rm_type
	tmp=door_img
	tmp=door_img
	door_img=tmp
	coords=[]
	for k in range(2,254):
		for h in range(2,254):
			if(tmp[k][h]==1):
				if((tmp[k-1][h]==0) & (tmp[k-1][h-1]==0)&(tmp[k][h-1]==0)):
					coords.append([h,k,0,0])
				elif (tmp[k+1][h]==0)&(tmp[k+1][h-1]==0)&(tmp[k][h-1]==0):
					coords.append([h,k,0,0])
				elif (tmp[k+1][h]==0)&(tmp[k+1][h+1]==0)&(tmp[k][h+1]==0): 
					coords.append([h,k,0,0])
				elif (tmp[k-1][h]==0)&(tmp[k-1][h+1]==0)&(tmp[k][h+1]==0): 
					coords.append([h,k,0,0])
				elif(tmp[k+1][h]==1)&(tmp[k][h+1]==1)& (tmp[k+1][h+1]==0):
					coords.append([h,k,0,0])
				elif(tmp[k-1][h]==1)&(tmp[k][h+1]==1)& (tmp[k-1][h+1]==0):
					coords.append([h,k,0,0])
				elif(tmp[k+1][h]==1)&(tmp[k][h-1]==1)&(tmp[k+1][h-1]==0) : 
					coords.append([h,k,0,0])
				elif(tmp[k-1][h]==1) & (tmp[k][h-1]==1) & (tmp[k-1][h-1]==0):
					coords.append([h,k,0,0])
	for k in range(len(coords)):
		for p in range(-2,3):
			for t in range(-2,3):
				tmp[coords[k][1]+p][coords[k][0]+t]=0
	tmp=door_img
	tmp=door_img

	for k in range(1,253):
		for h in range(1,253):
			if(tmp[k-1][h]==0) & (tmp[k+1][h]==1) & (tmp[k+2][h]==0):
				tmp[k+1][h] =0
				tmp[k][h]=0
			
	for k in range(1,253):
		for h in range(1,253):
			if(tmp[h][k-1]==0) & (tmp[h][k+1]==1) & (tmp[h][k+2]==0):
				tmp[h][k+1] =0
				tmp[h][k]=0
	
	for k in range(254):
		for h in range(254):
			if(tmp[k][h]==0) & (tmp[k+1][h]==1) & (tmp[k+2][h]==0):
				tmp[k][h] =1
				tmp[k+2][h]==1
				
			
	for k in range(254):
		for h in range(254):
			if(tmp[h][k]==0) & (tmp[h][k+1]==1) & (tmp[h][k+2]==0):
				tmp[h][k] =1
				tmp[h][k+2]=1
	
	coords=[]
	door_img=tmp
	for k in range(2,254):
		for h in range(2,254):
			if(tmp[k][h]==1):
				if((tmp[k-1][h]==0) & (tmp[k-1][h-1]==0)&(tmp[k][h-1]==0)):
					coords.append([h,k,0,0,-1])
				elif (tmp[k+1][h]==0)&(tmp[k+1][h-1]==0)&(tmp[k][h-1]==0):
					coords.append([h,k,0,0,-1])
				elif (tmp[k+1][h]==0)&(tmp[k+1][h+1]==0)&(tmp[k][h+1]==0): 
					coords.append([h,k,0,0,-1])
				elif (tmp[k-1][h]==0)&(tmp[k-1][h+1]==0)&(tmp[k][h+1]==0): 
					coords.append([h,k,0,0,-1])
				elif(tmp[k+1][h]==1)&(tmp[k][h+1]==1)& (tmp[k+1][h+1]==0):
					coords.append([h,k,0,0,-1])
				elif(tmp[k-1][h]==1)&(tmp[k][h+1]==1)& (tmp[k-1][h+1]==0):
					coords.append([h,k,0,0,-1])
				elif(tmp[k+1][h]==1)&(tmp[k][h-1]==1)&(tmp[k+1][h-1]==0) : 
					coords.append([h,k,0,0,-1])
				elif(tmp[k-1][h]==1) & (tmp[k][h-1]==1) & (tmp[k-1][h-1]==0):
					coords.append([h,k,0,0,-1])
	doors=[]
	h=0
	no_doors=int(len(coords)/4)
        
	for c in range(len(coords)):

		for c2 in range(len(coords)):
			if(c2==c):
				continue
			if(coords[c][0]==coords[c2][0])&(coords[c][2]!=1) &(coords[c2][2]!=1):
				coords[c][2]=1
				coords[c2][2]=1
				if(coords[c][4]==-1) &(coords[c2][4]==-1):
					coords[c2][4]=h
					coords[c][4]=h
					h=h+1
				elif(coords[c][4]!=-1):
					coords[c2][4]=coords[c][4]
					coords[c][4]=coords[c][4]

				elif (coords[c2][4]!=-1):
					coords[c2][4]=coords[c2][4]
					coords[c][4]=coords[c2][4]
				walls.append([coords[c][0],coords[c][1],coords[c2][0],coords[c2][1],-1,17,len(rms_type)+coords[c][4],-1,0])  
				doors.append([coords[c][0],coords[c][1],coords[c2][0],coords[c2][1]]) 
			if(coords[c][1]==coords[c2][1])&(coords[c][3]!=1) &(coords[c2][3]!=1):
				coords[c][3]=1
				coords[c2][3]=1
				if(coords[c][4]==-1) &(coords[c2][4]==-1):
					coords[c2][4]=h
					coords[c][4]=h
					h=h+1
				elif(coords[c][4]!=-1):
					coords[c2][4]=coords[c][4]
					coords[c][4]=coords[c][4]

				elif (coords[c2][4]!=-1):
					coords[c2][4]=coords[c2][4]
					coords[c][4]=coords[c2][4]
				walls.append([coords[c][0],coords[c][1],coords[c2][0],coords[c2][1],-1,17,len(rms_type)+coords[c][4],-1,0]) 
				doors.append([coords[c][0],coords[c][1],coords[c2][0],coords[c2][1]])

	for i in range(no_doors):
		rms_type.append(17)
		poly.append(4)
	out=1
	for i in range(len(poly)):
		if(poly[i]<4):
			out=-1
	if (len(doors)%4!=0):
			out=-3			
	return rms_type,poly,doors,walls,out
