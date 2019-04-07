import h5py
import numpy as np
import matplotlib.pyplot as plt
f = h5py.File("particles/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_withPars_truth_0.z")
list(f)
f['t_allpar_new'].shape
f['t_allpar_new'][2]
dset = f['t_allpar_new'][:]
dset[0]
count=100
string=[[] for i in range(count)]
list=[[] for i in range(count)]
for i in range(count):
	string[i]=str(dset[i])
	list[i]=string[i].split(",")
	
full_list=[]
for i in range(count):
	full_list.append(list[i][2])

	

final=[]
for item in full_list:
	final.append(float(item))
	

final_list=np.asarray(final)

val=0
j_g=[]
j_q=[]
j_w=[]
j_z=[]
j_t=[]

for i in range(count):
	if list[i][72]==' 1':
		val+=1
		j_g.append(final_list[i])
		
	if list[i][73]==' 1':
		val+=1
		j_q.append(final_list[i])
		
	if list[i][74]==' 1':
		val+=1
		j_w.append(final_list[i])
		
	if list[i][75]==' 1':
		val+=1
		j_z.append(final_list[i])
		
	if list[i][76]==' 1':
		val+=1
		j_t.append(final_list[i])
		
plt.hist(j_g, histtype='step', label="j_g")
plt.hist(j_q, histtype='step', label="j_q")
plt.hist(j_w, histtype='step', label="j_w")
plt.hist(j_z, histtype='step', label="j_z")
plt.hist(j_t, histtype='step', label="j_t")

plt.xlabel('Feature Value')
plt.ylabel('No. of particles')
plt.title('Feature plot')
labels=["j_g", "j_q", "j_w", "j_z", "j_t"]
plt.legend(labels)
plt.savefig('feature2.png')

print(val)
plt.show()



	
