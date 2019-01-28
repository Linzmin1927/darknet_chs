import matplotlib.pyplot as plt
import sys
# import numpy as np
# plt.rcParams['font.sas-serig']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# data=np.loadtxt('loss.txt',delimiter='\t') 
# print(data)
# x = [row[0] for row in data]
# y = [row[3] for row in data]
# z = [row[4] for row in data]
# s = [row[1] for row in data]
x = []
y = []
z = []
s = []
f = open(sys.argv[1], 'r')
lines = f.readlines() 
f.close()
avg = -1.0
 
for line in lines :
	l = line.strip().split('\t') 
	x.append(float(l[0]))
	loss = float(l[3]) 
	if(avg < 0.0) :
		avg = loss
	else :
		avg = 0.9 * avg + 0.1 * loss
	
	y.append( loss )
	z.append( avg )
	

plt.title('Loss Tendency of Palms Recognition Training')
# plt.axis([0,6,0,6])
plt.plot(x, y, color='green', label='Loss')

plt.plot(x, z, color='blue', label='Avg Loss')

plt.xlabel('Iteration times')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()

# plt.savefig('scatter.png')