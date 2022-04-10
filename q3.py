import libf
import csv
from matplotlib import pyplot as plt

def func_conj(x,y,N):    # Matrix generator
    i1=x%N
    i2=y%N
    j1=x//N
    j2=y//N 
    if x==y:
        return -0.96
    elif ((i1+1)%N,j1)==(i2,j2) or (i1,(j1+1)%N)==(i2,j2) or ((i1-1)%N,j1)==(i2,j2) or (i1,(j1-1)%N)==(i2,j2):
        return 0.5
    else: return 0
    

N=20
n=N**2

p1=[]
I=[[(1 if i==j else 0) for j in range(n)] for i in range(n)]
for i in range(n):
    print('Iteration no.',i)
    p,q=libf.conjugate_fly(func_conj, I[i],e=10e-6)    # sending the function as argument instead of matrix                                                                                 
    p1.append(p)
q1=[(i+1) for i in range(len(q))]

# residue plot
plt.plot(q1,q,'bo',label='Conjugate gradient method on fly')
plt.xlabel('Iterations')
plt.ylabel('Residue')
plt.legend()
plt.show()
print("The inverse of the matrix A::")
libf.print_mat(libf.transpo(p1)) 

b=libf.transpo(p1)
with open("new_file.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(b)
