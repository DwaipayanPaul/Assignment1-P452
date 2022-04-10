import libf
import copy
from matplotlib import pyplot as plt

print("Q2) A2 and B2 are the matrices::")
a=libf.read_write('A2.txt')    # Reading matrix A
ac=copy.deepcopy(a)
# Jacobi, GS and CG
a1=copy.deepcopy(a)
a2=copy.deepcopy(a)
a3=copy.deepcopy(a)

print("The matrix A(input)=")
libf.print_mat(a)             # print A
print()

n=len(a)
b=libf.read_write('B2.txt')[0]    # Reading matrix B
bc=copy.deepcopy(b)
print("The matrix B=")
print(b)
print()

libf.part_pivot(a,b)          # partial pivoting
x=libf.lu(a)                       # L U decomposition
print()

X=libf.forward_backward(a,b)     # forward-backward substitution
print('For LU decomposition::')
print("The solution matrix x=")
for i in range(n):          # print solutions
    print(X[i])
    
print()

xx,tt=libf.jacobi(ac, bc,10e-4)
print('For Jacobi method::')
print("The solution matrix x=")
for i in range(n):          # print solutions
    print(xx[i])


print()

# Finding inverse of A
I=[[(1 if i==j else 0) for j in range(len(a))] for i in range(len(a))]
x1,x2,x3=[],[],[]
for i in range(len(a)):
    p,q=libf.jacobi(a1, I[i],10e-4)
    q1=[i for i in range(len(q))]
    x1.append(p)
    r,t=libf.gauss_sidel(a2,I[i],10e-4)
    t1=[i for i in range(len(t))]
    x2.append(r)
    u,v=libf.conjugate(a3, I[i],e=10e-4)
    x3.append(u)
    v1=[i for i in range(len(v))]
    
    if i==0:
        plt.plot(q1,q,label='Jacobi method ')
        plt.plot(t1,t,label='Gauss-sidel ')
        plt.plot(v1,v,label='Conjugate gradient')
        plt.xlabel('Iterations')
        plt.ylabel('Residue')
        plt.legend()
        plt.show()

print('For Jacobi method::')
print("The inverse matrix of A=")
libf.print_mat(libf.transpo(x1))  



print()
print('For Gauss-sidel::')
print("The inverse matrix of A=")
libf.print_mat(libf.transpo(x2)) 

print()
print('For Conjugate gradient::')
print("The inverse matrix of A=")
libf.print_mat(libf.transpo(x3)) 
print("It didnot converge to its exact solution as conjugate gradient works for symmetric matrices")


'''
OUTPUT:-

Q2) A2 and B2 are the matrices::
The matrix A(input)=
2.0  -3.0  0.0  0.0  0.0  0.0  
-1.0  4.0  -1.0  0.0  -1.0  0.0  
0.0  -1.0  4.0  0.0  0.0  -1.0  
0.0  0.0  0.0  2.0  -3.0  0.0  
0.0  -1.0  0.0  -1.0  4.0  -1.0  
0.0  0.0  -1.0  0.0  -1.0  4.0  

The matrix B=
[-1.666, 0.666, 3.0, -1.333, -0.333, 1.666]


For LU decomposition::
The solution matrix x=
-0.33319480519480493
0.33320346320346333
0.999926406926407
-0.6663766233766232
8.225108225123601e-05
0.6665021645021646

For Jacobi method::
The solution matrix x=
-0.3318444411457584
0.33395305637331496
1.0002504792423041
-0.6650025143240597
0.0008188910682745826
0.6668319353536103

For Jacobi method::
The inverse matrix of A=
0.9364277939631936  0.8713866065954895  0.26113283646973384  0.20913149762667424  0.41687972369156634  0.17020677662165076  
0.2907738889057398  0.580793186383608  0.1739105839101  0.13927159460443211  0.27774184205798835  0.11331378006745998  
0.08690715752024498  0.17346177589985706  0.3206805232345288  0.05659847090421728  0.11286497205721706  0.10855523875699219  
0.20913149762667424  0.41687972369156645  0.1702067766216508  0.9364277939631936  0.8713866065954895  0.26113283646973384  
0.1392715946044321  0.27774184205798835  0.11331378006745998  0.29077388890573985  0.580793186383608  0.1739105839101  
0.05659847090421728  0.11286497205721707  0.10855523875699219  0.08690715752024498  0.17346177589985706  0.3206805232345288  

For Gauss-sidel::
The inverse matrix of A=
0.9356835049889147  0.8707008638352757  0.2603552481985477  0.20840021136317632  0.41613427658379865  0.1694347955174304  
0.29031940434708225  0.5803414575440043  0.173434688786332  0.1387995362734017  0.2773017213966432  0.11282355656636656  
0.08667948311173394  0.17325192478229745  0.32044514138075675  0.05637475492650232  0.11264246841183419  0.10832210356334387  
0.20829836390120343  0.4160516415860657  0.16933439434065078  0.9355624450627019  0.8705798039090628  0.2602341882723349  
0.1387540740870348  0.27726483517881384  0.11277873996591947  0.2902653661921778  0.5802874193890999  0.1733806506314276  
0.05635838929969218  0.11262918999027782  0.10830597033666906  0.08666003027967004  0.17323247195023353  0.32042568854869286  

For Conjugate gradient::
The inverse matrix of A=
0.935192899224954  0.8719707675842138  -1.7790275623458278  0.2076694899322602  0.4150655875163029  2.222802865304813  
0.29007109719171953  0.5806393413081147  -0.3801171008816391  0.13856996023723867  0.2772682894230531  0.7284244422882888  
0.08659808156000169  0.17312533603183758  0.13651305861240023  0.05632262730329626  0.11283642330075229  0.2527266201029125  
0.2076694899322602  0.4150655875163027  2.2228028653048164  0.935192899224954  0.8719707675842142  -1.7790275623458058  
0.13856996023723867  0.2772682894230533  0.7284244422882925  0.2900710971917197  0.5806393413081146  -0.3801171008816361  
0.056322627303296235  0.11283642330075255  0.25272662010291824  0.08659808156000175  0.17312533603183747  0.13651305861239552  
It didnot converge to its exact solution as conjugate gradient works for symmetric matrices
'''