# LU Decomposition function:
import libf
import copy


print("Q1) A and B are the matrices::")
a=libf.read_write('A.txt')    # Reading matrix A
ac=copy.deepcopy(a)
print("The matrix A(input)=")
libf.print_mat(a)             # print A
print()
n=len(a)
b=libf.read_write('B.txt')[0]    # Reading matrix B
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

X1,Y1=libf.Gaus_Jordan(ac,bc)    # Gauss Jordan function
# Output
print("For Gauss Jordan::")
print("The solution matrix X=")
for i in range(len(X1)):
    print(X1[i])

'''
OUTPUT:-

Q1) A and B are the matrices::
The matrix A(input)=
1.0  -1.0  4.0  0.0  2.0  9.0  
0.0  5.0  -2.0  7.0  8.0  4.0  
1.0  0.0  5.0  7.0  3.0  -2.0  
6.0  -1.0  2.0  3.0  0.0  8.0  
-4.0  2.0  0.0  5.0  -5.0  3.0  
0.0  7.0  -1.0  5.0  4.0  -2.0  

The matrix B=
[19.0, 2.0, 13.0, -7.0, -9.0, 2.0]


For LU decomposition::
The solution matrix x=
-1.7618170439978655
0.896228033874014
4.051931404116159
-1.6171308025395421
2.041913538501913
0.15183248715593525

For Gauss Jordan::
The solution matrix X=
-1.7618170439978567
0.8962280338740136
4.051931404116157
-1.6171308025395428
2.041913538501914
0.15183248715593495
'''
