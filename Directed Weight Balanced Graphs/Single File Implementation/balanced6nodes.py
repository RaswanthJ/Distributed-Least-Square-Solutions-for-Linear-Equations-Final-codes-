import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt


# Adjacency matrix of the graph network
Adj = np.array([
    [0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 0]
])

Adj = Adj.T

n = len(Adj)

# Indegree matrix of the graph network
Din = np.array([
    [3, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 3],
])


# Matrix A and vector b for each node
A1 = np.array( [[0, -5, 2, 3, 1], [1, -4, 3, -5, 4], [-2, -1, 1, -1, 3], [-4, -3, 0, 2, 4], [-2, -2, 5, -1, 5], [-4, 1, -1, -4, -4], [1, -2, 5, -4, 3]] )
A2 = np.array( [[5, 4, -3, -5, -3], [1, -1, -1, -4, 1], [-1, -4, 4, -1, 2], [-5, 5, -4, 3, -1], [5, 0, -1, -3, -5], [3, 0, 4, -2, -4], [-3, -4, 5, 3, -2]] )
A3 = np.array( [[1, 4, 1, 0, -1], [-4, -2, 5, -2, 5], [3, 4, 5, 3, -5], [0, 2, -2, 3, 1], [-5, -1, -3, 5, -1], [-5, -5, 5, 3, -5], [-1, 0, 4, 0, 0]] )
A4 = np.array( [[-1, 4, 0, 5, -5], [4, -4, -3, -1, -2], [-5, -4, 0, -4, -1], [1, 4, 5, 1, 4], [3, -5, -3, -1, 5], [-4, -4, 1, -1, 0], [-3, -2, 3, 3, -5]] )
A5 = np.array( [[5, 4, 4, 4, -5], [-1, -3, 0, 3, 5], [3, -4, -3, 5, 1], [4, -4, -2, -2, 3], [3, 0, 5, 1, 4], [4, -5, 4, 1, 0], [-4, -5, 2, 3, -4]] )
A6 = np.array( [[-9, -4, 1, -1, 17], [6, 19, 0, 15, -12], [4, 13, 0, 0, 1], [8, 1, 12, -6, -8], [1, 9, 4, 1, -5], [7, 14, -12, 4, 14], [13, 20, -16, 2, 11]] )
 
b1 = np.array( [[1], [5], [8], [2], [-2], [10], [12]] )
b2 = np.array( [[-2], [-5], [-5], [5], [6], [10], [8]] )
b3 = np.array( [[0], [10], [8], [12], [1], [-1], [-4]] )
b4 = np.array( [[13], [7], [9], [11], [6], [6], [-5]] )
b5 = np.array( [[6], [-2], [11], [7], [-3], [0], [-4]] )
b6 = np.array( [[-8], [-8], [-25], [-23], [-18], [-10], [1]] )

# Compute Asum and bsum
Asum = A1 + A2 + A3 + A4 + A5 + A6
bsum = b1 + b2 + b3 + b4 + b5 + b6
p = len(A1)
q = len(A1[0])

I1 = np.eye(q)
I2 = np.eye(p)

print("Sum(Ai):")
print(Asum)
print("Sum(bi):")
print(bsum)

# Laplacian computation
L = Din - Adj
print(L)
LL = np.kron(L, I1)
LL1 = np.kron(L, I2)

# Initialize solution vectors
x = np.array([[0]] * (n * q))
ld = x


# Block diagonal matrix A and concatenated vector b
A = block_diag(A1, A2, A3, A4, A5, A6)
b = np.concatenate([b1, b2, b3, b4, b5, b6])
y = -1 * b


# Step sizes
alpha, beta, gamma = 50, 0.08, 0.06
delta = 1e-3
iterations = 10000000
time=[0]

# Iterative updates
xvals = [x.tolist()]
for i in range(iterations):
    if(i%100000==0):
        print(i)
    
    x_change =  -1*np.dot(LL, ld) - (2*n*beta)*(np.dot(A.T,y)) - alpha*np.dot(LL, x)
    l_change = 1*np.dot(LL, x)
    y_change = -1*np.dot(LL1,y) + np.dot(A,-1*np.dot(LL, ld) - (2*n*beta)*(np.dot(A.T,y)) - alpha*np.dot(LL, x))
    
    x = x + delta * x_change
    ld = ld + delta * l_change
    y = y + delta * y_change
    
    if(i%1==0):
        time += [i+1]
        xvals += [x.tolist()]
    
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
j=0

i = 0
for k in time:
    x1+=xvals[i][j]
    x2+=xvals[i][j+1]
    x3+=xvals[i][j+2]
    x4+=xvals[i][j+3]
    x5+=xvals[i][j+4]
    i+=1

print("___________________________")


for i in range(0,n):
    print(f"Node {i+1} has values:",x[i*q:i*q+q].T)

    
plt.plot(time,x1,label=f'v1')
plt.plot(time,x2,label=f'v2')
plt.plot(time,x3,label=f'v3')
plt.plot(time,x4,label=f'v4')
plt.plot(time,x5,label=f'v5')


plt.xlabel('Iterations')
plt.ylabel('Values')
plt.title(f'Change in values of node_{int((j+5)/5)}')
plt.legend()
plt.grid(True)
plt.show()

# print(x.T, "X \n")
# print(y.T, "Y \n")
# print(ld.T, "Ld \n")
