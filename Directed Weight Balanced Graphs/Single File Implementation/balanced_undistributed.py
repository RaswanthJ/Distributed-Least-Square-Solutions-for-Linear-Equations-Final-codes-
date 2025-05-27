import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt


# Adjacency matrix of the graph network
Adj = np.array([
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 0]
])

n = len(Adj)

# Indegree matrix of the graph network
Din = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 2, 0],
    [0, 0, 0, 1]
])

# Identity matrices
I1 = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

I2 = [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
]

# Matrix A and vector b for each node
A1 = np.array([[5, 0, 2, 3], [5, 2, 2, 6], [3, 4, 5, 8], [0, 5, 4, 2], [8, 2, 2, 0]])
A2 = np.array([[6, 1, 6, 4], [4, 0, 4, 2], [1, 0, 3, 6], [0, 0, 0, 1], [4, 1, 2, 3]])
A3 = np.array([[2, 1, 0, 3], [6, 2, 3, 5], [1, 4, 6, 0], [2, 3, 2, 2], [0, 3, 1, 0]])
A4 = np.array([[-6, 2, 0, -5], [-8, -1, -2, -5], [0, -4, -6, -6], [1, -2, -1, -3], [-4, 0, -3, 2]])

b1 = np.array([[61], [51], [1], [55], [1]])
b2 = np.array([[112], [11], [2], [80], [1]])
b3 = np.array([[57], [38], [100], [58], [53]])
b4 = np.array([[-117], [-4], [6], [-107], [12]])

# Compute Asum and bsum
Asum = A1 + A2 + A3 + A4
bsum = b1 + b2 + b3 + b4
p = len(A1)
q = len(A1[0])
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
x = np.array([[0]] * (n * len(A1[0])))
ld = x

# Block diagonal matrix A and concatenated vector b
A = block_diag(A1, A2, A3, A4)
b = np.concatenate([b1, b2, b3, b4])
y = -1 * b

# Step sizes
alpha, beta, gamma = 25, 5e-5, 5e-3
delta1, delta2, delta3 = 1e-3, 0.1, 1e-3
iterations = 100000
time=[0]

oe = np.array([[1]]*n)
one = np.kron(oe,I2)

# Iterative updates
xvals = [x.tolist()]
for i in range(iterations):
    
    x_change =  -1*np.dot(LL, ld) - (2)*(np.dot(np.dot(A.T,one),np.dot(one.T,(np.dot(A,x)-b)))) - alpha*np.dot(LL, x)
    l_change = np.dot(LL, x)
    
    x = x + delta1 * x_change
    ld = ld + delta3 * l_change
    
    time += [i+1]
    xvals += [x.tolist()]
    
x1 = []
x2 = []
x3 = []
x4 = []
j=0



for i in range(0,iterations+1):
    x1+=xvals[i][j]
    x2+=xvals[i][j+1]
    x3+=xvals[i][j+2]
    x4+=xvals[i][j+3]

print("___________________________")


for i in range(0,n):
    print(f"Node {i+1} has values:",x[i*q:i*q+q].T)
    
plt.plot(time,x1,label=f'v1')
plt.plot(time,x2,label=f'v2')
plt.plot(time,x3,label=f'v3')
plt.plot(time,x4,label=f'v4')

plt.xlabel('Iterations')
plt.ylabel('Values')
plt.title(f'Change in values of node_{int((j+4)/4)}')
plt.legend()
plt.grid(True)
plt.show()
