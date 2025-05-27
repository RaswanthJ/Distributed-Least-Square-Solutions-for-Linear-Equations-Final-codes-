import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import time

st = time.time()

#Unbalanced:

Adj = np.array([
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0]
])

Din = np.array([
    [1, 0, 0, 0],
    [0, 3, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

n = len(Adj)


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

p = len(A1)
q = len(A1[0])

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


# Eigen vector Values
v = np.array([[5]] * (n * len(A1[0])))
v_dash = np.array([[5]] * (n * len(A1)))



# Block diagonal matrix A and concatenated vector b
A = block_diag(A1, A2, A3, A4)
b = np.concatenate([b1, b2, b3, b4])
y = -1 * b

# Step sizes
alpha, beta = 25, 5e-2
delta = 1e-3
iterations = 100000
times=[0]
cap = 100

oe = np.array([[1]]*q)
ol = np.array([[1]]*p)
one = np.kron(oe,I2)

xvals = [x.tolist()]

# Iterative updates
for i in range(iterations):

    V = np.diag(v.flatten())
    V_dash = np.diag(v_dash.flatten())
    x_change =  -1*np.dot(np.dot(LL,V), ld) - (2*n*beta)*(np.dot(A.T,y)) - alpha*np.dot(np.dot(LL,V), x)
    l_change =  np.dot(np.dot(LL,V), x)
    y_change = -1*np.dot(np.dot(LL1,V_dash),y) + np.dot(A,-1*np.dot(np.dot(LL,V), ld) - (2*n*beta)*(np.dot(A.T,y)) - alpha*np.dot(np.dot(LL,V), x))
    v_change = -np.dot(LL,v)
    v_dash_change = -np.dot(LL1,v_dash)

    x = x + delta * x_change
    ld = ld + delta * l_change
    y = y + delta * y_change
    v = v + delta * v_change
    v_dash = v_dash + delta * v_dash_change
    
    
    if (i%cap==0):
        times += [i+1]
        xvals += [x.tolist()]
    
    
x1 = []
x2 = []
x3 = []
x4 = []
j=0



for i in range(0,iterations//cap+1):
    x1+=xvals[i][j]
    x2+=xvals[i][j+1]
    x3+=xvals[i][j+2]
    x4+=xvals[i][j+3]

print("___________________________")

#print(x.T, "X \n")
#print(y.T, "Y \n")
#print(ld.T, "Ld \n")

for i in range(0,n):
    print(f"Node {i+1} has values:",x[i*q:i*q+q].T)
    
end_t = time.time()

print("Time taken:",end_t-st)
    
plt.plot(times,x1,label=f'v1')
plt.plot(times,x2,label=f'v2')
plt.plot(times,x3,label=f'v3')
plt.plot(times,x4,label=f'v4')

plt.xlabel('Iterations')
plt.ylabel('Values')
plt.title(f'Change in values of node_{int((j+4)/4)}')
plt.legend()
plt.grid(True)
#plt.xscale("log")
plt.show()


