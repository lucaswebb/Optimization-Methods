from LineSearch import *
from TrustRegion import *

import matplotlib.pyplot as plt


# Rosenbrock function
a = 1
b = 100
rosenbrock = lambda x : (a-x[0])**2 + b*(x[1]-x[0]**2)**2

rosen_grad, rosen_hess = compute_gradient_hessian(rosenbrock, [x, y])

trust_rosen_results = trust_region(rosenbrock, rosen_grad, rosen_hess, [5.0, 5.0], [1, 1])
print('Trust Region # of iterations:', trust_rosen_results[1])


##Test Functions

def rosenbrock2d(x):##g(x,y)=Rosenbrock
    f=[[1-x[0]],[10*(x[1]-x[0]**2)]]
    return np.matrix(f)
def rosen_jacobian(x):
    j=[[-1,0],[-20*x[0],10]]
    return np.matrix(j)

##Executions

plt.figure(1)
sol=np.matrix([1.,1.]).T
# Quadratic
[x,n,log,er]=linetrace(rosenbrock2d,rosen_jacobian,quadratic,gradg,[5.0,5.0],15., sol)
plt.plot(log[0],log[1],'b', label='Quadratic')

# Wolfe
[x,n,log,er]=linetrace(rosenbrock2d,rosen_jacobian,Wolfe,gradg,[5.0,5.0],15., sol)
plt.plot(log[0],log[1],'m', label='Wolfe')

#Trust Region
plt.plot([item[0] for item in trust_rosen_results[2]], label='Trust-Region')


plt.legend()
plt.title("Rosenbrock starting at (5,5)")
plt.yscale("log")
plt.ylabel('Log of Absolute Error')
# plt.xscale("log")
# plt.xlabel('Log of # of Iterations')
plt.xlabel('# of Iterations')
# plt.xlim((0, 7500))
plt.show()

plt.figure(2)
plt.plot([item[0] for item in trust_rosen_results[2]], [item[1] for item in trust_rosen_results[2]], label='Trust-Region')
plt.xscale('log')
plt.gca().invert_xaxis()
plt.title('Rosenbrock starting at (5,5)')
plt.xlabel('Log of absolute error')
plt.ylabel('# of Function Evaluations')
plt.legend()
plt.show()

plt.figure(3)
plt.plot([item[0] for item in trust_rosen_results[2]], [item[2] for item in trust_rosen_results[2]], label='Trust-Region')
plt.xscale('log')
plt.gca().invert_xaxis()
plt.title('Rosenbrock starting at (5,5)')
plt.xlabel('Log of absolute error')
plt.ylabel('# of Gradient Evaluations')
plt.legend()
plt.show()

plt.figure(4)
plt.plot([item[0] for item in trust_rosen_results[2]], [item[3] for item in trust_rosen_results[2]], label='Trust-Region')
plt.xscale('log')
plt.gca().invert_xaxis()
plt.title('Rosenbrock starting at (5,5)')
plt.xlabel('Log of absolute error')
plt.ylabel('# of Hessian Evaluations')
plt.legend()
plt.show()

print(trust_rosen_results)