from LineSearch import *
from TrustRegion import *
from sympy import symbols

import matplotlib.pyplot as plt


######### Rosenbrock function

# Trust Region
a = 1
b = 100
rosenbrock = lambda x : (a-x[0])**2 + b*(x[1]-x[0]**2)**2
rosen_grad, rosen_hess = compute_gradient_hessian(rosenbrock, [x, y])
trust_rosen_results = trust_region(rosenbrock, rosen_grad, rosen_hess, [5.0, 5.0], [1, 1])
print('Trust Region # of iterations:', trust_rosen_results[1])


# Line search was implemented differently

def rosenbrock2d(x):##g(x,y)=Rosenbrock
    f=[[1-x[0]],[10*(x[1]-x[0]**2)]]
    return np.matrix(f)
def rosen_jacobian(x):
    j=[[-1,0],[-20*x[0],10]]
    return np.matrix(j)


####


plt.figure(1)
sol=np.matrix([1.,1.]).T
# Quadratic
quad_log = linetrace(rosenbrock2d,rosen_jacobian,quadratic,gradg,[5.0,5.0],15., sol)
plt.plot(quad_log[0], quad_log[1], 'b', label='Quadratic')

# Wolfe
wolfe_log = linetrace(rosenbrock2d,rosen_jacobian,Wolfe,gradg,[5.0,5.0],15., sol)
plt.plot(wolfe_log[0], wolfe_log[1], 'm', label='Wolfe')

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
plt.plot(quad_log[1], quad_log[2], label='Quadratic')
plt.plot(wolfe_log[1], wolfe_log[2], label='Wolfe')
plt.xscale('log')
plt.yscale('log')
plt.gca().invert_xaxis()
plt.title('Rosenbrock starting at (5,5)')
plt.xlabel('Log of absolute error')
plt.ylabel('Log # of Function Evaluations')
plt.legend()
plt.show()

plt.figure(3)
plt.plot([item[0] for item in trust_rosen_results[2]], [item[2] for item in trust_rosen_results[2]], label='Trust-Region')
plt.plot(quad_log[1], quad_log[3], label='Quadratic')
plt.plot(wolfe_log[1], wolfe_log[3], label='Wolfe')
plt.xscale('log')
plt.yscale('log')
plt.gca().invert_xaxis()
plt.title('Rosenbrock starting at (5,5)')
plt.xlabel('Log of absolute error')
plt.ylabel('Log # of Gradient Evaluations')
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


######### Sphere function

startsphere=[]
for i in range(20):
    startsphere.append(5.)

solutions=[]
for i in range(20):
    solutions.append(0.)
    
sphere= lambda x:  sum(x**2)
sphere_grad, sphere_hess = compute_gradient_hessian(sphere, symbols(x:20))
trust_sphere_results = trust_region(sphere, sphere_grad, sphere_hess, startsphere, solution)
print('Trust Region # of iterations:', trust_sphere_results[1])

# Line search was implemented differently

def sphere20d(x):##g(x,y)=Sphere
    f=[]
    for xi in x:
        f.append([xi])
    return np.matrix(f)
def sphere_jacobian(x):
    return np.identity(20)


####

plt.figure(5)
sol=np.matrix(solution).T
# Quadratic
quad_log = linetrace(sphere20d,sphere_jacobian,quadratic,gradg,startsphere,15., sol)
plt.plot(quad_log[0], quad_log[1], 'b', label='Quadratic')

# Wolfe
wolfe_log = linetrace(sphere20d,sphere_jacobian,Wolfe,gradg,startsphere,15., sol)
plt.plot(wolfe_log[0], wolfe_log[1], 'm', label='Wolfe')

#Trust Region
plt.plot([item[0] for item in trust_sphere_results[2]], label='Trust-Region')

plt.legend()
plt.title("Rosenbrock starting at (5,5)")
plt.yscale("log")
plt.ylabel('Log of Absolute Error')
# plt.xscale("log")
# plt.xlabel('Log of # of Iterations')
plt.xlabel('# of Iterations')
# plt.xlim((0, 7500))
plt.show()

plt.figure(6)
plt.plot([item[0] for item in trust_sphere_results[2]], [item[1] for item in trust_sphere_results[2]], label='Trust-Region')
plt.plot(quad_log[1], quad_log[2], label='Quadratic')
plt.plot(wolfe_log[1], wolfe_log[2], label='Wolfe')
plt.xscale('log')
plt.yscale('log')
plt.gca().invert_xaxis()
plt.title('Rosenbrock starting at (5,5)')
plt.xlabel('Log of absolute error')
plt.ylabel('Log # of Function Evaluations')
plt.legend()
plt.show()

plt.figure(7)
plt.plot([item[0] for item in trust_sphere_results[2]], [item[2] for item in trust_sphere_results[2]], label='Trust-Region')
plt.plot(quad_log[1], quad_log[3], label='Quadratic')
plt.plot(wolfe_log[1], wolfe_log[3], label='Wolfe')
plt.xscale('log')
plt.yscale('log')
plt.gca().invert_xaxis()
plt.title('Rosenbrock starting at (5,5)')
plt.xlabel('Log of absolute error')
plt.ylabel('Log # of Gradient Evaluations')
plt.legend()
plt.show()

plt.figure(8)
plt.plot([item[0] for item in trust_sphere_results[2]], [item[3] for item in trust_sphere_results[2]], label='Trust-Region')
plt.xscale('log')
plt.gca().invert_xaxis()
plt.title('Rosenbrock starting at (5,5)')
plt.xlabel('Log of absolute error')
plt.ylabel('# of Hessian Evaluations')
plt.legend()
plt.show()
