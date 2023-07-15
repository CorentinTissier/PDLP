import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import scipy.sparse as sp

def weighted_norm(vec: np.array, weights: np.array):
    sum = 0.0
    for i,el in enumerate(vec):
        sum += weights[i] * el * el
        result = np.sqrt(sum)
    try: 
        result=result.flatten()[0]
    except:
        pass
    
    return result

class BoundConstrainedTrustRegionResult:
    def __init__(self, x: np.ndarray, f: float):
        self.x = x
        self.f = f


def solve_bound_constrained_trust_region(
    center_point: np.ndarray,
    objective_vector: np.ndarray,
    variable_lower_bounds: np.array,
    variable_upper_bounds: np.array,
    norm_weights: np.ndarray,
    target_radius: float,
    
):
    
    #change the shape of the arguments to have 1d array if necessary
    for param in [center_point,objective_vector,variable_lower_bounds,variable_upper_bounds,norm_weights]:
        if len(param.shape)==2:
            param.resize(len(param))


    assert 0.0 <= target_radius < np.inf
    if target_radius == 0.0 or np.linalg.norm(objective_vector, 2) == 0.0:
        return (np.copy(center_point), 0.0)

    direction = np.zeros(center_point.shape[0])
    threshold = np.zeros(center_point.shape[0])
    for idx in range(center_point.shape[0]):
        if (
            center_point[idx] >= variable_upper_bounds[idx]
            and objective_vector[idx] <= 0
        ):
            #z-lambda * g will stay greater than u no matter the positive lambda, so no need to move in that direction as 
            # clip(z - lambda * g,l,u) will always be u
            continue
        if (
            center_point[idx] <= variable_lower_bounds[idx]
            and objective_vector[idx] >= 0 
            # Similarly clip(z - lambda * g,l,u) will always be l for such index
        ):
            continue
        direction[idx] = -objective_vector[idx] / norm_weights[idx]
        if direction[idx] > 0:
            threshold[idx] = (
                variable_upper_bounds[idx] - center_point[idx]
            ) / direction[idx]
        elif direction[idx] < 0:
            threshold[idx] = (
                variable_lower_bounds[idx] - center_point[idx]
            ) / direction[idx]
        else:
            # Variable doesn't move.  Rather than an infinite threshold, or a NaN if
            # the corresponding bound was infinite as well, treat it as fixed, which
            # is effectively equivalent.
            #(g=0 or lambda =0 is the same here)
            threshold[idx] = 0.0

    # The weighted radius squared of the indices discarded because they are below
    # the threshold.
    low_radius_sq = 0.0
    # The weighted norm squared of the objective coefficients of the indices
    # discarded because they are above the threshold.
    high_radius_sq = 0.0

    indices = np.arange(center_point.shape[0])
    # Infinite thresholds can combine with zeros to create NaNs.  To avoid this,
    # handle indices with infinite thresholds separately.
    infinite_indices = np.where(np.isinf(threshold))[0]
    high_radius_sq += np.sum(
        direction[infinite_indices] ** 2 * norm_weights[infinite_indices]
    )
    indices = np.delete(indices, infinite_indices) #non infinite indices
    
    while len(indices) > 0:
        test_threshold = np.median(threshold[indices])
        test_point = np.clip(
            center_point[indices] + test_threshold * direction[indices],
            variable_lower_bounds[indices],
            variable_upper_bounds[indices])
        #||z(lambda)-z|| restricted to the indices inside the threshold
        test_radius = weighted_norm(
            test_point - center_point[indices], norm_weights[indices])
        if (low_radius_sq + test_radius**2 +
                test_threshold**2 * high_radius_sq >= target_radius**2):
            # test_threshold is too high.  Discard indices greater than it.
            discard_indices = list(filter(
                lambda i: threshold[i] >= test_threshold, indices))
            high_radius_sq += np.power(
                weighted_norm(direction[discard_indices],
                              norm_weights[discard_indices]), 2)
            indices = list(filter(
                lambda i: threshold[i] < test_threshold, indices))
        else:
            # test_threshold is too low.  Discard indices less than it.
            discard_indices = list(filter(
                lambda i: threshold[i] <= test_threshold, indices))
            discard_point = np.clip(
                center_point[discard_indices] +
                test_threshold * direction[discard_indices],
                variable_lower_bounds[discard_indices],
                variable_upper_bounds[discard_indices])
            low_radius_sq += weighted_norm(discard_point - center_point[discard_indices],norm_weights[discard_indices])**2
            indices = list(filter(
                lambda i: threshold[i] > test_threshold, indices))
    
    # target_threshold is the solution of
    # low_radius_sq + target_threshold^2 * high_radius_sq = target_radius^2.
    if high_radius_sq <= 0.0:
        # Special case: high_radius_sq = 0.0, means all bounds hit before reaching
        # target radius.
        target_threshold = np.max(threshold)
    else:
        target_threshold = np.sqrt((target_radius**2 - low_radius_sq) / high_radius_sq)
 
    candidate_point = np.clip(center_point + target_threshold * direction,
                              variable_lower_bounds,
                              variable_upper_bounds)
    return (candidate_point, np.dot(objective_vector, candidate_point - center_point))


def ComputePrimalGradient( y: np.array, c :np.array, K: np.array):
    return c-K.T.dot(y)

def ComputeDualGradient(x: np.array, K: np.array, q: np.array):
    return q-K.dot(x)

def Lagrangian(x: np.array, y: np.array, c :np.array, K: np.array, q: np.array):
    '''
         L(x,y)=ctx − ytKx + qty 
    '''
    L=c.T.dot(x) - np.dot(y.T,K.dot(x)) + q.T.dot(y) -14
    return np.ndarray.flatten(L)[0]

def bound_optimal_objective(c: np.array, G: np.array, A:np.array, b:np.array, l:np.array, u:np.array,
                            x: np.array, y: np.array, norm_weights: np.array, r: float,max_norm=False):
    #Defining the saddle point problem variables
    K=sp.vstack([G,A])
    q=np.vstack([h,b])
    z=np.vstack([x,y])
    
    #Defining the dimensions of the problem
    m1=G.shape[0]
    m2=A.shape[0]
    dim=c.shape[0]
    
    primal_norm_weights=norm_weights[:dim]
    dual_norm_weights=norm_weights[dim:]
    
    lagrangian_value=Lagrangian(x, y, c, K, q)
    
    #Computing the gradients
    primal_gradient = ComputePrimalGradient(y,c,K)
    dual_gradient= ComputeDualGradient(x,K,q)
    z_gradient = np.vstack([primal_gradient, -dual_gradient])
    
    #Defining the bound
    dual_variable_lower_bounds = -np.inf * np.ones(len(y)).reshape(-1,1)
    dual_variable_upper_bounds = np.inf * np.ones(len(y)).reshape(-1,1)
    dual_variable_lower_bounds[:m1] = [0.0]
    z_lower_bound = np.vstack([l, dual_variable_lower_bounds])
    z_upper_bound = np.vstack([u, dual_variable_upper_bounds])
   
   
    if not max_norm:
        # Compute the lower bound
        sol,_ = solve_bound_constrained_trust_region(z, z_gradient, z_lower_bound, z_upper_bound, norm_weights, r)
        primal_tr_solution = sol[:dim].reshape(-1,1)
        dual_tr_solution = sol[dim:].reshape(-1,1)
        
        return (lagrangian_value, 
            lagrangian_value + np.dot(primal_tr_solution.T - primal_solution.T, primal_gradient).flatten()[0],
           lagrangian_value + np.dot(dual_tr_solution.T - dual_solution.T, dual_gradient).flatten()[0],
           primal_tr_solution,
           dual_tr_solution)
    
    if max_norm:
        # We can split the max norm into two cases.
        # Compute the primal part
        primal_result = solve_bound_constrained_trust_region(
          x,
          primal_gradient,
          l,
          u,
          primal_norm_weights,
          r)

        # Compute the dual part
        dual_result = solve_bound_constrained_trust_region(
          y,
          -dual_gradient,
          dual_variable_lower_bounds,
          dual_variable_upper_bounds,
          dual_norm_weights,
          r)
        
        
        return (lagrangian_value,
          # lower bound
          lagrangian_value + primal_result[1],
          # upper bound
          lagrangian_value - dual_result[1],
          # primal solution
          primal_result[0],
          # dual solution
          dual_result[0])
    
def bound_optimal_objective(c: np.array, G: np.array, A:np.array, b:np.array,h:np.array, l:np.array, u:np.array,
                            x: np.array, y: np.array, norm_weights: np.array, r: float,max_norm=False):
    #Defining the saddle point problem variables
    K=sp.vstack([G,A])
    q=np.vstack([h,b])
    z=np.vstack([x,y])
    
    #Defining the dimensions of the problem
    m1=G.shape[0]
    m2=A.shape[0]
    dim=c.shape[0]
    
    primal_norm_weights=norm_weights[:dim]
    dual_norm_weights=norm_weights[dim:]
    
    lagrangian_value=Lagrangian(x, y, c, K, q)
    
    #Computing the gradients
    primal_gradient = ComputePrimalGradient(y,c,K)
    dual_gradient= ComputeDualGradient(x,K,q)
    z_gradient = np.vstack([primal_gradient, -dual_gradient])
    
    #Defining the bound
    dual_variable_lower_bounds = -np.inf * np.ones(len(y)).reshape(-1,1)
    dual_variable_upper_bounds = np.inf * np.ones(len(y)).reshape(-1,1)
    dual_variable_lower_bounds[:m1] = [0.0]
    z_lower_bound = np.vstack([l, dual_variable_lower_bounds])
    z_upper_bound = np.vstack([u, dual_variable_upper_bounds])
    
   
    if not max_norm:
        # Compute the lower bound
        sol,_ = solve_bound_constrained_trust_region(z, z_gradient, z_lower_bound, z_upper_bound, norm_weights, r)
        primal_tr_solution = sol[:dim].reshape(-1,1)
        dual_tr_solution = sol[dim:].reshape(-1,1)
        
        return (lagrangian_value, 
            lagrangian_value + np.dot(primal_tr_solution.T - x.T, primal_gradient).flatten()[0],
           lagrangian_value + np.dot(dual_tr_solution.T - y.T, dual_gradient).flatten()[0],
           primal_tr_solution,
           dual_tr_solution)
    
    if max_norm:
        # We can split the max norm into two cases.
        # Compute the primal part
        primal_result = solve_bound_constrained_trust_region(
          x,
          primal_gradient,
          l,
          u,
          primal_norm_weights,
          r)

        # Compute the dual part
        dual_result = solve_bound_constrained_trust_region(
          y,
          -dual_gradient,
          dual_variable_lower_bounds,
          dual_variable_upper_bounds,
          dual_norm_weights,
          r)
        
        
        return (lagrangian_value,
          # lower bound
          lagrangian_value + primal_result[1],
          # upper bound
          lagrangian_value - dual_result[1],
          # primal solution
          primal_result[0],
          # dual solution
          dual_result[0])



def solve_normalised_duality_gap(c: np.array, G: np.array, A:np.array, b:np.array,h:np.array, l:np.array, u:np.array,
                            x: np.array, y: np.array, norm_weights: np.array, r: float):
    #Defining the saddle point problem variables
    K=sp.vstack([G,A])
    q=np.vstack([h,b])
    z=np.vstack([x,y])
    
    #Defining the dimensions of the problem
    m1=G.shape[0]
    m2=A.shape[0]
    dim=c.shape[0]
    
    primal_norm_weights=norm_weights[:dim]
    dual_norm_weights=norm_weights[dim:]
    
    
    #Computing the gradients
    primal_gradient = ComputePrimalGradient(y,c,K)
    dual_gradient= ComputeDualGradient(x,K,q)
    z_gradient = np.vstack([primal_gradient, -dual_gradient])
    
    #Defining the bound
    dual_variable_lower_bounds = -np.inf * np.ones(len(y)).reshape(-1,1)
    dual_variable_upper_bounds = np.inf * np.ones(len(y)).reshape(-1,1)
    dual_variable_lower_bounds[:m1] = [0.0]
    z_lower_bound = np.vstack([l, dual_variable_lower_bounds])
    z_upper_bound = np.vstack([u, dual_variable_upper_bounds])
    
   
    
    sol,_ = solve_bound_constrained_trust_region(z, z_gradient, z_lower_bound, z_upper_bound, norm_weights, r)
    primal_tr_solution = sol[:dim].reshape(-1,1)
    dual_tr_solution = sol[dim:].reshape(-1,1)

    return (primal_tr_solution, dual_tr_solution)   
    


if __name__=="__main__":
    pass


    
    
  