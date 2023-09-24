#!~/Python/env/bin/python3

#-------------------------------------------------------------
# plot_loss: plot losses within sub-optimizations
def plot_loss(hist, title):
    plt.figure()
    plt.subplot(121)
    plt.plot(hist['loss'], color='blue')
    plt.title("Loss")
    plt.subplot(122)
    plt.plot(hist['losspen'], color='purple')
    plt.title("Loss + Penalty")
    plt.show()
    
    return None

#----------------------------------------------------------------------------------
# functions to check if step-size is appropriate.
def Q_L_xi(w, y, X, Z, G, Lbar):
    diff = w - y
    return assoc_loss(X=X, Z=Z, G=G, Xi=y, type='view_sub') + torch.trace(diff.t().matmul(y.grad)) + (Lbar/2.0)*(torch.pow(diff, 2).sum())
    
def Q_L_g(w, y, X, Z, Xi, Lbar):
    diff = w - y
    return assoc_loss(X=X, Z=Z, G=y, Xi=Xi, type='view') +  torch.trace(diff.t().matmul(y.grad)) + (Lbar/2.0)*(torch.pow(diff, 2).sum())
    
def Q_L_z(w, y, Y, X, G, Xi, theta_dict, Lbar):
    diff = w - y
    return assoc_loss(X=X, Z=y, G=G, Xi=Xi, type='sub') + pred_loss(Y=Y, Z=y, theta_dict=theta_dict, single=True) + torch.trace(diff.t().matmul(y.grad)) + (Lbar/2.0)*(torch.pow(diff, 2).sum())

def Q_L_theta(w_t, w_b, y_t, y_b, Y, Z, tau, Lbar):
    diff = torch.cat((w_b, w_t)) - torch.cat((y_b, y_t))
    return pred_loss(Y=Y, Z=Z, theta_dict={'theta':y_t, 'beta':y_b, 'tau':tau}) + torch.trace(diff.t().matmul(torch.cat((y_b.grad, y_t.grad)))) + (Lbar/2.0)*(torch.pow(diff, 2).sum())
#----------------------------------------------------------------------------------


#----------------------------------------------------------------------------------
# opt_xi: optimize Xi^d,s using Adagrad in PyTorch
def opt_xi(X, Z, G, Xi, lam_xi, max_iter = 5000, update_thresh = 10**-5, verbose=False):
    
    xi_opt = Xi.clone().detach()
    
    hist = {'loss': [],
            'losspen': []}
    
    old_loss = 1
    old_losspen = 1
    hit_max = True
    
    xi_opt.requires_grad = True
    optimizer = torch.optim.Adagrad(params=[xi_opt])
    
    for c in range(max_iter):
    
        # Zero gradients
        optimizer.zero_grad()
        # Calculate current gradient
        xi_obj = assoc_loss(X=X, Z=Z, G=G, Xi=xi_opt, type = 'view_sub') + lam_xi*torch.pow(xi_opt,  2).sum(dim = 1).sqrt().sum()
        xi_obj.backward()
        # update estimates
        optimizer.step()
            
        # check for convergence
        with torch.no_grad():
        
            cur_loss = assoc_loss(X=X, Z=Z, G=G, Xi=xi_opt, type = 'view_sub')
            cur_losspen =  cur_loss + lam_xi*torch.pow(xi_opt,  2).sum(dim = 1).sqrt().sum()
            hist['loss'].append(cur_loss)
            hist['losspen'].append(cur_losspen)
            relloss = abs(cur_loss - old_loss)/old_loss
            rellosspen = abs(cur_losspen - old_losspen)/old_losspen

            if verbose and (c % 10 == 0):
                print("Xi: c =", c, "; loss =", cur_loss, "; losspen", cur_losspen, "; rellosspen =", rellosspen)
            
            if c > 1  and rellosspen < update_thresh:
                print("Xi converged in", c, "iterations.")
                message = 'Converged'
                hit_max = False
                break

        # update
        with torch.no_grad():
            old_loss = cur_loss
            old_losspen = cur_losspen
    
    if hit_max:
        print("Xi failed to converge in", c, "iterations.")
        message = 'MAX ITERS'
    
    if verbose:
        plot_loss(hist, 'Xi Loss')
    
    return {'val': xi_opt,
            'message': message
           }

#----------------------------------------------------------------------------------
# opt_g: optimize G^d using Adagrad in PyTorch
def opt_g(X, Z, G, Xi, lam_g, max_iter = 5000, update_thresh = 10**-5, verbose = False):
    
    g_opt = G.clone().detach()
    
    hist = {'loss': [],
            'losspen': []}
    
    old_loss = 1
    old_losspen = 1
    hit_max = True
    
    g_opt.requires_grad = True
    optimizer = torch.optim.Adagrad(params=[g_opt])
    
    for c in range(max_iter):
    
        # Zero gradients
        optimizer.zero_grad()
        # Calculate current gradient
        g_obj = assoc_loss(X=X, Z=Z, G=g_opt, Xi=Xi, type = 'view') + lam_g*torch.pow(g_opt,  2).sum(dim = 1).sqrt().sum()
        g_obj.backward()
        # update estimates
        optimizer.step()
            
        # check for convergence
        with torch.no_grad():
        
            cur_loss = assoc_loss(X=X, Z=Z, G=g_opt, Xi=Xi, type = 'view')
            cur_losspen = cur_loss + lam_g*torch.pow(g_opt,  2).sum(dim = 1).sqrt().sum()
            hist['loss'].append(cur_loss)
            hist['losspen'].append(cur_losspen)
            relloss = abs(cur_loss - old_loss)/old_loss
            rellosspen = abs(cur_losspen - old_losspen)/old_losspen

            if verbose and (c % 10 == 0):
                print("G: c =", c, "; loss =", cur_loss, "; losspen =", cur_losspen, "; rellosspen =", rellosspen)
            
            if c > 1  and rellosspen < update_thresh:
                print("G converged in", c, "iterations.")
                message = 'Converged'
                hit_max = False
                break

        # update
        with torch.no_grad():
            old_loss = cur_loss
            old_losspen = cur_losspen
    
    if hit_max:
        print("G failed to converge in", c, "iterations.")
        message = 'MAX ITERS'
    
    if verbose:
        plot_loss(hist, 'G Loss')
    
    return {'val': g_opt,
            'message': message
           }
           
#----------------------------------------------------------------------------------
# fista_z: optimize Z^s using FISTA with backtracking
def fista_z(Y, X, Z, G, Xi, theta_dict, max_iter = 500, update_thresh = 10**-5, verbose = False, L0=1, eta=2, max_iter_inner=30):
    
    x0 = Z.clone().detach()
    y_opt = Z.clone().detach()
    
    hist = {'loss':[], 'losspen': []} # losspen not used but included to prevent key error
    hist['loss'].append(assoc_loss(X=X, Z=y_opt, G=G, Xi=Xi, type = 'sub') + pred_loss(Y=Y, Z=y_opt, theta_dict=theta_dict, single=True))

    L = L0
    t_old = 1
    
    hit_max = True
    
    # i starts at 1 now; so i = k
    for c in range(1, max_iter):
        
        # Calculate the gradient at current estimate using PyTorch
        y_opt.requires_grad = True
        y_obj = assoc_loss(X=X, Z=y_opt, G=G, Xi=Xi, type = 'sub') + pred_loss(Y=Y, Z=y_opt, theta_dict=theta_dict, single=True)
        y_obj.backward()
        
        hit_max_inner = True
        
        # Find appropriate step size - backtracking
        for j in range(max_iter_inner):
            Lbar = L*(eta**j)

            with torch.no_grad():

                xk = y_opt - (1.0/Lbar)*y_opt.grad
                
                F_val = assoc_loss(X=X, Z=xk, G=G, Xi=Xi, type = 'sub') + pred_loss(Y=Y, Z=xk, theta_dict=theta_dict, single=True)
                Q_val = Q_L_z(w=xk, y=y_opt, Y=Y, X=X, G=G, Xi=Xi, Lbar=Lbar, theta_dict=theta_dict)
                
                if F_val <= Q_val:
                    L = Lbar
                    hit_max_inner = False
                    break
        
        if hit_max_inner:
            print("Learning rate issue; returning previous estimate")
            return {'val': x0,
                    'message': "L satisfying condition not found"
                    }
        
        # check for convergence
        with torch.no_grad():
            
            hist['loss'].append(assoc_loss(X=X, Z=xk, G=G, Xi=Xi, type = 'sub') + pred_loss(Y=Y, Z=xk, theta_dict=theta_dict, single=True))
            relloss = abs(hist['loss'][c-1] - hist['loss'][c])/hist['loss'][c-1]
            max_diff = torch.max(torch.abs(xk - x0))
            
            if verbose:
                print("Z: c =", c, "; loss =", hist['loss'][c], "; relloss =", relloss, "; max diff =", max_diff)
            
            if c > 1 and relloss < update_thresh:
                print("Z converged in", c, "iterations.")
                message = 'Converged'
                hit_max = False
                break
            
        # update
        with torch.no_grad():
            t_new = (1 + math.sqrt(1 + 4*t_old**2))/2.0
            y_opt = xk + ((t_old - 1)/t_new)*(xk - x0)
            
            x0 = xk
            t_old = t_new
    
    if hit_max:
        message = 'MAX ITERS'
        print("Z failed to converge in", c, "iterations.")
    
    if verbose:
        plot_loss(hist, 'Z Loss')
        
    return {'val': xk,
            'message': message
           }

#----------------------------------------------------------------------------------
# ista_theta: optimize theta/beta_0 using ISTA with backtracking
def ista_theta(Y, Z, theta_dict, max_iter = 500, update_thresh = 10**-5, verbose = False, L0=1, eta=2, max_iter_inner=30):
    
    t_opt = theta_dict['theta'].clone().detach()
    b_opt = theta_dict['beta'].clone().detach()
    
    hist = {'loss':[], 'losspen': []} # losspen not used but included to prevent key error
    
    hist['loss'].append(pred_loss(Y=Y, Z=Z, theta_dict={'theta':t_opt, 'beta':b_opt, 'tau':theta_dict['tau']}))
    
    L = L0
    hit_max = True
    
    # i starts at 1 now; so i = k
    for c in range(1, max_iter):
        
        # Calculate gradient at current estimate
        t_opt.requires_grad = True
        b_opt.requires_grad = True
        obj = pred_loss(Y=Y, Z=Z, theta_dict={'theta':t_opt, 'beta':b_opt, 'tau':theta_dict['tau']})
        obj.backward()
        
        hit_max_inner = True
        
        # Find appropriate step size - backtracking
        for j in range(max_iter_inner):
            Lbar = L*(eta**j)

            with torch.no_grad():
                tk = t_opt - (1.0/Lbar)*t_opt.grad
                bk = b_opt - (1.0/Lbar)*b_opt.grad
                
                F_val = pred_loss(Y=Y, Z=Z, theta_dict={'theta':tk, 'beta':bk, 'tau':theta_dict['tau']})
                Q_val = Q_L_theta(w_t=tk, w_b=bk, y_t=t_opt, y_b=b_opt, Y=Y, Z=Z, tau=theta_dict['tau'], Lbar=Lbar)
                
                if F_val <= Q_val:
                    L = Lbar
                    hit_max_inner = False
                    break
        
        if hit_max_inner:
            print("Learning rate issue; returning previous estimate")
            return {'val_theta': t_opt,
                    'val_beta': b_opt,
                    'message': "L satisfying condition not found"
                    }
        
        with torch.no_grad():
            # check for convergence
            hist['loss'].append(pred_loss(Y=Y, Z=Z, theta_dict={'theta':tk, 'beta':bk, 'tau':theta_dict['tau']}))
            
            relloss = abs(hist['loss'][c-1] - hist['loss'][c])/hist['loss'][c-1]
            max_diff = torch.max(torch.abs(tk - t_opt).max(), torch.abs(bk - b_opt).max())
            
            if verbose:
                print("theta: c =", c, "; loss =", hist['loss'][c], "; relloss =", relloss, "; max diff =", max_diff)
            
            if c > 1 and (relloss < update_thresh):
                print("Theta converged in", c, "iterations.")
                message = 'Converged'
                hit_max = False
                break
            
            # update
            t_opt = tk
            b_opt = bk
   
    if hit_max:
        print("Theta failed to converge in", c, "iterations.")
        message = 'MAX ITERS'
    
    if verbose:
        plot_loss(hist, 'Theta/Beta Loss')
   
    return {'val': {'theta': tk.clone().detach(), 'beta': bk.clone().detach(), 'tau': theta_dict['tau']},
            'message': message
           }

