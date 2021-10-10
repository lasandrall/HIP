# helper_functions.py
# Contains functions not meant to be called by the user.
# Author: Jessica Butts and Sandra Safo
# Date: June 24, 2021

#----------------------------------------------------------------------------------
# class_matrix: takes an  nx1 vector of integer classes and converts to an nxm indicator matrix where m is the number of unique classes
def class_matrix(Y, S):
    classes = [list() for s in range(S)]
    column = [list() for s in range(S)]
    if S > 1:
        for s in range(S):
            classes[s], column[s] = torch.unique(Y[s], return_inverse=True)
        nclass  = max(len(classes[s]) for s in range(S))
        indicator_mat = [torch.zeros((Y[s].shape[0], nclass)) for s in range(S)]
        for s in range(S):
            for i in range(len(Y[s])):
                indicator_mat[s][i][column[s][i]] = 1
    else:
        classes, column = torch.unique(Y, return_inverse=True)
        nclass = len(classes)
        indicator_mat = torch.zeros(Y.shape[0], nclass)
        for i in range(len(Y)):
            indicator_mat[i][column[i]] = 1

    return indicator_mat
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# calc_probs: calculates the probability of being in each of m classes;
#   returns a matrix of probabilities of being in each class for each observation
def calc_probs(W, S):
    if S > 1:
        probs = [list() for s in range(S)]
        for s in range(S):
            W_exp = torch.exp(W[s])
            row_sums = torch.sum(W_exp, dim = 1, keepdim=True)
            probs[s] = torch.true_divide(W_exp, row_sums)
    else:
        W_exp = torch.exp(W)
        row_sums = torch.sum(W_exp, dim = 1, keepdim=True)
        probs = torch.true_divide(W_exp, row_sums)
    return probs
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# nonZero: returns the number of non_zero rows in a matrix or list of matrices
def nonZero(X):
    count = 0
    Cpy = copy.deepcopy(X)
    if type(Cpy) == list:
        for d in range(len(Cpy)):
            for s in range(len(Cpy[d])):
                Cpy[d][s] = torch.abs(Cpy[d][s])
                row_sums = torch.sum(Cpy[d][s], 1)
                count += sum(row_sums!=0)
    else:
        row_sums = torch.sum(torch.abs(Cpy), 1)
        count = sum(row_sums != 0)
    return count
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# Zero: returns True if all entries in a matrix are 0 and False otherwise
def Zero(X):
    Cpy = copy.deepcopy(X)
    Cpy = torch.abs(Cpy)
    row_sums = torch.sum(Cpy, 1)
    if (sum(row_sums == 0) == Cpy.shape[0]):
        return True
    else:
        return False
#----------------------------------------------------------------------------------
        
#----------------------------------------------------------------------------------
# Projl21ball: calculate the projection onto the L21 Ball
def Projl21ball(U, mylambda):
    if mylambda==0:
        X=U
    else:
        pd,K=U.size()
        l2X=torch.norm(U,p=2,dim=1)
        ll=mylambda/l2X
        ll3=torch.ones(pd,1)- torch.unsqueeze(ll,1)
        max_val=torch.max(ll3, torch.zeros(pd,1))
        X=max_val.repeat(1,K) * U
    return X
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# normF2: Calculate the squared Frobenius norm of a matrix
def normF2(A):
    Asq=A *A
    Asq=torch.unsqueeze(Asq,1)
    AF=torch.sum(Asq)
    return AF
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# FxLoss: Loss function for association term when solving for Xi
def FxLoss(Xdata,Z,G,myXi,mylambda):
    penaltyTerm=mylambda*torch.sum((torch.norm(myXi,p=2,dim=1)))
    myloss2=torch.norm(Xdata-torch.matmul(Z,torch.t(G * myXi)),'fro')**2 + penaltyTerm
    return myloss2
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# FxLossG: Loss function for association term when solving for G
def FxLossG(Xdata,Z,G,myXi,mylambda):
    S=len(Z)
    myloss2=0
    for s in range(S):
        penaltyTerm=mylambda*torch.sum((torch.norm(myXi[s],p=2,dim=1)))
        myloss2=myloss2+ torch.norm(Xdata[s]-torch.matmul(Z[s],torch.t(G * myXi[s])),'fro')**2 + penaltyTerm
    return myloss2
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# fxfunc: Loss Function for association term without penalty solving for Xi
def fxfunc(Xdata,Z,G,myXi):
    myloss=torch.norm(Xdata-torch.matmul(Z,torch.t(G * myXi)),'fro')**2
    return myloss
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# fxfuncG: Loss Function for association term without penalty solving for G
def fxfuncG(Xdata,Z,G,myXi):
    S=len(Z)
    myloss=0
    for s in range(S):
        myloss=myloss+torch.norm(Xdata[s]-torch.matmul(Z[s],torch.t(G * myXi[s])),'fro')**2
    return myloss
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# z_loss: Loss function for association term used in optimizing Z with multiclass outcome
def z_loss(X, B, Z, D):
    res = 0
    for d in range(D):
        res += torch.norm(X[d] - torch.matmul(Z, torch.t(B[d])), 'fro')**2
    return res
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# class_loss: Calculates the loss for prediction term with multiclass outcome
def class_loss(Indicator_mat, Prob_mat, S):
    loss = 0
    if S > 1:
        for s in range(S):
            loss += torch.sum(Indicator_mat[s]*torch.log(Prob_mat[s]))
    else:
        loss += torch.sum(Indicator_mat*torch.log(Prob_mat))
    return -1*loss
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# OptimizeXi: used to perform optimization of Xi^d,s when adding penalty
def OptimizeXi(Xdata,myZ,myG,mylambda,max_iter=10000,update_thresh=10**-5):
    L=torch.norm( torch.matmul(myZ,torch.t(myZ) ),'fro')* torch.norm(myG * myG,'fro')
    #print(L)
    Linv=1/L
    #print(Linv)

    pd,K=myG.size()
    myXi=torch.zeros(pd,K, dtype=torch.double)

    x_old=myXi #initial
    y_old=x_old
    vold=torch.zeros_like(y_old) #zeros with same size as y_old
    mold=torch.zeros_like(y_old)
    t_old=1
    myiter=0
    ObjHist=torch.zeros(max_iter+1)
    ObjHist[0]=FxLoss(Xdata,myZ,myG,x_old,mylambda)
    while myiter< max_iter:
        myiter=myiter+1
        y_oldP=y_old.clone()
        #calculate gradient of f(y)
        y_old.requires_grad_(True)
        y_old2=fxfunc(Xdata,myZ,myG,y_old)
        y_old2.backward() #computes gradient

        #form projection function
        #equation pL(y) on page 189 of Breck FISTA paper
        with torch.no_grad():
            U=y_oldP - y_old.grad*Linv
            mylambda2=mylambda*Linv
            x_new=Projl21ball(U, mylambda2) #projection
            t_new=0.5 + 0.5*math.sqrt(1 + 4.0 *t_old *t_old)
            y_new=x_new + ((t_old -1 )/t_new) *(x_new- x_old)
            reldiff=torch.norm(x_new-x_old,'fro')/x_new.size(0)
            #print('reldiff at i',myiter, reldiff)
            if(reldiff<update_thresh):
                break

            #update
            x_old=x_new
            y_old=y_new
            t_old=t_new
            ObjHist[myiter]=FxLoss(Xdata,myZ,myG,x_new,mylambda)
        
    return x_new.detach()#,ObjHist.detach(), myiter
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# adagrad_xi: used to perform optimization of Xi^d,s when no penalty
def adagrad_xi(X, Z, G, lambda_xi):
    max_iter = 1000
    update_thresh = 10**-5
    obj_old = 1

    myXi_old = Variable(torch.rand(size=G.shape), requires_grad = True)
    
    X = torch.as_tensor(X, dtype=torch.double)
    Z = torch.as_tensor(Z, dtype=torch.double)
    G = torch.as_tensor(G, dtype=torch.double)

    optimizer = torch.optim.Adagrad(params = [myXi_old], lr = 1)

    for i in range(max_iter):
        optimizer.zero_grad()
        # Objective is the loss function we want to minimize
        obj = torch.norm(X - torch.mm(Z, torch.t(G*myXi_old)), 'fro')**2 + lambda_xi*torch.sum(torch.norm(myXi_old, p=2, dim = 1))
        obj.backward() # differentiates
        optimizer.step()
        #print("t = ", i, ", loss = ", obj)
        if abs(obj - obj_old)/obj_old < update_thresh:
            #print("Xi stopped at iteration", i)
            break
        obj_old = obj
        
    return myXi_old
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# OptimizeG2: used to perform optimization of G^d when adding penalty
def OptimizeG2(Xdata,myZ,myXi,mylambda,max_iter=10000,update_thresh=10**-5):
     myZ_stack = torch.cat(myZ)
     myXi_stack = torch.cat(myXi)
     L=torch.norm( torch.matmul(myZ_stack,torch.t(myZ_stack) ),'fro')* torch.norm(myXi_stack * myXi_stack,'fro')
     Linv=1/L
     
     pd,K=myXi[0].size()
     myG=torch.zeros(pd,K)
     
     x_old=myG #initial
     y_old=x_old
     t_old=1
     myiter=0
     ObjHist=torch.zeros(max_iter+1)
    # ObjHist[0]=FxLossG(Xdata,myZ,x_old,myXi,mylambda)
     while myiter< max_iter:
         myiter=myiter+1
         y_oldP=y_old.clone()
         #calculate gradient of f(y)
         y_old.requires_grad_(True)
         y_old2=fxfuncG(Xdata,myZ,y_old,myXi)
         y_old2.backward() #computes gradient

         #form projection function
         #equation pL(y) on page 189 of Breck FISTA paper
         with torch.no_grad():
             U=y_oldP - y_old.grad*Linv
             mylambda2=mylambda*Linv
             x_new=Projl21ball(U, mylambda2) #projection
             t_new=0.5 + 0.5*math.sqrt(1 + 4.0 *t_old *t_old)
             y_new=x_new + ((t_old -1 )/t_new) *(x_new- x_old)
             reldiff=torch.norm(x_new-x_old,'fro')/x_new.size(0)
             #print('reldiff at i',myiter, reldiff)
             if(reldiff<update_thresh):
                 break

             #update
             x_old=x_new
             y_old=y_new
             t_old=t_new
             #ObjHist[myiter]=FxLossG(Xdata,myZ,myG,x_new,mylambda)
     G_new=x_new
     return G_new.detach() #,ObjHist.detach(), myiter
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# adagrad_G: used to perform optimization of G^d when no penalty
def adagrad_G(X,Z,Xi,lambda_g):
    max_iter = 1000
    update_thresh = 10**-5
    obj_old = 1

    myG_old = Variable(torch.rand(Xi[0].shape), requires_grad = True)

    for s in range(S):
        X[s] = torch.as_tensor(X[s], dtype=torch.double)
        Z[s] = torch.as_tensor(Z[s], dtype=torch.double)
        Xi[s] = torch.as_tensor(Xi[s], dtype=torch.double)

    optimizer = torch.optim.Adagrad([myG_old], lr = 1)

    for j in range(max_iter):
        optimizer.zero_grad()
        obj = 0
        for s in range(S):
            obj += torch.norm(X[s] - torch.mm(Z[s], torch.t(myG_old*Xi[s])), 'fro')**2
        obj += lambda_g*torch.sum(torch.norm(myG_old, p=2, dim=1))
        obj.backward()
        optimizer.step()
        #print("t = ", j, ", loss = ", obj)
        if abs(obj - obj_old)/obj_old < update_thresh:
            #print("G stopped at iteration", j)
            break
        obj_old = obj
        
    return myG_old
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# solve_Z: Closed form solution to optimize Z when outcome is continuous
def solve_Z(X, B, Y, theta):
    B_tilde = torch.cat((torch.t(torch.cat(B)), theta), dim=1)
    xy = torch.cat((torch.cat(X, dim=1), Y), dim=1)
    return torch.t(torch.inverse(torch.mm(B_tilde, torch.t(B_tilde))) @ B_tilde @ torch.t(xy))
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# adagrad_Z: performs Z optimization when outcome is multiclass
def adagrad_Z(X, B, Y, theta):
    max_iter = 1000
    update_thresh = 10**-5
    obj_old = 1
    D = len(X)

    myZ_old = Variable(torch.rand(size=(X[0].shape[0], B[0].shape[1])).double(), requires_grad = True)

    optimizer = torch.optim.Adagrad([myZ_old], lr = 1)

    for m in range(max_iter):
        optimizer.zero_grad()
        obj = 0
        obj += z_loss(X, B, myZ_old, D) # Association
        obj -= class_loss(Indicator_mat = Y,
                          Prob_mat = calc_probs(torch.mm(myZ_old, theta), 1),
                          S = 1) # Prediction
        obj.backward()
        optimizer.step()
        with torch.no_grad():
            if abs(obj - obj_old)/obj_old < update_thresh:
                break
            obj_old = obj

    return myZ_old
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# adagrad_theta: performs theta optimization for multiclass outcome
def adagrad_theta(Y, Z):
    max_iter = 1000
    update_thresh = 10**-5
    obj_old = 1
                          
    Y_cat = torch.cat(Y)
    Z_cat = torch.cat(Z)
    
    old = torch.ones((Z_cat.shape[1], Y_cat.shape[1]))
                          
    theta_old = Variable(torch.rand(size=(Z_cat.shape[1], Y_cat.shape[1])).double(), requires_grad = True)
                          
    optimizer = torch.optim.Adagrad([theta_old], lr=1)
                          
    for j in range(max_iter):
        optimizer.zero_grad()
        obj = class_loss(Indicator_mat = Y_cat,
                         Prob_mat = calc_probs(torch.mm(Z_cat, theta_old), 1),
                         S = 1)
        obj.backward()
        optimizer.step()
        with torch.no_grad():
            reldiff = abs(obj-obj_old)/obj_old
            normdiff = torch.norm(theta_old - old, 'fro')**2/torch.norm(old, 'fro')**2
            if min(reldiff, normdiff) < update_thresh:
                break
                old = theta_old.clone()
                obj_old = obj
                          
    return theta_old.detach()
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# summary_B(B): give number of common and subgroup specific variables for each view
# Diagonals give the number of variables selected in that subgroup
# Off-diagonals give the number of variables that overlap between subgroup i and j
def summary_B(B):
    D = len(B)
    S = len(B[0])
    selected = [[torch.sum(torch.abs(B[d][s]), 1) != 0 for s in range(S)] for d in range(D)]
    common = [np.zeros((S,S)) for d in range(D)]
    for d in range(D):
        for s in range(S):
            for s2 in range(s, S):
                match=0
                for j in range(B[d][0].shape[0]):
                    match += selected[d][s][j] & selected[d][s2][j]
                common[d][s][s2] = match
    return common
#----------------------------------------------------------------------------------
