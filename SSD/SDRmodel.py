import numpy as np
import spams
import torch

class Model:
    def __init__(self):
        """
        Sparse Decompositional Regression (SDR) model training model

        Attributes:
        - X (numpy.ndarray): input matrix (m x N), feature dim: m, sample dim: N
        - L (numpy.ndarray): label vector (1 x N), regression target for targeted mode
        - D (numpy.ndarray): Dictionary matrix (m x h).
        - Z (numpy.ndarray): Sparse representation matrix (h x N).
        - W (numpy.ndarray): The encoder for sparse code (h x m).
        - G (numpy.ndarray): Diagonal matrix (h x h).
        - A (numpy.ndarray): Targeted-only, the bias (intercept) and regression coefficient vector (1 x (1 + h)).
        """
        self.X = None
        self.L = None
        self.D = None
        self.Z = None
        self.W = None
        self.G = None
        self.A = None

    def config(self, h, lambda_, nepoch, method, optmethod, tauw, initialization, beta, device='cpu'):
        """
        Parameters:
        h (int): number of atoms, dictionary elements 
        lambda_: sparsity, non-zero elements in the sparse code
        nepoch (int): number of iterations
        method: method for Dictionary Learning, the Dictionary Learning here is used for dictionary initialization
        optmethod: method for updating W, 
                optmethod=0 -> ProjW (gradient descent); optmehtod = 1 -> minFunc
        tauw: learning rate for W
        initialization: whethter to initialize W as D'
        beta: whethter to use targeted model, beta=0 -> non-targeted beta=1 -> targeted
        """
        self.h = h
        self.lambda_ = lambda_
        self.nepoch = nepoch
        self.method = method
        self.optmethod = optmethod
        self.tauw = tauw
        self.initialization = initialization
        self.beta = beta
        self.device = device
        
        self.trloss = np.zeros(nepoch)

        # # spams
        self.DLparam = {
            'mode': 2,
            'K': h, # learns a dictionary with h elements
            'lambda1': lambda_,
            'numThreads': 7, # number of threads
            'batchsize': 2500,
            'iter': 100,
        }

        if method == 0:
            self.param = {'L': lambda_, 'eps': 0.1, 'numThreads': -1}
        elif method == 1:
            self.param = {'lambda1': lambda_, 'lambda2': 0, 'numThreads': -1, 'mode': 2, 'pos': True}

        self.options = {
            'maxiter': 100,
            'disp': False
        }

    @torch.no_grad()
    def fit(self, X, L, D=None):
        """
        Train Sparse Decomposition Regression with the minimization objective: 
        || DZ - X ||_F + || IZ - G SIGMA(WX) ||_F + || AZ - (L-A0) ||_F 

        Parameters:
        X (numpy.ndarray): input matrix (m x N), feature dim: m, sample dim: N
        L (numpy.ndarray): label vector (1 x N), regression target for targeted mode

        Side Effects:
        - Modifies attributes: W, G, D, A, Z
        
        """
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = self.device
        X_cpu = np.asfortranarray(X)

        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.L = torch.tensor(L, dtype=torch.int32, device=device)
        self.L_rescale = self.L * self.beta
        self.A0_rescale = torch.tensor(np.random.rand(), dtype=torch.float32, device=device) * self.beta

        X = self.X
        m, N = X.shape
        FIX_D = D is not None  # Flag for whether to use the fixed dict mode

        torch.manual_seed(42)
        np.random.seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if not FIX_D:
            D_init = spams.trainDL(X_cpu, **self.DLparam)
            D0 = torch.tensor(D_init, dtype=torch.float32, device=device)
            eps = torch.finfo(torch.float32).tiny
            D = D0 @ torch.diag(1 / torch.sqrt(torch.sum(D0**2, dim=0)))
            if np.isnan(D.cpu().numpy()).any():
                D = D0 @ torch.diag(1 / (torch.sqrt(torch.sum(D0**2, dim=0)) + eps))

        K = D.shape[1]
        if self.initialization == 1:
            W = torch.cat([torch.rand(K, 1, device=device), D.T], dim=1)
        else:
            W = 0.1 * (-0.05 + 0.1 * torch.rand(K, m + 1, device=device))

        X_intercept = torch.cat([torch.ones(1, N, device=device), X], dim=0)

        G_old = torch.eye(D.shape[1], device=device)
        W_old = W
        Z = torch.tensor([]).to(device)
        A = torch.rand(1, K, device=device)

        for ite in range(self.nepoch):
            G = G_old
            W = W_old
            SigmaWX = torch.sigmoid(torch.matmul(W, X_intercept))

            if Z.size(0) != 0:
                tau_d = 0.99 * 2 / torch.norm(torch.matmul(Z, Z.T), p='fro')
                for _ in range(50):
                    if FIX_D:
                        if self.beta != 0:
                            A = ProjC2(A, Z, torch.cat([(self.L_rescale - self.A0_rescale * torch.ones_like(self.L_rescale)).view(1, -1)]), tau_d)
                            self.A0_rescale = torch.mean(self.A0_rescale - 0.1 * (self.A0_rescale + torch.matmul(A, Z) - self.L_rescale))
                    else:
                        if self.beta != 0:
                            DNew = ProjC2(torch.cat([D, A], dim=0), Z, torch.cat([X, (self.L_rescale - self.A0_rescale * torch.ones_like(self.L_rescale)).view(1, -1)]), tau_d)
                            D = DNew[:-1, :]
                            A = DNew[-1, :].reshape(1, K)
                            self.A0_rescale = torch.mean(self.A0_rescale - 0.1 * (self.A0_rescale + torch.matmul(A, Z) - self.L_rescale))
                            D = torch.abs(D)
                            D = D @ torch.diag(1 / torch.sqrt(torch.sum(D**2, dim=0)))
                        else:
                            D = ProjC2(D, Z, X, tau_d)
                            D = torch.abs(D)
                            D = D @ torch.diag(1 / torch.sqrt(torch.sum(D**2, dim=0)))
            
            if self.beta != 0:
                X_eq = torch.cat([X, torch.matmul(G, SigmaWX), (self.L_rescale - self.A0_rescale * torch.ones_like(self.L_rescale)).view(1, -1)], dim=0)
                D_eq = torch.cat([D, torch.eye(D.shape[1], device=device), A], dim=0)
            else:
                X_eq = torch.cat([X, torch.matmul(G, SigmaWX)], dim=0)
                D_eq = torch.cat([D, torch.eye(D.shape[1], device=device)], dim=0)

            D_eq = D_eq @ torch.diag(1 / torch.sqrt(torch.sum(D_eq**2, dim=0)))

            X_eq_cpu = np.asfortranarray(X_eq.detach().cpu().numpy())
            D_eq_cpu = np.asfortranarray(D_eq.detach().cpu().numpy())
            if self.method == 0:
                Z = spams.omp(X_eq_cpu, D_eq_cpu, **self.param).toarray()
                Z = torch.tensor(Z, dtype=torch.float32, device=device)
            elif self.method == 1:
                Z = spams.lasso(X_eq_cpu, D_eq_cpu, **self.param).toarray()
                Z = torch.tensor(Z, dtype=torch.float32, device=device)

            if self.beta != 0:
                Z = torch.diag(1 / torch.sqrt(torch.sum(torch.cat([D, torch.eye(D.shape[1], device=device), A], dim=0)**2, dim=0))) @ Z
            else:
                Z = torch.diag(1 / torch.sqrt(torch.sum(torch.cat([D, torch.eye(D.shape[1], device=device)], dim=0)**2, dim=0))) @ Z

            SigmaWX = torch.sigmoid(torch.matmul(W, X_intercept))
            eps = torch.finfo(torch.float32).tiny
            temp = torch.diag(1 / (torch.sum(SigmaWX**2, dim=1) + eps))
            G = torch.diag((temp @ SigmaWX) @ Z.T)
            G = torch.diag(G)

            err_old = torch.norm(torch.matmul(G, torch.sigmoid(torch.matmul(W_old, X_intercept))) - Z, p='fro') / torch.norm(Z, p='fro')
            W_old = W

            if self.optmethod == 0:
                ###########################################
                # TODO: ProjW (stochastic) gradient descent
                for _ in range(50):
                    W = self.ProjW(W, Z, X_intercept, self.tauw, G)
                W_new = W
                ###########################################
            elif self.optmethod == 1:
                theta = W_old.flatten().clone().detach().requires_grad_(True).to(device)
                optimizer = torch.optim.LBFGS([theta], max_iter=100, history_size=10, line_search_fn="strong_wolfe")
                
                def closure():
                    optimizer.zero_grad()
                    loss = object_fun_matrix_form(theta, Z, X_intercept, G)
                    loss.backward(retain_graph=True)
                    return loss

                optimizer.step(closure)
                W_new = theta.view(Z.shape[0], X_intercept.shape[0])

            print(f'old error: {torch.norm(torch.matmul(G, torch.sigmoid(torch.matmul(W_old, X_intercept))) - Z, p="fro")}')
            print(f'cur error: {torch.norm(torch.matmul(G, torch.sigmoid(torch.matmul(W_new, X_intercept))) - Z, p="fro")}')
            print(f'old - cur = {err_old - torch.norm(torch.matmul(G, torch.sigmoid(torch.matmul(W_new, X_intercept))) - Z, p="fro") / torch.norm(Z, p="fro")}')

            err_cur = torch.norm(torch.matmul(G, torch.sigmoid(torch.matmul(W_new, X_intercept))) - Z, p='fro') / torch.norm(Z, p='fro')
            if err_cur <= err_old:
                W_old = W_new
                err_old = err_cur
                print('decreasing')
            else:
                print('ascending')
                self.tauw *= 0.1

            self.trloss[ite] = torch.norm(X - torch.matmul(D, Z), p='fro')**2 + torch.norm(torch.matmul(G, torch.sigmoid(torch.matmul(W_old, X_intercept))) - Z, p='fro')**2
            X_approximation_error_ratio = torch.norm(X - torch.matmul(D, torch.matmul(G, torch.sigmoid(torch.matmul(W_old, X_intercept)))), p='fro') / torch.norm(X, p='fro')
            X_error_ratio = torch.norm(X - torch.matmul(D, Z), p='fro') / torch.norm(X, p='fro')

            self.A0 = torch.tensor([[self.A0_rescale]], dtype=torch.float32, device=device)

            if self.beta != 0:
                Z_intercept = torch.cat([torch.ones(1, N, device=device), Z], dim=0)
                L_hat = torch.matmul(torch.cat([self.A0, A], dim=1), Z_intercept)
                reg_mse = torch.mean((L_hat - self.L)**2)

                # L_hat2 = torch.matmul(torch.cat([self.A0, A], dim=1), torch.cat([torch.ones(1, N, device=device), torch.matmul(G, torch.sigmoid(torch.matmul(W_old, X_intercept)))], dim=0))
                # reg_mse2 = torch.mean((L_hat2 - self.L)**2)

                # L_hat3 = torch.matmul(torch.cat([self.A0, A], dim=1), torch.cat([torch.ones(1, N, device=device), sparsity_enforcement(torch.matmul(G, torch.sigmoid(torch.matmul(W_old, X_intercept))), self.lambda_)], dim=0))
                # reg_mse3 = torch.mean((L_hat3 - self.L)**2)

                # print(f'Ite {ite} Object error1: {self.trloss[ite].item()} Z approx err ratio: {X_approximation_error_ratio.item()} Reg MSE: {reg_mse.item()} Reg MSE2: {reg_mse2.item()} Reg MSE3: {reg_mse3.item()}')
                print(f'Ite {ite} Object error: {self.trloss[ite].item()} Za err ratio: {X_approximation_error_ratio.item()} Z err ratio: {X_error_ratio.item()} Reg MSE: {reg_mse.item()}')
            else:
                print(f'Ite {ite} Object error: {self.trloss[ite].item()} Za err ratio: {X_approximation_error_ratio.item()} Z err ratio: {X_error_ratio.item()}')

            G_old = G

        self.X_approximation_error_ratio = X_approximation_error_ratio
        self.X_error_ratio = X_error_ratio
        self.W = W_old
        self.G = G
        self.D = D
        self.Z = Z
        self.Z_approximation = torch.matmul(G, torch.sigmoid(torch.matmul(W_old, X_intercept)))
        if self.beta != 0:
            self.A = torch.cat([self.A0, A], dim=1)
            self.reg_mse = reg_mse



def ProjC2(D, Z, X, tau_d):
    """
    Update dictionary D based on either gradient descent or mini-batch gradient descent.
    
    Parameters:
    D (numpy.ndarray): Dictionary matrix.
    Z (numpy.ndarray): Sparse code matrix, each column is a sparse code.
    X (numpy.ndarray): Feature matrix, each column is a feature.
    taud (float): Learning rate.
    
    Returns:
    numpy.ndarray: Updated dictionary matrix D.
    """
    # if X.shape[1] < 1000:
    #     # Gradient descent
    #     D = D - tau_d * (D @ Z - X) @ Z.T
    # else:
    #     # Mini-batch gradient descent
    #     S = X.shape[1]
    #     ##########################################################
    #     # TODO:r = randperm(S);for i = 1:floor(S/1000)
    #     r = np.random.permutation(S)
    #     for i in range(0, S, 1000):
    #         X1 = X[:, r[i:i+1000]]
    #         Z1 = Z[:, r[i:i+1000]]
    #         D = D - tau_d * (D @ Z1 - X1) @ Z1.T
    #     ##########################################################
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    D = D - tau_d * (torch.matmul(D, Z) - X) @ Z.T
    return D

def ProjW(self, W, Z, X_intercept, tauw, G):
    # Define the gradient projection update for W (Not implemented here) Y
    pass

def object_fun_matrix_form(theta, Z, X, G):

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    W = theta.view(Z.shape[0], X.shape[0])
    SigmaWX = torch.sigmoid(torch.matmul(W, X))

    dW = 2 * (torch.matmul(G, SigmaWX) - Z) * SigmaWX * (1 - SigmaWX)
    dW = torch.matmul(dW, X.T) + 2 * 0.0001 * W

    cost = torch.sum((torch.matmul(G, SigmaWX) - Z)**2) + 0.0001 * torch.sum(W**2)

    # grad = dW.flatten()

    # return cost, grad
    return cost

def sparsity_enforcement(X, lambda_):
    """
    Enforce sparsity on the input matrix X.

    Parameters:
    X (torch.Tensor): Input matrix.
    lambda_ (int): Number of largest elements to keep in each column.

    Returns:
    torch.Tensor: Sparse matrix with enforced sparsity.
    """
    device = X.device
    X_abs = torch.abs(X)
    X_sparse = X.clone()
    
    rows, cols = X.shape
    
    for n in range(cols):
        x_abs = X_abs[:, n]
        x = X[:, n]
        # Get indices of the largest `lambda_` elements
        _, maxidx = torch.topk(x_abs, lambda_, largest=True, sorted=False)
        # Get the indices that are not in maxidx
        all_indices = torch.arange(rows, device=device)
        mask = torch.ones_like(all_indices, dtype=torch.bool)
        mask[maxidx] = False
        zero_idx = all_indices[mask]
        # Zero out the elements not in maxidx
        x[zero_idx] = 0
        X_sparse[:, n] = x
    
    return X_sparse