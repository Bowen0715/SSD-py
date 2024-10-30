import numpy as np
import spams
import torch
from torch import tensor, matmul, cat, norm

class Model:
    def __init__(self):
        """
        Supervised Sparse Decomposition (SSD) training model

        Attributes:
        - X (numpy.ndarray): input matrix (m x N), feature dim: m, sample dim: N
        - L (numpy.ndarray): label vector (1 x N), regression target for targeted mode
        - D (numpy.ndarray): Dictionary matrix (m x h).
        - Z (numpy.ndarray): Sparse representation matrix (h x N).
        - W (numpy.ndarray): The encoder for sparse code (h x m).
        - G (numpy.ndarray): Diagonal matrix (h x h).
        - A (numpy.ndarray): The bias (intercept) and regression coefficient vector (1 x (1 + h)).
        """
        self.X = None
        self.L = None
        self.D = None
        self.Z = None
        self.W = None
        self.G = None
        self.A = None

    def config(self, h, lambda_, nepoch=1000, beta=1, device="cpu"):
        """
        Parameters:
        h (int): number of atoms, dictionary elements 
        lambda_: sparsity, non-zero elements in the sparse code
        nepoch (int): number of iterations
        beta: whethter to use targeted model, beta=0 -> non-targeted beta=1 -> targeted
        """
        self.h = h
        self.lambda_ = lambda_
        self.nepoch = nepoch
        self.beta = beta
        self.device = device

    @torch.no_grad()
    def fit(self, X, L, D=None):
        """
        Train Sparse Decomposition Regression with the minimization objective: 
        || DZ - X ||_F + || Z - G SIGMA(WX) ||_F + || AZ - (L-A0) ||_F 

        Parameters:
        X (numpy.ndarray): input matrix (m x N), feature dim: m, sample dim: N
        L (numpy.ndarray): label vector (1 x N), regression target for targeted mode

        Side Effects:
        - Modifies attributes: W, G, D, A, Z
        """
        # Set seeds
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        eps = torch.finfo(torch.float32).tiny
        # Set parameters
        device = self.device
        # spams.trainDL
        DLparam = {
            "mode": 2,
            "K": self.h, # learns a dictionary with h elements
            "lambda1": self.lambda_,
            "numThreads": 7, # number of threads
            "batchsize": 2500,
            "iter": 100,
        }
        # spams.lasso
        param = {"lambda1": self.lambda_, "lambda2": 0, "numThreads": -1, "mode": 2, "pos": True}

        # Initialize X and L
        X = self.X = tensor(X, dtype=torch.float32, device=device)
        L = self.L = tensor(L, dtype=torch.int32, device=device)
        X_intercept = cat([torch.ones(1, N, device=device), X], dim=0)

        # Initialize D
        _, N = X.shape
        X_cpu = np.asfortranarray(X)
        FIX_D = D is not None  # Flag for whether to use the fixed dict mode
        if not FIX_D:
            D_init = spams.trainDL(X_cpu, **DLparam)
            D = tensor(D_init, dtype=torch.float32, device=device)
            D = D @ torch.diag(1 / (torch.sqrt(torch.sum(D**2, dim=0)) + eps))
        K = D.shape[1]

        # Initialize Z, W, G and A
        Z = tensor([]).to(device)
        W = cat([torch.rand(K, 1, device=device), D.T], dim=1)
        G = torch.eye(D.shape[1], device=device)
        A0 = tensor(np.random.rand(), dtype=torch.float32, device=device)
        A = torch.rand(1, K, device=device)

        # Training loop
        trloss = np.zeros(self.nepoch) # Object error array
        for ite in range(self.nepoch):
            # Update D and A
            if Z.size(0) != 0:
                tau_d = 0.99 * 2 / norm(matmul(Z, Z.T), p="fro")
                for _ in range(50):
                    if FIX_D:
                        if self.beta != 0:
                            A = gradDesc(A, Z, cat([(L - A0 * torch.ones_like(L)).view(1, -1)]), tau_d)
                            A0 = torch.mean(A0 - 0.1 * (A0 + matmul(A, Z) - L))
                    else:
                        if self.beta != 0:
                            DNew = gradDesc(cat([D, A], dim=0), Z, cat([X, (L - A0 * torch.ones_like(L)).view(1, -1)]), tau_d)
                            D = DNew[:-1, :]
                            A = DNew[-1, :].reshape(1, K)
                            A0 = torch.mean(A0 - 0.1 * (A0 + matmul(A, Z) - L))
                            D = torch.abs(D)
                            D = D @ torch.diag(1 / (torch.sqrt(torch.sum(D**2, dim=0)) + eps))
                        else:
                            D = gradDesc(D, Z, X, tau_d)
                            D = torch.abs(D)
                            D = D @ torch.diag(1 / (torch.sqrt(torch.sum(D**2, dim=0)) + eps))
            # Update Z 
            SigmaWX = torch.sigmoid(matmul(W, X_intercept))
            if self.beta != 0:
                X_eq = cat([X, matmul(G, SigmaWX), (L - A0 * torch.ones_like(L)).view(1, -1)], dim=0)
                D_eq = cat([D, torch.eye(D.shape[1], device=device), A], dim=0)
            else:
                X_eq = cat([X, matmul(G, SigmaWX)], dim=0)
                D_eq = cat([D, torch.eye(D.shape[1], device=device)], dim=0)
            D_eq = D_eq @ torch.diag(1 / (torch.sqrt(torch.sum(D_eq**2, dim=0)) + eps))
            X_eq_cpu = np.asfortranarray(X_eq.cpu().numpy())
            D_eq_cpu = np.asfortranarray(D_eq.cpu().numpy())
            Z = spams.lasso(X_eq_cpu, D_eq_cpu, **param).toarray()
            Z = tensor(Z, dtype=torch.float32, device=device)
            if self.beta != 0:
                Z = torch.diag(1 / torch.sqrt(torch.sum(cat([D, torch.eye(D.shape[1], device=device), A], dim=0)**2, dim=0))) @ Z
            else:
                Z = torch.diag(1 / torch.sqrt(torch.sum(cat([D, torch.eye(D.shape[1], device=device)], dim=0)**2, dim=0))) @ Z

            # Update G
            SigmaWX = torch.sigmoid(matmul(W, X_intercept))
            temp = torch.diag(1 / (torch.sum(SigmaWX**2, dim=1) + eps))
            G = torch.diag((temp @ SigmaWX) @ Z.T)
            G = torch.diag(G)

            err_old = norm(matmul(G, torch.sigmoid(matmul(W, X_intercept))) - Z, p="fro") / norm(Z, p="fro")
            # Update W
            theta = W.flatten().clone().requires_grad_(True).to(device)
            optimizer = torch.optim.LBFGS([theta], max_iter=100, history_size=10, line_search_fn="strong_wolfe")
            def closure():
                optimizer.zero_grad()
                loss = object_fun(theta, Z, X_intercept, G)
                loss.backward(retain_graph=True)
                return loss
            optimizer.step(closure)
            W_new = theta.view(Z.shape[0], X_intercept.shape[0])

            # Calculate error
            err_cur = norm(matmul(G, torch.sigmoid(matmul(W_new, X_intercept))) - Z, p="fro") / norm(Z, p="fro")
            print(f'old error: {err_old}')
            print(f'cur error: {err_cur}')
            print(f'old - cur = {err_old - err_cur}')
            if err_cur <= err_old:
                W = W_new
                err_old = err_cur
                print('decreasing')
            else:
                print('ascending')

            trloss[ite] = norm(X - matmul(D, Z), p="fro")**2 + norm(matmul(G, torch.sigmoid(matmul(W, X_intercept))) - Z, p="fro")**2
            X_approx_err_ra = norm(X - matmul(D, matmul(G, torch.sigmoid(matmul(W, X_intercept)))), p="fro") / norm(X, p="fro")
            X_err_ra = norm(X - matmul(D, Z), p="fro") / norm(X, p="fro")
            
            if self.beta != 0:
                A0tensor = tensor([[A0]], dtype=torch.float32, device=device)
                Z_intercept = cat([torch.ones(1, N, device=device), Z], dim=0)
                L_hat = matmul(cat([A0tensor, A], dim=1), Z_intercept)
                reg_mse = torch.mean((L_hat - L)**2)

                print(f'Ite {ite} Object error: {trloss[ite].item()} Za err ratio: {X_approx_err_ra.item()} Z err ratio: {X_err_ra.item()} Reg MSE: {reg_mse.item()}')
            else:
                print(f'Ite {ite} Object error: {trloss[ite].item()} Za err ratio: {X_approx_err_ra.item()} Z err ratio: {X_err_ra.item()}')

        self.X_approx_err_ra = X_approx_err_ra
        self.X_err_ra = X_err_ra
        self.W = W
        self.G = G
        self.D = D
        self.Z = Z
        self.Z_approx = matmul(G, torch.sigmoid(matmul(W, X_intercept)))
        if self.beta != 0:
            self.A = cat([A0tensor, A], dim=1)
            self.reg_mse = reg_mse



def gradDesc(D, Z, X, tau_d):
    """
    Update dictionary D based on either gradient descent.
    
    Parameters:
    D (numpy.ndarray): Dictionary matrix.
    Z (numpy.ndarray): Sparse code matrix, each column is a sparse code.
    X (numpy.ndarray): Feature matrix, each column is a feature.
    taud (float): Learning rate.
    
    Returns:
    numpy.ndarray: Updated dictionary matrix D.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return D - tau_d * (matmul(D, Z) - X) @ Z.T

def object_fun(theta, Z, X, G):

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    W = theta.view(Z.shape[0], X.shape[0])
    SigmaWX = torch.sigmoid(matmul(W, X))

    dW = 2 * (matmul(G, SigmaWX) - Z) * SigmaWX * (1 - SigmaWX)
    dW = matmul(dW, X.T) + 2 * 0.0001 * W

    cost = torch.sum((matmul(G, SigmaWX) - Z)**2) + 0.0001 * torch.sum(W**2)

    return cost
