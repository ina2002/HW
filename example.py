import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import random

# 固定参数
lambda_reg = 0.1
hat_rho = 3.0
T = 100
runs = 5
tau, eta = 1.0, 1.0
beta_seq = [hat_rho + 1 / (0.5 * np.sqrt(t + 1)) for t in range(T + 1)]

moreau_grads_all = []
final_losses = []

for seed in range(runs):
    np.random.seed(seed)
    random.seed(seed)

    # 数据生成
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X = StandardScaler().fit_transform(X)
    y = 2 * y - 1
    y01 = (y + 1) // 2

    n, d = X.shape
    x_t = np.zeros(d)
    x_trace = [x_t.copy()]

    for t in range(T):
        beta_t = beta_seq[t]
        i = random.randint(0, n - 1)
        z_i, y_i = X[i], y[i]

        def g_model(x): return np.log(1 + np.exp(-y_i * np.dot(x, z_i)))

        def prox_subproblem(x):
            reg_term = (lambda_reg / 2) * np.linalg.norm(x)**2
            prox_term = (beta_t / 2) * np.linalg.norm(x - x_t)**2
            return g_model(x) + reg_term + prox_term

        res = minimize(prox_subproblem, x_t, method='L-BFGS-B')
        x_t = res.x
        x_trace.append(x_t.copy())

    # 计算 Moreau 梯度范数
    moreau_grad_sq = []
    for t, x in enumerate(x_trace):
        def full_prox(y):
            pred = 1 / (1 + np.exp(-X @ y))
            loss = log_loss(y01, pred)
            reg = (lambda_reg / 2) * np.linalg.norm(y)**2
            prox_term = (hat_rho / 2) * np.linalg.norm(y - x)**2
            return loss + reg + prox_term

        prox_x = minimize(full_prox, x, method='L-BFGS-B').x
        grad_norm_sq = hat_rho**2 * np.linalg.norm(x - prox_x)**2
        moreau_grad_sq.append(grad_norm_sq)

    # 保存结果
    moreau_grads_all.append(moreau_grad_sq)
    pred_final = 1 / (1 + np.exp(-X @ x_t))
    final_loss = log_loss(y01, pred_final) + (lambda_reg / 2) * np.linalg.norm(x_t)**2
    final_losses.append(final_loss)
moreau_grads_all = np.array(moreau_grads_all)
mean_grad = moreau_grads_all.mean(axis=0)
std_grad = moreau_grads_all.std(axis=0)

# === 图 1: Moreau 梯度下降曲线 ===
plt.figure(figsize=(10,5))
plt.plot(mean_grad, label='Mean Moreau Gradient Norm')
plt.fill_between(range(T + 1),
                 mean_grad - std_grad,
                 mean_grad + std_grad,
                 color='gray', alpha=0.3, label='±1 std. dev.')
plt.xlabel("Iteration")
plt.ylabel(r"$\|\nabla \varphi_{1/\hat{\rho}}(x_t)\|^2$")
plt.title("Convergence of Moreau Envelope Gradient (Mean ± Std)")
plt.legend()
plt.grid()
plt.savefig("moreau_gradient_convergence.png", bbox_inches='tight')   
plt.show()

# === 图 2: 最终损失柱状图 ===
plt.figure(figsize=(6,5))
plt.bar(range(runs), final_losses)
plt.axhline(np.mean(final_losses), color='r', linestyle='--', label='Mean')
plt.title("Final Loss across 5 Runs")
plt.ylabel("Loss")
plt.xlabel("Run Index")
plt.legend()
plt.grid()
plt.savefig("final_loss_comparison.png", bbox_inches='tight')  
plt.show()
