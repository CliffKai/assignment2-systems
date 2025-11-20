import torch

# --- 1. 定义刚才那个自定义算子 ---
class MyCube(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_val):
        # input_val 就是网络中间层传进来的值 (即 w * x)
        print(f"[Cube Forward] 输入是 {input_val.item()}, 把它存进 ctx")
        ctx.save_for_backward(input_val)
        return input_val ** 3

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output 是 Loss 对 Cube输出的导数
        input_val, = ctx.saved_tensors
        print(f"[Cube Backward] 收到上游梯度: {grad_output.item()}")
        
        # 导数公式: 3 * input^2
        local_grad = 3 * (input_val ** 2)
        print(f"[Cube Backward] 本地梯度 (3*input^2): {local_grad.item()}")
        
        # 链式法则: 上游梯度 * 本地梯度
        final_grad = grad_output * local_grad
        print(f"[Cube Backward] 传给下一层的梯度: {final_grad.item()}")
        return final_grad

def my_cube(x):
    return MyCube.apply(x)

# --- 2. 准备数据和权重 ---
x = torch.tensor(2.0)                  # 输入
target = torch.tensor(64.0)            # 目标
w = torch.tensor(1.5, requires_grad=True) # 我们要优化的权重 (初始猜测 1.5)

print(f"--- 初始权重 w = {w.item()} ---")

# --- 3. 前向传播 (Forward) ---
print("\n1. 开始前向传播:")
# 步骤 A: 线性运算 (相当于全连接层)
u = w * x    # u = 1.5 * 2.0 = 3.0
# 步骤 B: 激活函数 (我们自定义的 Cube)
y_pred = my_cube(u) # y_pred = 3.0^3 = 27.0

# 步骤 C: 计算损失 (MSE Loss: (预测 - 目标)^2)
loss = (y_pred - target) ** 2 
# Loss = (27 - 64)^2 = (-37)^2 = 1369
print(f"   预测值: {y_pred.item()}, 目标: {target.item()}")
print(f"   Loss: {loss.item()}")

# --- 4. 反向传播 (Backward) ---
print("\n2. 开始反向传播:")
# 这一步会触发连环反应
loss.backward()

# --- 5. 查看结果与更新 ---
print("\n3. 结果分析:")
print(f"   w 的梯度 (w.grad): {w.grad.item()}")

# 手动更新权重 (模拟优化器)
learning_rate = 0.001
with torch.no_grad():
    w -= learning_rate * w.grad

print(f"   更新后的 w: {w.item()}")