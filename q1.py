import torch
x = torch.tensor([2.0], requires_grad=True)

y1 = x**2
y1.backward(retain_graph=True)   # 第一次 backward
print(x.grad)   # tensor([4.])
x = x.detach()
y2 = 3*x
y2.backward()   # 第二次 backward
print(x.grad)   # tensor([7.]) = 4 + 3
