from saitama_dl.tensor.core import Tensor
import saitama_dl as sdl

# Simple dataset: y = 3x + 2
x_data = [1, 2, 3, 4, 5]
y_data = [5, 8, 11, 14, 17]


# Initialize weights
w = Tensor([0.5])
b = Tensor([0.1])
lr = Tensor([0.0001])

for epoch in range(20000):
    total_loss = 0
    for i in range(len(x_data)):
        x = Tensor([x_data[i]])
        y = Tensor([y_data[i]])

        # Forward
        y_pred = x * w + b

        # Loss
        error = y_pred - y
        loss = error * error
        
        total_loss += loss.to_data()[0]

        # Backward
        loss.backward()

        # Update
        with sdl.no_grad():
            w.data = (w - lr * w.grad).data
            b.data = (b - lr * b.grad).data
    
        # Zero grad after each sample update
        w.grad.zero_()
        b.grad.zero_()
    
    if epoch % 1000 == 0:
        print(f"epoch {epoch} | loss: {total_loss:.4f} | w: {w.to_data()} | b: {b.to_data()}")