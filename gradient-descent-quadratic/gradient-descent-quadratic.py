def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    gradient = lambda x:2*a*x + b

    for i in range(steps):
        x0 -= lr * gradient(x0)

    return x0