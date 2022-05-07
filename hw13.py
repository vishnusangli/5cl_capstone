# %%
import numpy as np
# %%
a1 = np.array([1, 1j, 0])
a2 = np.array([0, -1, 0])
a3 = np.array([1, 1, 1])

def proj(a, b):
    """
    Projection of a onto b
    """
    val = np.vdot(b, a)
    print(val)
    print((val*b))
    return (val*b)

def norm(a):
    mag = np.vdot(a, a)
    return np.divide(a, np.sqrt(mag))
# %%
q1 = norm(a1)
z2 = a2 - proj(a2, q1)
q2 = norm(z2)

z3 = a3 - proj(a3, q1) - proj(a3, q2)
q3 = norm(z3)
# %%
print(proj(q1, q2))
print(proj(q3, q2))
print(proj(q1, q3))

# %%
