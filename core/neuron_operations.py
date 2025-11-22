# Weighted Sum Operation
def weighted_sum_operation(CONTRIB_n, delta_n, w_i, delta_i):
    return CONTRIB_n * delta_n * w_i * delta_i

# Average Pooling Operation
def avg_pool_operation(CONTRIB_n, delta_n, delta_i):
    return CONTRIB_n * delta_n * delta_i

# Max Pooling Operation
def max_pool_operation(CONTRIB_n, delta_n, delta_i, x, y):
    if x == y:
        return CONTRIB_n * delta_n * delta_i
    else:
        return 0.0
    
# ReLU Operation
def relu_operation(CONTRIB_n, delta_n, delta_i, x):
    if x > 0:
        return CONTRIB_n * delta_n * delta_i
    else:
        return 0.0

# Batch Normalization Operation
def scale_operation(CONTRIB_n, delta_n, delta_i):
    return CONTRIB_n * delta_n * delta_i
