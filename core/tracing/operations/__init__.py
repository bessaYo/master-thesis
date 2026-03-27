from core.tracing.operations.ops_vec import BackwardOperations as VectorizedOps
from core.tracing.operations.ops_loop import BackwardOperationsLoop as LoopOps


def get_backward_operations(theta=0.0, vectorized=True):
    if vectorized:
        return VectorizedOps(theta=theta)
    return LoopOps(theta=theta)
