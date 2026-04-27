import torch
import einops
import triton
import triton.language as tl
from jaxtyping import Float
from torch import Tensor


def weighted_sum(x: Float[Tensor, "... d"], weight: Float[Tensor, "d"]):
    """PyTorch version weighted sum"""
    return (weight * x).sum(axis=-1)

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,  # Raw GPU memory pointers to the input tensors
    output_ptr,  # Raw GPU memory pointer to the output tensor
    x_stride_row, x_stride_dim,  # How many elements to skip to move one row / one column in x
    weight_stride_dim,  # How many elements to skip to move one position in weight (usually 1)
    output_stride_row,  # How many elements to skip to move one position in output (usually 1)
    NUM_ROWS,  # Total number of rows in x after flattening all dims except D
    D,  # The last dimension size (the dimension we sum over)
    ROWS_TILE_SIZE: tl.constexpr,  # How many rows each kernel instance processes (compile-time constant)
    D_TILE_SIZE: tl.constexpr,  # How many columns we process per loop iteration (compile-time constant)
):
    """Forward pass for weighted sum: output[i] = sum_d(x[i, d] * weight[d]).

    Parallelization strategy: each kernel instance handles a tile of ROWS_TILE_SIZE rows,
    and loops over the D dimension in chunks of D_TILE_SIZE.
    """

    # Each kernel instance gets a unique index (0, 1, 2, ...).
    # This tells us which chunk of rows we are responsible for.
    row_tile_idx = tl.program_id(0)

    # === Set up block pointer for x ===
    # Think of this as a sliding window over the 2D (NUM_ROWS x D) matrix.
    # - shape: the full logical shape, used for out-of-bounds checking
    # - strides: memory layout (how to jump between rows/columns)
    # - offsets: where this window starts (our row chunk, column 0)
    # - block_shape: the window size (ROWS_TILE_SIZE x D_TILE_SIZE)
    # - order: (1, 0) means dim 1 (columns) is contiguous in memory (row-major)
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D,),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # === Set up block pointer for weight ===
    # weight is 1D of shape (D,). We read D_TILE_SIZE elements at a time,
    # starting from position 0, and slide right each iteration.
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # === Set up block pointer for output ===
    # output is 1D of shape (NUM_ROWS,). Each kernel instance writes
    # ROWS_TILE_SIZE results starting at its row chunk offset.
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(NUM_ROWS, ),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # Accumulator for the partial dot products, one per row in our tile.
    # Uses float32 for numerical stability even if inputs are float16/bfloat16.
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    
    # Loop over the D dimension in chunks of D_TILE_SIZE.
    # ceil division so we don't miss the tail when D is not divisible by D_TILE_SIZE.
    num_d_tiles = tl.cdiv(D, D_TILE_SIZE)
    for i in range(num_d_tiles):
        # Load a (ROWS_TILE_SIZE x D_TILE_SIZE) tile from x.
        # boundary_check=(0, 1): pad with zeros if we go past NUM_ROWS or D.
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # Load a (D_TILE_SIZE,) slice from weight.
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")

        # Elementwise multiply: broadcast weight from (D_TILE_SIZE,) to (ROWS_TILE_SIZE, D_TILE_SIZE),
        # then sum over the column axis (axis=1) to get partial sums per row.
        output += tl.sum(row * weight[None, :], axis=1)

        # Slide both windows right by D_TILE_SIZE columns for the next iteration.
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
    
    # Write the final accumulated results back to GPU memory.
    tl.store(output_block_ptr, output, boundary_check=(0,))


@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,          # Saved inputs from forward pass
    grad_output_ptr,            # Gradient flowing back from downstream: shape (NUM_ROWS,)
    grad_x_ptr,                 # Output: gradient wrt x, shape (NUM_ROWS, D)
    partial_grad_weight_ptr,    # Output: each kernel instance's partial gradient wrt weight, shape (n_row_tiles, D)
    stride_xr, stride_xd,      # Strides for x (row, column)
    stride_wd,                  # Stride for weight
    stride_gr,                  # Stride for grad_output
    stride_gxr, stride_gxd,    # Strides for grad_x (row, column)
    stride_gwb, stride_gwd,    # Strides for partial_grad_weight (tile_idx, column)
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    """Backward pass for weighted sum.

    Given output[i] = sum_d(x[i,d] * weight[d]), the gradients are:
      grad_x[i,d]    = grad_output[i] * weight[d]         (outer product)
      grad_weight[d]  = sum_i(x[i,d] * grad_output[i])    (reduced across rows)

    For grad_weight, each kernel instance computes a partial sum over its
    ROWS_TILE_SIZE rows. The Python caller then sums these partials.
    """

    # Which chunk of rows this kernel instance handles
    row_tile_idx = tl.program_id(0)
    # Total number of kernel instances (= total row tiles)
    n_row_tiles = tl.num_programs(0)

    # === Block pointer for grad_output (1D, read-only) ===
    # Each instance reads the same ROWS_TILE_SIZE slice across all D iterations
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # === Block pointer for x (2D, read-only, slides along D) ===
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D,), strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # === Block pointer for weight (1D, read-only, slides along D) ===
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,), strides=(stride_wd,),
        offsets=(0,), block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # === Block pointer for grad_x (2D, write-only, slides along D) ===
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D,), strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # === Block pointer for partial_grad_weight (2D, write-only, slides along D) ===
    # Shape is (n_row_tiles, D): one row per kernel instance.
    # Each instance writes to its own row (row_tile_idx), so no race conditions.
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D,), strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    # Loop over D in chunks of D_TILE_SIZE (same pattern as forward)
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # grad_output shape: (ROWS_TILE_SIZE,) — same slice reloaded each iteration
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")

        # --- Compute grad_x[i,d] = grad_output[i] * weight[d] ---
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE,)
        # Outer product: (ROWS_TILE_SIZE, 1) * (1, D_TILE_SIZE) -> (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_x_tile = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_tile, boundary_check=(0, 1))

        # --- Compute partial grad_weight[d] = sum_i(x[i,d] * grad_output[i]) ---
        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        # Multiply each row of x by its corresponding grad_output scalar, then sum across rows
        # Result shape: (1, D_TILE_SIZE) — partial contribution from this tile's rows
        partial_gw_tile = tl.sum(x_tile * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, partial_gw_tile, boundary_check=(1,))

        # Slide all D-dimension windows right
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))
    

class WeightedSumFunc(torch.autograd.Function):
    """PyTorch autograd wrapper that bridges normal PyTorch tensors and the Triton kernels.

    PyTorch's autograd expects a class with forward() and backward() methods.
    forward() runs the computation; backward() computes gradients for training.
    """

    @staticmethod
    def forward(ctx, x, weight):
        D = x.shape[-1]        # The dimension we'll sum over
        input_shape = x.shape  # Remember original shape for reshaping output later

        # Flatten all leading dims into one: e.g. (batch, seq, D) -> (batch*seq, D).
        # The Triton kernel only understands 2D: (NUM_ROWS, D).
        x = einops.rearrange(x, "... d -> (...) d")

        # Save tensors needed by backward(). PyTorch stores these and provides
        # them later when backward() is called during loss.backward().
        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        # Choose tile sizes. D_TILE_SIZE controls how many columns we process per
        # loop iteration. ROWS_TILE_SIZE controls how many rows per kernel instance.
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16
        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = input_shape

        # Allocate output as a flat 1D tensor. The kernel writes to it with stride 1.
        # We reshape to the original leading dims at the very end.
        n_rows = x.shape[0]  # = product of all original dims except D
        y = torch.empty(n_rows, device=x.device)

        # Launch the kernel. The grid has one instance per tile of rows.
        # Triton syntax: kernel[(grid_size,)](args...) — the tuple in [] is the grid.
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight,              # Triton auto-converts torch.Tensor to raw GPU pointers
            y,
            x.stride(0), x.stride(1),   # For contiguous 2D tensor: (D, 1)
            weight.stride(0),            # For contiguous 1D tensor: (1,)
            y.stride(0),                 # For contiguous 1D tensor: (1,)
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE
        )

        # Reshape flat output back to original leading dimensions.
        # e.g. if input was (batch, seq, D), output becomes (batch, seq).
        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        """Compute gradients wrt x and weight.

        grad_out has the same shape as forward()'s return value, e.g. (32, 8).
        It is NOT explicitly flattened, but since it's contiguous, its memory layout
        is identical to a flat (NUM_ROWS,) tensor. The kernel handles this via strides.
        We need:
          grad_x[i,d]   = grad_out[i] * weight[d]        (for each row and column)
          grad_weight[d] = sum_i(grad_out[i] * x[i,d])    (sum across all rows)
        """
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape

        # For grad_weight, each kernel instance sums over its own ROWS_TILE_SIZE rows,
        # producing a (1, D) partial result. We allocate a (n_tiles, D) buffer for all
        # partials, then sum them on the Python side afterward.
        n_tiles = triton.cdiv(n_rows, ROWS_TILE_SIZE)
        partial_grad_weight = torch.empty((n_tiles, D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)

        weighted_sum_backward[(n_tiles,)](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )

        # Final reduction: sum all partial grad_weight contributions across tiles.
        # (n_tiles, D) -> (D,)
        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x, grad_weight


if __name__ == "__main__":
    f_weightedsum = WeightedSumFunc.apply
    x = torch.randn((1024, 1024, 1024), device="cuda")
    weight = torch.randn(1024, device="cuda")
    result = f_weightedsum(x, weight)
    print(result)
    print(f"shape: {result.shape}")
    