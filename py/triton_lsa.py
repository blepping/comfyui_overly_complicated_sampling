import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"num_warps": 4, "num_stages": 2}, num_warps=4, num_stages=2),
        triton.Config({"num_warps": 8, "num_stages": 2}, num_warps=8, num_stages=2),
        triton.Config({"num_warps": 4, "num_stages": 3}, num_warps=4, num_stages=3),
        triton.Config({"num_warps": 8, "num_stages": 3}, num_warps=8, num_stages=3),
    ],
    key=[
        "B",
        "R",
        "C",
        "BLOCK_SIZE",
    ],  # Retune if matrix dimensions change significantly
)
@triton.jit
def auction_lap_kernel(
    cost_ptr,
    assign_ptr,
    stride_b,
    stride_r,
    stride_c,
    stride_assign_b,
    stride_assign_r,
    B: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    epsilon,
    max_iter,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    cost_base = cost_ptr + pid * stride_b
    assign_base = assign_ptr + pid * stride_assign_b

    offs = tl.arange(0, BLOCK_SIZE)
    col_mask = offs < C

    # Prices and Owners in SRAM/Registers
    prices = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    owners = tl.full([BLOCK_SIZE], -1, dtype=tl.int32)
    row_to_col = tl.full([BLOCK_SIZE], -1, dtype=tl.int32)

    iter_idx = 0
    unassigned_count = R

    loop_continue = tl.full([], 1, dtype=tl.int1)

    # Loop condition:
    # 1. unassigned_count > 0: Logic handled inside, but we need a break mechanism
    # 2. iter_idx < max_iter: Safety break
    # 3. loop_continue: Did we make progress last time?

    while unassigned_count > 0 and iter_idx < max_iter and loop_continue:
        # Reset progress flag
        # loop_continue &= False
        loop_continue = tl.full([], 0, dtype=tl.int1)

        # Gauss-Seidel pass over all rows
        for i in tl.range(0, R):
            # Check if row i is unassigned
            curr_c = tl.sum(tl.where(offs == i, row_to_col, 0))

            if curr_c == -1:
                # Load costs
                row_cost_ptr = cost_base + i * stride_r + offs
                row_costs = tl.load(row_cost_ptr, mask=col_mask, other=-torch.inf)

                # Net Value
                values = row_costs - prices

                # Find Best
                best_val, best_idx = tl.max(values, axis=0, return_indices=True)

                # CRITICAL: Only proceed if this is a valid edge (not -inf)
                if best_val > -torch.inf:
                    # We have a valid move, so we continue the outer loop
                    loop_continue = tl.full([], 1, dtype=tl.int1)

                    # Find Second Best
                    mask_not_best = (offs != best_idx) & col_mask
                    vals_no_best = tl.where(mask_not_best, values, -torch.inf)
                    second_best_val = tl.max(vals_no_best, axis=0)

                    # Compute Bid
                    bid = best_val - second_best_val + epsilon

                    # Update Price
                    prices = tl.where(offs == best_idx, prices + bid, prices)

                    # Update Owners
                    prev_owner = tl.sum(tl.where(offs == best_idx, owners, 0))

                    if prev_owner != -1:
                        # Kick out previous owner
                        row_to_col = tl.where(offs == prev_owner, -1, row_to_col)
                        unassigned_count += 1

                    # Assign to current row
                    owners = tl.where(offs == best_idx, i, owners)
                    row_to_col = tl.where(offs == i, best_idx, row_to_col)
                    unassigned_count -= 1

        iter_idx += 1

    # Store Result
    store_offs = tl.arange(0, BLOCK_SIZE)
    store_mask = store_offs < R
    tl.store(assign_base + store_offs, row_to_col, mask=store_mask)


# -----------------------------------------------------------------------------
# Python Helpers
# -----------------------------------------------------------------------------


def rescale_simple(
    t: torch.Tensor,
    target_min: float = 0.0,
    target_max: float = 1.0,
    *,
    start_dim: int = 1,
    eps: float = 1e-07,
) -> torch.Tensor:
    width = target_max - target_min
    if width == 0.0:
        return torch.zeros_like(t)
    orig_shape = t.shape
    t = t.flatten(start_dim=start_dim)
    min_val, max_val = t.aminmax(dim=-1, keepdim=True)
    normalized = t - min_val
    normalized /= (max_val - min_val).add_(eps)
    normalized *= width
    if target_min != 0.0:
        normalized += target_min
    return normalized.clamp_(target_min, target_max).reshape(orig_shape)


def _greedy_fill_missing(assignments: torch.Tensor, C: int) -> None:
    """
    Fills unassigned rows (-1) in the assignments tensor with available columns.
    This acts as a fallback when the Auction algorithm hits max_iter without
    full convergence.

    Args:
        assignments: Tensor of shape (B, R) containing col indices or -1.
        C: Total number of columns available.
    """
    # Identify which batch items have unassigned rows
    # This is usually a very small subset (e.g., < 1% of the batch)
    problem_batches = (assignments == -1).any(dim=1).nonzero().flatten()

    if problem_batches.numel() == 0:
        return

    device = assignments.device

    # Iterate only over the problematic batch items
    # (Looping is acceptable here as B_subset is typically tiny)
    for b_idx in problem_batches:
        # 1. Find which rows are missing an assignment
        row_mask = assignments[b_idx] == -1
        missing_rows = row_mask.nonzero().flatten()
        n_needed = missing_rows.shape[0]

        # 2. Find which columns are already used
        used_cols = assignments[b_idx][~row_mask]

        # 3. Find free columns (Set difference: All - Used)
        # Create a boolean mask of all columns, then mark used ones as False
        # efficient on GPU for mid-sized C
        col_mask = torch.ones(C, device=device, dtype=torch.bool)
        col_mask[used_cols.long()] = False

        free_cols = col_mask.nonzero().flatten()

        # 4. Assign the first N free columns to the N missing rows
        # Since R <= C in this context (due to transpose logic in wrapper),
        # free_cols.numel() is guaranteed to be >= n_needed.
        assignments[b_idx, missing_rows] = free_cols[:n_needed].to(assignments.dtype)


def batch_linear_assignment(
    cost_matrix: torch.Tensor,
    *,
    maximize: bool = False,
    max_iter: int | None = None,
    fill_missing: bool = True,
    rescale_costs: tuple[float, float] | None = (0.0, 1.0),
    invert_costs_mode: bool = True,
    eps: float = 1e-3,
):
    if cost_matrix.ndim != 3:
        raise ValueError("Cost matrix must be (B, R, C)")
    if not cost_matrix.is_cuda:
        raise ValueError("Cost matrix must be a CUDA tensor")
    if not cost_matrix.is_contiguous():
        raise ValueError("Cost matrix must be contiguous")

    B, R, C = cost_matrix.shape
    device = cost_matrix.device

    # 1. Handle Rectangular Matrices
    # The Auction algorithm assigns Rows -> Cols.
    # It naturally handles R <= C (finding best col for every row).
    # If R > C, we must transpose to match Cols -> Rows, then invert the result.
    if R > C:
        transposed = True
        cost_matrix = cost_matrix.mT.contiguous()
        # Swap R and C for the kernel execution
        R, C = C, R
    else:
        transposed = False

    if rescale_costs is not None:
        cost_matrix = rescale_simple(cost_matrix, *rescale_costs)
    # Note: We use float32 for atomic compatibility and speed
    cost_matrix = cost_matrix.to(torch.float32, copy=rescale_costs is None)

    if not maximize:
        # Maximize (Value - Price) -> Minimize Cost
        cost_matrix = cost_matrix.neg_()
        if invert_costs_mode and rescale_costs is not None:
            cost_matrix += sum(rescale_costs)

    assignments = torch.full(
        (B, R),
        -1,
        device=device,
        dtype=torch.int32,
    )

    max_dim = max(R, C)
    BLOCK_SIZE = max(32, triton.next_power_of_2(max_dim))

    # Safety limit
    max_iter = max_iter if max_iter is not None else int(max(2000, R * C))

    grid = (B,)

    auction_lap_kernel[grid](
        cost_matrix,
        assignments,
        cost_matrix.stride(0),
        cost_matrix.stride(1),
        cost_matrix.stride(2),
        assignments.stride(0),
        assignments.stride(1),
        B,
        R,
        C,
        eps,
        max_iter,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if fill_missing:
        _greedy_fill_missing(assignments, C)

    assignments = assignments.long()

    if not transposed:
        return assignments

    # 2. Post-process Rectangular Results
    # We computed Col -> Row. We need Row -> Col.
    # assignments shape is currently (B, Original_Cols)
    # We want output shape (B, Original_Rows)

    real_rows = C  # C is the 'large' dimension (Original Rows)
    output = torch.full((B, real_rows), -1, device=device, dtype=torch.long)

    # Create indices for the scatter source
    # We want: output[row_idx] = col_idx
    # Currently we have: assignments[col_idx] = row_idx
    src_col_indices = torch.arange(R, device=device).unsqueeze(0).expand(B, R)

    # We use scatter. index=assignments (the rows), src=col_indices
    # To handle -1s in assignments, we clamp to 0 and then mask the result
    safe_assigns = assignments.clamp(min=0)
    output.scatter_(1, safe_assigns, src_col_indices)

    # Cleanup: Any row that wasn't targeted by the scatter should be -1
    # The scatter might have written to index 0 if assignment was -1
    # Re-verify logic:
    for b in range(B):
        valid_mask = assignments[b] >= 0
        # Reset output
        output[b].fill_(-1)
        # Only write valid mappings
        # output[b, row_id] = col_id
        output[b, assignments[b, valid_mask]] = src_col_indices[b, valid_mask]

    return output


def assignments_to_indices(
    assignments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a dense assignment tensor (from Triton/Hungraian) to
    batched SciPy-style indices.

    Args:
        assignments (torch.Tensor): Shape (B, R). Values are col indices or -1.

    Returns:
        row_ind (torch.Tensor): Shape (B, K) where K = min(R, C).
        col_ind (torch.Tensor): Shape (B, K).
    """
    B, R = assignments.shape
    device = assignments.device

    # 1. Create a mask of valid assignments (values >= 0)
    # In a rectangular assignment, the number of valid matches
    # is always min(Rows, Cols).
    mask = assignments >= 0

    # 2. Extract Column Indices
    # We select the values from the assignment tensor that are valid.
    # We reshape to (B, -1) to preserve the batch dimension.
    col_ind = assignments[mask].view(B, -1)

    # 3. Extract Row Indices
    # We need a grid of row indices [0, 1, 2, ... R-1] repeated B times
    row_grid = (
        torch.arange(R, device=device, dtype=assignments.dtype)
        .unsqueeze(0)
        .expand(B, R)
    )
    row_ind = row_grid[mask].view(B, -1)

    return row_ind, col_ind


def batch_linear_assignment_shuffled(
    cost_matrix: torch.Tensor,
    *args,
    **kwargs: dict,
) -> torch.Tensor:
    generator = kwargs.pop("generator", None)
    # cost_matrix: [B, R, C]
    B, R = cost_matrix.shape[:2]

    # 1. Generate a random permutation for the rows
    # We use one perm for the whole batch for efficiency,
    # or you can do it per-batch-item if B is small and quality is critical.
    # Here we shuffle all rows commonly.
    perm = torch.randperm(
        R,
        device=cost_matrix.device,
        generator=generator,
    )

    # 2. Shuffle the input (Row dimension is dim 1)
    # This creates a shuffled view/copy of the cost matrix
    shuffled_cost = cost_matrix[:, perm, :]

    # 3. Run the Solver
    shuffled_assignments = batch_linear_assignment(
        shuffled_cost,
        *args,
        **kwargs,
    )  # Returns [B, R]

    # 4. Un-shuffle the results
    # We need to map the results back to their original row positions.
    # shuffled_assignments[b, i] corresponds to the row 'perm[i]'
    # We want final_assignments[b, perm[i]] = shuffled_assignments[b, i]

    # Create the inverse permutation or just scatter back
    final_assignments = torch.empty_like(shuffled_assignments)

    # Expand perm for the batch: [B, R]
    batch_perm = perm.unsqueeze(0).expand(B, R)

    # Scatter the results back to original positions
    # dim=1, index=batch_perm, src=shuffled_assignments
    final_assignments.scatter_(1, batch_perm, shuffled_assignments)

    return final_assignments
