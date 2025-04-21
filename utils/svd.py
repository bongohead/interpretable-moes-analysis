import torch

def decompose_orthogonal(hidden_states: torch.Tensor, target_tensor: torch.Tensor, method: str = 'svd', svd_tol: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decomposes hidden states into components parallel and orthogonal to the target tensor. The target tensor can be either a routing gate or the MLP operation.

    For example, if the target tensor is the routing gate:
    - The component parallel to the row space ('h_para') contains the information seen by the linear routing mechanism (logits = W_g @ h).
    - The component orthogonal to the row space ('h_orth') contains information ignored by the linear router mechanism, but potentially used by the non-linear expert MLP or downstream layers.

    Params:
        @hidden_states: Tensor of shape (n_samples, D) representing the pre-routing hidden states.
        @target_tensor: Target tensor of shape (., D). If a routing gate, a tensor of shape (n_experts, D) representing the linear router gate weights for the layer.
        @method: Decomposition method, 'svd' (default) or 'qr'.
        @svd_tol: Tolerance for determining non-zero singular values in SVD to establish the matrix rank.

    Returns:
        A tuple containing:
        - h_para (torch.Tensor): Projection onto the row space ("used" by target tensor). Shape (n_samples, D).
        - h_orth (torch.Tensor): Projection onto the orthogonal complement ("unused" by target tensor). Shape (n_samples, D).

    Example:
        h_para, h_orth = decompose_orthogonal(all_pre_mlp_hs[0:10_000, 1, :].to(torch.float32), model.model.layers[1].mlp.gate.weight.to(torch.float32).detach().cpu(), 'svd')
        dot_products_svd = torch.sum(h_para * h_orth, dim=1)
        print(f"Mean dot product (SVD): {torch.mean(dot_products_svd).item():.4e}")
        print(f"Max absolute dot product (SVD): {torch.max(torch.abs(dot_products_svd)).item():.4e}")

        reconstruction_diff_svd = torch.norm(all_pre_mlp_hs[0:10_000, 1, :].to(torch.float32) - (h_para + h_orth), dim=1)
        print(f"Mean reconstruction norm diff (SVD): {torch.mean(reconstruction_diff_svd).item():.4e}")

        # Can also verify that QR orthogonality/reconstruction is close to 0, and also that SVD and QR results shoudl be close torch.norm(h_svd = h_qr)
    """
    _, D = hidden_states.shape

    assert D == target_tensor.shape[1], 'Hidden state dim != router gate dim'

    if method == 'svd':
        # Compute SVD: W_g = U S V^T
        # V^T (Vt) has shape (k, D), where k = min(n_experts, D)
        # The rows of V^T are the right singular vectors (orthonormal)
        # The first 'rank' rows of V^T span the row space of W_g
        U, S, Vt = torch.linalg.svd(target_tensor, full_matrices = False) # Use full_matrices = False for efficiency if D > n_experts

        # Determine rank based on tolerance
        rank = torch.sum(S > svd_tol)
        if rank == 0:
             raise Exception('Router weights matrix has rank 0 according to tolerance.')

        # Basis for the row space (columns of Vr)
        # Vt[:rank] selects the first 'rank' rows (shape rank x D)
        # .T makes it (D x rank) - columns are the orthonormal basis vectors
        Vr = Vt[:rank, :].T

        # Project hidden_states onto the row space (Vr)
        # Formula: h_para = Vr @ Vr^T @ h
        # Batched: H_row = (H @ Vr) @ Vr^T
        # (n_samples, D) @ (D, rank) -> (n_samples, rank)
        h_projected_coeffs = hidden_states @ Vr
        # (n_samples, rank) @ (rank, D) -> (n_samples, D)
        h_para = h_projected_coeffs @ Vr.T

    elif method == 'qr':
        # Compute QR decomposition of W_g^T: W_g^T = Q R
        # Q will have shape (D, k), where k = min(D, n_experts)
        # Columns of Q form an orthonormal basis for column space of W_g^T, which is the row space of W_g.
        Q, R = torch.linalg.qr(target_tensor.T, mode = 'reduced') # Use 'reduced' mode for efficiency

        # Q's columns are the orthonormal basis (shape D x k)
        # Need to consider rank deficiency if applicable, but QR handles it implicitly by the shape of Q returned by 'reduced' mode.

        # Project hidden_states onto the column space of Q
        # Formula: h_para = Q @ Q^T @ h
        # Batched: H_row = (H @ Q) @ Q^T
        # (n_samples, D) @ (D, k) -> (n_samples, k)
        h_projected_coeffs = hidden_states @ Q
        # (n_samples, k) @ (k, D) -> (n_samples, D)
        h_para = h_projected_coeffs @ Q.T

    else:
        raise ValueError('Method must be svd or qr')

    # The orthogonal component is the residual
    h_orth = hidden_states - h_para

    return h_para.to(torch.float16), h_orth.to(torch.float16)


def decompose_sideways(hidden_states: torch.Tensor, target_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decomposes hidden states into components parallel and sideways to the target tensor (such as the routing gate). This should typically be hidden states for those routed to a single expert,
     and the target tensor only the D-dimensional routing gate for that single tensor.

    Params
        @hidden_states: Tensor of shape (n_samples, D) representing the pre-routing hidden states.
        @target_tensor: Target tensor of shape (D,) or (N, D). If a routing gate, a tensor of shape (, D) representing the linear router gate weights for a single expert.

    Returns:
        A tuple containing:
        - h_para (torch.Tensor): Projection onto the row space ("used" by target tensor). Shape (n_samples, D).
        - h_orth (torch.Tensor): Projection onto the orthogonal complement ("unused" by target tensor). Shape (n_samples, D).
    """
    if target_tensor.dim() == 1:
        # same expert for all tokens: expand to (N , D)
        target_tensor = target_tensor.unsqueeze(0).expand_as(hidden_states)

    # normalise each weight vector to unit length
    w_unit = torch.nn.functional.normalize(target_tensor, dim = 1)  # (N , D)

    # dot product per token  ->  (N , 1)
    coeff = (hidden_states * w_unit).sum(dim=1, keepdim=True)

    # projection onto w_E   ->  (N , D)
    h_para_single = coeff * w_unit

    # sideways / orthogonal to *that one* axis
    h_sideways = hidden_states - h_para_single

    return h_para_single, h_sideways
