import torch
import gc

def compare_bf16_fp16_batched(tensor_bf16, tensor_fp16, batch_size = 50_000, verbose = True):
    """
    Compares large BF16 and FP16 tensors in batches using CUDA for computation, calculating statistics for both and checking for infinities in the FP16 tensor.
    This is useful for checking if bf16 -> f16 casts do not cause any loss, especially inf casts.

    Params:
        @tensor_bf16 (torch.Tensor): The input tensor in bfloat16 format.
        @tensor_fp16 (torch.Tensor): The input tensor in float16 format.
        @batch_size (int): The number of elements from the first dimension to process in each batch.
        @verbose (bool): If True, prints progress and summary info.

    Returns:
        tuple: A tuple containing:
            - stats (dict): A dictionary containing comparison statistics.
            - total_inf_count (int): The total number of infinite values found in the input FP16 tensor.
    """
    compute_device = torch.device('cuda')

    if tensor_bf16.dtype != torch.bfloat16:
        raise ValueError(f"Expected tensor_bf16 to have dtype torch.bfloat16, but got {tensor_bf16.dtype}")
    if tensor_fp16.dtype != torch.float16:
        raise ValueError(f"Expected tensor_fp16 to have dtype torch.float16, but got {tensor_fp16.dtype}")
    if tensor_bf16.shape != tensor_fp16.shape:
        raise ValueError(f"Input tensors must have the same shape, but got {tensor_bf16.shape} and {tensor_fp16.shape}")

    n_samples = tensor_bf16.shape[0]
    shape = tensor_bf16.shape

    if verbose:
        print(f"Starting batched comparison:")
        print(f"  Tensor Shape: {shape}")
        print(f"  BF16 Device: {tensor_bf16.device}, FP16 Device: {tensor_fp16.device}")
        print(f"  Compute Device: {compute_device}, Batch Size: {batch_size}")

    total_inf_count = 0
    total_processed_elements = 0

    # --- Accumulators (on CPU for stability and low sync overhead) ---
    bf16_sum = torch.tensor(0.0, dtype = torch.float64, device = 'cpu')
    bf16_sum_sq = torch.tensor(0.0, dtype = torch.float64, device = 'cpu')
    overall_bf16_max = torch.tensor(-float('inf'), dtype = torch.float32, device = 'cpu')
    overall_bf16_min = torch.tensor(float('inf'), dtype = torch.float32, device = 'cpu')

    fp16_sum = torch.tensor(0.0, dtype = torch.float64, device = 'cpu')
    fp16_sum_sq = torch.tensor(0.0, dtype = torch.float64, device = 'cpu')
    overall_fp16_max = torch.tensor(-float('inf'), dtype = torch.float32, device = 'cpu')
    overall_fp16_min = torch.tensor(float('inf'), dtype = torch.float32, device = 'cpu')
    # --- End Accumulators ---

    for i in range(0, n_samples, batch_size):
        start_idx = i
        end_idx = min(i + batch_size, n_samples)
        batch_slice = slice(start_idx, end_idx)

        # 1. Get batches & move to CUDA
        #    Handles source tensors being on CPU or GPU
        batch_bf16_gpu = tensor_bf16[batch_slice].to(compute_device, non_blocking = True)
        batch_fp16_gpu = tensor_fp16[batch_slice].to(compute_device, non_blocking = True)

        # --- Process BF16 Batch ---
        batch_bf16_fp32_gpu = batch_bf16_gpu.float()
        bf16_batch_sum_cpu = torch.sum(batch_bf16_fp32_gpu).to('cpu', non_blocking = True)
        bf16_batch_sum_sq_cpu = torch.sum(batch_bf16_fp32_gpu**2).to('cpu', non_blocking = True)
        bf16_batch_max_cpu = torch.max(batch_bf16_gpu).cpu()
        bf16_batch_min_cpu = torch.min(batch_bf16_gpu).cpu()

        bf16_sum += bf16_batch_sum_cpu.double()
        bf16_sum_sq += bf16_batch_sum_sq_cpu.double()
        overall_bf16_max = torch.maximum(overall_bf16_max, bf16_batch_max_cpu.float())
        overall_bf16_min = torch.minimum(overall_bf16_min, bf16_batch_min_cpu.float())
        # Increment element count only once per loop iteration
        elements_in_batch = batch_bf16_gpu.numel()
        total_processed_elements += elements_in_batch
        # --- End BF16 Processing ---

        # --- Process FP16 Batch ---
        fp16_batch_fp32_gpu = batch_fp16_gpu.float() # Use float for stable sum
        fp16_batch_sum_cpu = torch.sum(fp16_batch_fp32_gpu).to('cpu', non_blocking = True)
        fp16_batch_sum_sq_cpu = torch.sum(fp16_batch_fp32_gpu**2).to('cpu', non_blocking = True)
        fp16_batch_max_cpu = torch.max(batch_fp16_gpu).cpu() # Max/min directly is fine
        fp16_batch_min_cpu = torch.min(batch_fp16_gpu).cpu()

        fp16_sum += fp16_batch_sum_cpu.double()
        fp16_sum_sq += fp16_batch_sum_sq_cpu.double()
        overall_fp16_max = torch.maximum(overall_fp16_max, fp16_batch_max_cpu.float())
        overall_fp16_min = torch.minimum(overall_fp16_min, fp16_batch_min_cpu.float())

        # Check infinities in the FP16 batch
        inf_in_batch = torch.isinf(batch_fp16_gpu).sum().item()
        total_inf_count += inf_in_batch
        # --- End FP16 Processing ---

        # --- Cleanup ---
        del batch_bf16_gpu, batch_fp16_gpu
        del batch_bf16_fp32_gpu, fp16_batch_fp32_gpu
        del bf16_batch_sum_cpu, bf16_batch_sum_sq_cpu, bf16_batch_max_cpu, bf16_batch_min_cpu
        del fp16_batch_sum_cpu, fp16_batch_sum_sq_cpu, fp16_batch_max_cpu, fp16_batch_min_cpu
        # gc.collect() # Optional
        torch.cuda.empty_cache()
        # --- End Cleanup ---

    # --- Final Statistics Calculation ---
    stats = {}
    # BF16 Stats
    stats['bf16_max'] = overall_bf16_max.item()
    stats['bf16_min'] = overall_bf16_min.item()
    if total_processed_elements > 0:
        bf16_mean = bf16_sum / total_processed_elements
        bf16_var = torch.clamp((bf16_sum_sq / total_processed_elements) - (bf16_mean**2), min=0.0)
        stats['bf16_mean'] = bf16_mean.item()
        stats['bf16_std'] = torch.sqrt(bf16_var).item()
    else: stats['bf16_mean'], stats['bf16_std'] = float('nan'), float('nan')

    # FP16 Stats
    stats['fp16_max'] = overall_fp16_max.item()
    stats['fp16_min'] = overall_fp16_min.item()
    if total_processed_elements > 0:
        if total_inf_count == 0: # Only calculate mean/std if no infinities
             fp16_mean = fp16_sum / total_processed_elements
             fp16_var = torch.clamp((fp16_sum_sq / total_processed_elements) - (fp16_mean**2), min=0.0)
             stats['fp16_mean'] = fp16_mean.item()
             stats['fp16_std'] = torch.sqrt(fp16_var).item()
        else:
             stats['fp16_mean'], stats['fp16_std'] = float('nan'), float('nan')
    else: stats['fp16_mean'], stats['fp16_std'] = float('nan'), float('nan')
    # --- End Statistics Calculation ---

    if verbose:
        print("\n--- Comparison Summary ---")
        print(f"Compared {total_processed_elements} elements.")
        print(f"Total Infinite Values Found in FP16 Tensor: {total_inf_count}")
        print(f"          | {'BF16 Input':<15} | {'FP16 Input':<15}")
        print(f"Max Value | {stats.get('bf16_max', 'N/A'):<15.6g} | {stats.get('fp16_max', 'N/A'):<15.6g}")
        print(f"Min Value | {stats.get('bf16_min', 'N/A'):<15.6g} | {stats.get('fp16_min', 'N/A'):<15.6g}")
        print(f"Mean      | {stats.get('bf16_mean', 'N/A'):<15.6g} | {stats.get('fp16_mean', 'N/A'):<15.6g}")
        print(f"Std Dev   | {stats.get('bf16_std', 'N/A'):<15.6g} | {stats.get('fp16_std', 'N/A'):<15.6g}")
        print("----------------------------")

    gc.collect()
    return stats, total_inf_count
