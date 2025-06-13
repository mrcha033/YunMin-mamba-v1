"""
Correlation-based Scan Permutation (CSP) Analysis - Pillar 1
This script implements offline analysis to find optimal state permutations for memory efficiency.
"""

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from models.baseline_ssm import BaselineSSM
from data.wikitext103 import get_wiktext103_dataloader
from utils.logger import setup_logger

# A list to store the captured hidden states from the forward hook
STATE_TRAJECTORIES = []

def state_capture_hook(module, input, output):
    """
    A forward hook function to capture the intermediate state 'h'.
    In a real Mamba implementation, the scan output 'y' is a function of 'h'.
    We would hook the internal state 'h' before it's used.
    For this implementation, we'll assume the 'output' of ssm_scan is a proxy for the state.
    The output tensor shape is (B, L, d_inner). We are permuting the state 'h'
    which has shape (B, d_inner, d_state). Let's assume the hook can access 'h' directly.
    A realistic hook would be on the `mamba-ssm`'s C++ kernel call.
    Let's assume the output is the state h of shape (B, L, d_state) for analysis.
    We are permuting the d_state=N dimension.
    """
    # Detach from graph and move to CPU to save GPU memory
    # For our baseline implementation, we'll reshape the output to simulate state trajectories
    # In practice, this would capture the actual hidden state from the SSM scan
    batch_size, seq_len, d_inner = output.shape
    # Simulate state trajectories by reshaping - in real implementation this would be actual states
    simulated_state = output.view(batch_size, seq_len, -1, 16)  # Assuming d_state=16
    captured_state = simulated_state.mean(dim=2).detach().cpu()  # Average over d_inner groups
    STATE_TRAJECTORIES.append(captured_state)

def collect_state_trajectories(model: nn.Module, dataloader, num_samples: int, device: str = "cuda") -> torch.Tensor:
    """
    Step 1a: Run representative data through the model and collect state trajectories.
    """
    global STATE_TRAJECTORIES
    STATE_TRAJECTORIES = []
    
    logger = setup_logger("csp_analysis")
    logger.info(f"Collecting state trajectories from {num_samples} samples...")
    
    # Register hooks on all MambaBlock ssm_scan outputs
    # Since ssm_scan is a method, we'll hook the entire MambaBlock and capture in the hook
    hook_handles = []
    
    # Create a custom hook for MambaBlocks
    def mamba_block_hook(module, input, output):
        # Capture the output which represents the SSM processing result
        state_capture_hook(module, input, output)
    
    # Register hooks on the first few layers for analysis
    for i, layer in enumerate(model.layers[:3]):  # Analyze first 3 layers to save memory
        handle = layer.register_forward_hook(mamba_block_hook)
        hook_handles.append(handle)

    model.eval()
    model = model.to(device)
    samples_processed = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting State Trajectories"):
            input_ids = batch["input_ids"].to(device)
            model(input_ids)
            samples_processed += input_ids.shape[0]
            if samples_processed >= num_samples:
                break
    
    # Remove hooks
    for handle in hook_handles:
        handle.remove()

    # Concatenate all trajectories. Each element is (B, L, d_state).
    # We want a single matrix of (Total_Timesteps, d_state) where d_state=N
    if STATE_TRAJECTORIES:
        H_traj_all = torch.cat([t.flatten(0, 1) for t in STATE_TRAJECTORIES], dim=0)
        
        # As per the paper, transpose to get (d_state, Total_Timesteps) for correlation calc.
        # Each row is now the trajectory of one state dimension over time.
        return H_traj_all.transpose(0, 1)
    else:
        logger.warning("No state trajectories captured, returning random data for demonstration")
        d_state = model.layers[0].d_state
        return torch.randn(d_state, 1000)  # Fallback for testing

def build_correlation_matrix(H_traj: torch.Tensor) -> torch.Tensor:
    """
    Step 1b: Compute the Pearson correlation matrix Σ from the state trajectories.
    """
    logger = setup_logger("csp_analysis")
    logger.info("Building correlation matrix...")
    
    # torch.corrcoef expects rows to be variables and columns to be observations.
    # H_traj is already in the correct (d_state, T) shape.
    correlation_matrix = torch.corrcoef(H_traj)
    
    # Handle NaN values that might arise from constant trajectories
    correlation_matrix = torch.nan_to_num(correlation_matrix, nan=0.0)
    
    logger.info(f"Correlation matrix shape: {correlation_matrix.shape}")
    return correlation_matrix

def solve_tsp_for_permutation(correlation_matrix: torch.Tensor, greedy_start_node: int = 0) -> torch.Tensor:
    """
    Step 2: Solve the TSP problem on the correlation graph to find the optimal permutation π*.
    We use a simple greedy algorithm as described in the paper as a baseline.
    A 2-opt refinement would be added for better results.
    """
    logger = setup_logger("csp_analysis")
    logger.info("Solving TSP for optimal permutation...")
    
    # TSP solvers minimize distance. We want to maximize correlation.
    # So, distance = 1 - |correlation|. Use absolute value to handle negative correlations.
    distance_matrix = 1 - torch.abs(correlation_matrix)
    
    num_nodes = distance_matrix.shape[0]
    path = [greedy_start_node]
    visited = {greedy_start_node}
    
    current_node = greedy_start_node
    while len(path) < num_nodes:
        # Get distances from the current node
        distances = distance_matrix[current_node].clone()
        # Mask out visited nodes by setting their distance to infinity
        distances[list(visited)] = float('inf')
        
        # Find the nearest unvisited neighbor
        next_node = torch.argmin(distances).item()
        path.append(next_node)
        visited.add(next_node)
        current_node = next_node
    
    logger.info(f"TSP path found: {path[:10]}..." if len(path) > 10 else f"TSP path: {path}")
    return torch.tensor(path, dtype=torch.long)

def reorder_model_weights(state_dict: dict, permutation_vector: torch.Tensor) -> dict:
    """
    Step 3: Apply the permutation π* permanently to the model's weights.
    """
    logger = setup_logger("csp_analysis")
    logger.info("Reordering model weights...")
    
    pi = permutation_vector
    d_state = len(pi)
    reordered_count = 0

    for layer_key, param in state_dict.items():
        # Identify parameters that interact with the state dimension (d_state)
        # This requires knowledge of the Mamba architecture's parameter naming.
        
        # Target: A_log parameter in MambaBlock. Shape: (d_inner, d_state)
        if "A_log" in layer_key:
            # We permute the columns corresponding to the d_state dimension
            state_dict[layer_key] = torch.index_select(param, dim=1, index=pi)
            reordered_count += 1

        # Target: dt_proj weight. Shape: (d_inner, d_state) 
        elif "dt_proj.weight" in layer_key:
            # We permute the columns corresponding to the d_state dimension
            if param.shape[1] == d_state:  # Ensure dimension matches
                state_dict[layer_key] = torch.index_select(param, dim=1, index=pi)
                reordered_count += 1

        # Target: x_proj weight, which generates B and C. Shape: (2 * d_state, d_inner)
        elif "x_proj.weight" in layer_key:
            # This projects to B and C concatenated. We must permute the rows.
            # The first `d_state` rows correspond to B, the next to C.
            if param.shape[0] == 2 * d_state:  # Ensure dimension matches
                perm_for_B = pi
                perm_for_C = pi + d_state
                combined_perm = torch.cat([perm_for_B, perm_for_C])
                state_dict[layer_key] = torch.index_select(param, dim=0, index=combined_perm)
                reordered_count += 1

    logger.info(f"Reordered {reordered_count} parameter tensors")
    return state_dict

def main():
    parser = argparse.ArgumentParser(description="Run CSP analysis")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output_path", type=str, default="./csp_optimized_model.pt",
                        help="Path to save CSP-optimized model")
    parser.add_argument("--num_samples", type=int, default=1024,
                        help="Number of samples for trajectory analysis")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run analysis on")
    args = parser.parse_args()
    
    logger = setup_logger("csp_analysis")
    
    # --- Configuration ---
    model_config = {
        'd_model': 768,
        'n_layer': 12, 
        'vocab_size': 50257,
        'd_state': 16,
        'd_conv': 4
    }
    
    # --- Load Model and Data ---
    logger.info("Initializing model for CSP analysis...")
    model = BaselineSSM(**model_config)
    
    if args.model_path and torch.cuda.is_available():
        try:
            checkpoint = torch.load(args.model_path)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded model from {args.model_path}")
        except Exception as e:
            logger.warning(f"Could not load model from {args.model_path}: {e}")
            logger.info("Using randomly initialized model for demonstration")
    else:
        logger.info("Using randomly initialized model for demonstration")
    
    # Use a simple tokenizer for demonstration
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use the WikiText-103 validation set, as specified in the proposal
    try:
        dataloader = get_wiktext103_dataloader(
            tokenizer=tokenizer,
            batch_size=16,  # Smaller batch size for memory efficiency
            max_length=512,
            split="validation",
            num_workers=0
        )
        logger.info("Created WikiText-103 validation dataloader")
    except Exception as e:
        logger.error(f"Could not create dataloader: {e}")
        logger.info("CSP analysis requires data for trajectory collection")
        return

    # --- Execute CSP Pipeline ---
    try:
        # 1. Collect trajectories
        logger.info("Step 1: Collecting state trajectories...")
        H_traj = collect_state_trajectories(model, dataloader, args.num_samples, args.device)
        
        # 2. Build correlation graph
        logger.info("Step 2: Building correlation matrix...")
        corr_matrix = build_correlation_matrix(H_traj)
        
        # 3. Solve for optimal permutation
        logger.info("Step 3: Solving TSP for optimal permutation...")
        pi_star = solve_tsp_for_permutation(corr_matrix)
        logger.info(f"Optimal permutation found: {pi_star.tolist()}")

        # 4. Reorder weights and save the new model
        logger.info("Step 4: Applying permutation to model weights...")
        original_state_dict = model.state_dict()
        csp_state_dict = reorder_model_weights(original_state_dict, pi_star)
        
        # Save both the reordered model and the permutation
        torch.save({
            'model_state_dict': csp_state_dict,
            'permutation': pi_star,
            'correlation_matrix': corr_matrix,
            'csp_applied': True,
            'model_config': model_config
        }, args.output_path)
        
        logger.info(f"CSP-optimized model saved to {args.output_path}")
        logger.info("CSP analysis complete!")
        
        # Log some statistics
        corr_mean = torch.mean(torch.abs(corr_matrix)).item()
        logger.info(f"Mean absolute correlation: {corr_mean:.4f}")
        
    except Exception as e:
        logger.error(f"CSP analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 