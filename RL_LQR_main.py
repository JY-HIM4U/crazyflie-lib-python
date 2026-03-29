#!/usr/bin/env python3
"""
Main entry point for RL+SMPC Training Pipeline

This script implements a hierarchical control system where:
1. RL agent (PPO) generates high-level velocity commands (vx, vy, vz)
2. SMPC generates low-level control actions (thrust, roll_rate, pitch_rate, yaw_rate)
3. Phoenix simulator executes the control actions

This version is modified for simplified parallel training using multiprocessing Pool.

🚀 PERFORMANCE OPTIMIZATIONS:
✅ Vectorized value estimation: 2 batched forward passes instead of 128 individual ones
✅ GPU-accelerated PPO updates with proper tensor batching
✅ Simplified parallel training with CPU workers + GPU learner
✅ Efficient multiprocessing via Pool (no file I/O)
✅ Wandb integration for learning trajectory tracking
✅ Timestamped model saving

🌪️ DISTURBANCE MODELING:
✅ Configurable position disturbance (x, y) during physics simulation
✅ Command line argument control for easy on/off switching
✅ Adjustable disturbance strength via --disturbance-strength
✅ Realistic environmental noise integrated into physics loop
✅ Disturbance statistics logged to wandb for analysis

USAGE EXAMPLES:
    # Train without disturbance (default)
    python RL_smpc_main.py
    
    # Train with disturbance enabled
    python RL_smpc_main.py --disturbance
    
    # Train with custom disturbance strength
    python RL_smpc_main.py --disturbance --disturbance-strength 1e-3
    
    # Train with custom parameters
    python RL_smpc_main.py --disturbance --num-workers 4 --episodes-per-worker 10
    
    # Show all available options
    python RL_smpc_main.py --help
"""

import sys
import os
import argparse
import numpy as np
import torch

# Fix argv[0] issue by setting it properly
if not sys.argv:
    sys.argv = ['RL_smpc_main.py']
elif not sys.argv[0]:
    sys.argv[0] = 'RL_smpc_main.py'

# Import our modularized components
from RL_smpc_config import *
from RL_smpc_utils import setup_device
from RL_smpc_world_map import WorldMap
from RL_LQR_parallel_training import run_parallel_training


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RL+SMPC Training with Position Disturbance')
    
    # Disturbance arguments
    parser.add_argument('--disturbance', action='store_true', default=False,
                       help='Enable position disturbance (default: False)')
    parser.add_argument('--disturbance-strength', type=float, default=1e-4,
                       help='Position disturbance strength (default: 5e-4)')
    parser.add_argument('--disturbance-verbose', action='store_true', default=False,
                       help='Enable verbose disturbance logging (default: False)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model to load (default: None)')

    # Wind arguments (region-dependent wind force in physics)
    parser.add_argument('--wind', action='store_true', default=False,
                       help='Enable region-dependent wind disturbance (default: False)')
    parser.add_argument('--wind-verbose', action='store_true', default=False,
                       help='Verbose per-step wind logging (default: False)')
    parser.add_argument('--wind-mean-frac-min', type=float, default=WIND_MEAN_FRAC_MIN,
                       help='Min mean wind magnitude as fraction of weight m*g')
    parser.add_argument('--wind-mean-frac-max', type=float, default=WIND_MEAN_FRAC_MAX,
                       help='Max mean wind magnitude as fraction of weight m*g')
    parser.add_argument('--wind-sigma-frac-min', type=float, default=WIND_SIGMA_FRAC_MIN,
                       help='Min wind sigma as fraction of weight m*g')
    parser.add_argument('--wind-sigma-frac-max', type=float, default=WIND_SIGMA_FRAC_MAX,
                       help='Max wind sigma as fraction of weight m*g')

    # Belief (Bayes) arguments: append 3D belief to observation (44 -> 47)
    parser.add_argument(
        '--belief',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Append Bayes belief (3D) to observation (default: True). Use --no-belief to disable.',
    )
    # Training arguments
    parser.add_argument('--num-workers', type=int, default=12,
                       help='Number of worker processes (default: auto-detect)')
    parser.add_argument('--episodes-per-worker', type=int, default=16,
                       help='Number of episodes per worker (default: 10)')
    parser.add_argument('--save-interval', type=int, default=1,
                       help='Model save interval (default: 10)')
    parser.add_argument('--train-continuous-path', type=str, default=None,
                       help='Load saved model path (default: None)')
    # Environment arguments
    parser.add_argument('--rl-decision-interval', type=float, default=1.0,
                       help='RL decision interval in seconds (default: 1.0)')
    parser.add_argument('--mpc-planning-interval', type=float, default=0.025,
                       help='MPC planning interval in seconds (default: 0.025)')
    parser.add_argument('--physics-time-step', type=float, default=0.005,
                       help='Physics time step in seconds (default: 0.005)')
    parser.add_argument('--episode-length', type=int, default=15,
                       help='Episode length in RL steps (default: 15)')
    
    return parser.parse_args()


def main():
    """Main entry point: launches the simplified parallel training."""
    print("🚀 Starting Simplified Parallel RL+SMPC Training with World Map (Multi-Process Workers + GPU Learner)")
    print(f"   State format: 44D base (+ optional 2D belief -> 46D)")
    print(f"   Action format: 2D [vx, vy] horizontal velocity commands")
    print(f"   Model save directory: {MODEL_SAVE_DIR}")
    print(f"   Wandb enabled: {WANDB_ENABLED}")
    print("")
    print("🏗️  Architecture:")
    print("   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐")
    print("   │   N Workers     │───▶│  1 GPU Learner  │───▶│  1 Policy Model │")
    print("   │ (CPU + SMPC)    │    │   (PPO + NN)    │    │   (Shared)      │")
    print("   └─────────────────┘    └─────────────────┘    └─────────────────┘")
    print("   • Collect experience  • Updates policy    • Single model file")
    print("   • Run in parallel     • Uses all exp      • Gets better over time")
    print("   • No file I/O         • Direct data return • Cleaner workflow")
    print("")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Update global configuration based on command line arguments
    global RL_DECISION_INTERVAL, MPC_PLANNING_INTERVAL, PHYSICS_TIME_STEP, EPISODE_LENGTH
    global ENABLE_POSITION_DISTURBANCE, POSITION_DISTURBANCE_STRENGTH, DISTURBANCE_VERBOSE
    # Import wind default from config so we can respect it unless user explicitly passes --wind
    from RL_smpc_config import ENABLE_WIND_DISTURBANCE
    from RL_smpc_config import WIND_VERBOSE
    
    RL_DECISION_INTERVAL = args.rl_decision_interval
    model_path = args.train_continuous_path
    MPC_PLANNING_INTERVAL = args.mpc_planning_interval
    PHYSICS_TIME_STEP = args.physics_time_step
    EPISODE_LENGTH = args.episode_length
    ENABLE_POSITION_DISTURBANCE = args.disturbance
    POSITION_DISTURBANCE_STRENGTH = args.disturbance_strength
    DISTURBANCE_VERBOSE = args.disturbance_verbose

    # Use config default unless user explicitly enables wind via CLI
    wind_enabled = ENABLE_WIND_DISTURBANCE if not args.wind else True
    wind_verbose = WIND_VERBOSE if not args.wind_verbose else True
    wind_mean_frac_min = args.wind_mean_frac_min
    wind_mean_frac_max = args.wind_mean_frac_max
    wind_sigma_frac_min = args.wind_sigma_frac_min
    wind_sigma_frac_max = args.wind_sigma_frac_max
    belief_enabled = bool(args.belief)
    
    # Setup device (GPU/CPU)
    DEVICE = setup_device()
    
    # Create model save directory ONLY in main process
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print(f"📁 Created model save directory: {MODEL_SAVE_DIR}")
    
    # Initialize world map and get target position from goal center
    print("🗺️  Loading world map...")
    WORLD_MAP = WorldMap(world_name=WORLD_MAP_NAME)
    TARGET_POSITION = WORLD_MAP.goal_center.copy()
    # TARGET_POSITION= np.array([0.3, 1.0, 1.0])
    
    print(f"   World map: {WORLD_MAP.world_name}.png ({WORLD_MAP.img_width}×{WORLD_MAP.img_height})")
    print(f"   Start position: {START_POSITION}")
    print(f"   Target position: {TARGET_POSITION}")
    
    print("🗺️  World Map Features:")
    print("   • Local grid encoding: 5×5 grid around drone position")
    print(f"   • Grid resolution: {WORLD_MAP.grid_resolution:.1f}m per pixel (10×10 image → 0~2m world)")
    print("   • Coordinate system: Pixel (0,0) → World (0.0, 2.0), Pixel (9,9) → World (1.8, 0.0)")
    print("   • Continuous space: Drone can be at any position, not just grid points")
    print("   • Interpolation: Uses bilinear interpolation for non-grid positions")
    print("   • Avoid areas: Red channel in PNG")
    print("   • Target areas: Blue channel in PNG")
    print("   • Safe areas: Black areas (navigable)")
    print("   • Goal reward: +5.0, Avoid penalty: -5.0")
    print("")
    
    # Configuration from command line arguments
    num_workers = args.num_workers if args.num_workers is not None else DEFAULT_NUM_WORKERS
    episodes_per_worker = args.episodes_per_worker
    save_interval = args.save_interval
    
    print(f"   Configuration: num_workers={num_workers}, episodes_per_worker={episodes_per_worker}, save_interval={save_interval}")
    print(f"   Result: {num_workers} workers contribute to 1 policy model (not {num_workers} separate models)")
    print("")
    print("🌪️  Disturbance Configuration:")
    print(f"   - Position disturbance: {'ENABLED' if ENABLE_POSITION_DISTURBANCE else 'DISABLED'}")
    print(f"   - Disturbance strength: {POSITION_DISTURBANCE_STRENGTH}")
    print(f"   - Position noise std: {np.sqrt(POSITION_DISTURBANCE_STRENGTH):.6f} m per physics step")
    print(f"   - Physics time step: {PHYSICS_TIME_STEP:.3f}s")
    print(f"   - Disturbance per second: {np.sqrt(POSITION_DISTURBANCE_STRENGTH/PHYSICS_TIME_STEP):.6f} m/s")
    print("")

    print("🌬️  Wind Configuration:")
    print(f"   - Wind disturbance: {'ENABLED' if wind_enabled else 'DISABLED'}")
    print(f"   - Mean |F| fraction of m*g: [{wind_mean_frac_min}, {wind_mean_frac_max}]")
    print(f"   - Sigma fraction of m*g: [{wind_sigma_frac_min}, {wind_sigma_frac_max}]")
    print("")

    print("🧠 Belief Configuration:")
    print(f"   - Belief appended to observation: {'ENABLED (46D)' if belief_enabled else 'DISABLED (44D)'}")
    print("")
    print("🎯 Training Objectives:")
    print(f"   • Start from position: {START_POSITION[:2]} (continuous world coordinates)")
    print(f"   • Navigate to goal center: {TARGET_POSITION[:2]}")
    print(f"   • Avoid unsafe areas (termination with -5.0 reward)")
    print(f"   • Reach goal area (termination with +5.0 reward)")
    print(f"   • Stay in safe areas during navigation")
    print(f"   • World coordinates: 0.0m to 2.0m (continuous space, 0.2m grid for PNG)")
    print("")
    
    # Initialize wandb if enabled
    if WANDB_ENABLED:
        try:
            import wandb
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                config={
                    "rl_decision_interval": RL_DECISION_INTERVAL,
                    "mpc_planning_interval": MPC_PLANNING_INTERVAL,
                    "physics_time_step": PHYSICS_TIME_STEP,
                    "start_position": START_POSITION,
                    "target_position": TARGET_POSITION.tolist(),
                    "episode_length": EPISODE_LENGTH,
                    "device": str(DEVICE),
                    "model_save_dir": MODEL_SAVE_DIR,
                    "timestamp": TIMESTAMP,
                    "architecture": "simplified_parallel_training",
                    "description": "N CPU workers collect experience, 1 GPU learner updates single policy model",
                    "world_map": WORLD_MAP.world_name,
                    "world_dimensions": [WORLD_MAP.img_width, WORLD_MAP.img_height],
                    "grid_resolution": WORLD_MAP.grid_resolution,
                    "state_dimensions": TOTAL_OBS_DIM,
                    "local_grid_size": LOCAL_GRID_DIM
                },
                name=f"simplified_parallel_training_{WORLD_MAP.world_name}_{TIMESTAMP}",
                tags=["simplified_parallel", "single_policy", "phoenix_drone", "world_map", WORLD_MAP.world_name]
            )
            print(f"✅ Wandb initialized successfully. Project: {WANDB_PROJECT}")
            print(f"   Run name: simplified_parallel_training_{TIMESTAMP}")
            print(f"   Architecture: Simplified parallel workers → Single policy model")
        except ImportError:
            print("⚠️  Wandb not installed. Install with: pip install wandb")
        except Exception as e:
            print(f"⚠️  Failed to initialize wandb: {e}")
    else:
        print("ℹ️  Wandb disabled. Set WANDB_ENABLED=true to enable.")
    
    # Make globals available to other modules
    globals().update({
        'WORLD_MAP': WORLD_MAP,
        'TARGET_POSITION': TARGET_POSITION,
        'DEVICE': DEVICE
    })
    
    # Launch the simplified parallel training
    if (model_path is not None):
        run_parallel_training(
            num_workers=num_workers, 
            episodes_per_worker=episodes_per_worker, 
            save_interval=save_interval,
            world_map=WORLD_MAP,
            target_position=TARGET_POSITION,
            model_path=model_path,
            disturbance_enabled=ENABLE_POSITION_DISTURBANCE,
            disturbance_strength=POSITION_DISTURBANCE_STRENGTH,
            wind_enabled=wind_enabled,
            wind_verbose=wind_verbose,
            wind_mean_frac_min=wind_mean_frac_min,
            wind_mean_frac_max=wind_mean_frac_max,
            wind_sigma_frac_min=wind_sigma_frac_min,
            wind_sigma_frac_max=wind_sigma_frac_max,
            belief_enabled=belief_enabled,
        )
    else:
        run_parallel_training(
            num_workers=num_workers, 
            episodes_per_worker=episodes_per_worker, 
            save_interval=save_interval,
            world_map=WORLD_MAP,
            target_position=TARGET_POSITION,
            disturbance_enabled=ENABLE_POSITION_DISTURBANCE,
            disturbance_strength=POSITION_DISTURBANCE_STRENGTH,
            wind_enabled=wind_enabled,
            wind_verbose=wind_verbose,
            wind_mean_frac_min=wind_mean_frac_min,
            wind_mean_frac_max=wind_mean_frac_max,
            wind_sigma_frac_min=wind_sigma_frac_min,
            wind_sigma_frac_max=wind_sigma_frac_max,
            belief_enabled=belief_enabled,
        )
    
    print("✅ Training finished.")


if __name__ == "__main__":
    main() 