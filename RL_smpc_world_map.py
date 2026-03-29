#!/usr/bin/env python3
"""
World map handling for RL+SMPC Training Pipeline
Handles safe/avoid/target set detection and coordinate transformations
"""

import numpy as np
import imageio.v3 as iio
from RL_smpc_config import START_POSITION, GRID_RESOLUTION, EPISODE_LENGTH

class WorldMap:
    """World map handler for safe/avoid/target set detection"""
    
    def __init__(self, world_name="ra_10"):
        """
        Initialize world map from PNG file
        
        Args:
            world_name: Name of the world file (without .png extension)
        """
        self.world_name = world_name
        self.load_world_map()
        
    def load_world_map(self):
        """Load world map from PNG and process safe/avoid/target sets"""
        # Load PNG image
        im = iio.imread(f"/home/realm/Jaeyoun/worlds/{self.world_name}.png")
        
        # Extract red (Safe) and blue (target) channels from PNG
        im_r = np.array(im[:, :, 0]) / 255.0  # Safe area (red channel)
        im_b = np.array(im[:, :, 2]) / 255.0  # Target area (blue channel)
        
        # Store dimensions
        self.img_height, self.img_width = im.shape[:2]
        
        # Use PNG channels as intended: red=safe, blue=target, black=avoid
        self.safe_set = im_r
        self.target_set = im_b - im_r  # Remove overlap with avoid set
        self.avoid_set = 1.0 - (im_r + self.target_set)  # Everything else (black areas)
        
        # YOUR COORDINATE SYSTEM: Image (x,y) → Physical [x*0.2, (x+1)*0.2] × [y*0.2, (y+1)*0.2]
        # 10×10 image → 0~2.0m physical coordinates
        # Each pixel represents 0.2m × 0.2m physical region
        self.grid_resolution = GRID_RESOLUTION  # meters per pixel
        
        print(f"🌍 World map loaded: {self.world_name}.png ({self.img_width}×{self.img_height})")
        print(f"   Grid resolution: {self.grid_resolution:.1f}m per pixel")
        print(f"   Physical coordinates: 0.0m to 2.0m (continuous space)")
        print(f"   Coordinate system: Image(x,y) → Physical[x*0.2,(x+1)*0.2]×[y*0.2,(y+1)*0.2]")
        
        # Find center of goal set for target position
        self.goal_center = self.find_goal_center()
        print(f"   Goal center: {self.goal_center}")
        
        # Debug coordinate mapping for the 10×10 image
        self.debug_coordinate_mapping()
        
        # Show raw pixel values for debugging
        self.show_pixel_values()
        
        # Print ASCII map representation
        self.print_map(START_POSITION)
        
    def find_goal_center(self):
        """Find the center of the goal set in world coordinates using YOUR coordinate system"""
        # Find all goal pixels
        goal_pixels = np.where(self.target_set > 0)
        
        if len(goal_pixels[0]) == 0:
            print("⚠️  No goal set found in world map!")
            return np.array([0.1, 1.6, 1.0])  # Default to center of leftmost column
        
        # Convert each goal pixel to world coordinates using YOUR system
        world_coords = []
        unique_rows = np.unique(goal_pixels[0])  # [0, 1]
        unique_cols = np.unique(goal_pixels[1])  # [7, 8, 9]
        unique_rows = np.append(unique_rows, np.max(unique_rows) + 1)  # [0, 1, 2]
        unique_cols = np.append(unique_cols, np.max(unique_cols) + 1)
        for i in range(len(unique_rows)):
            for j in range(len(unique_cols)):
                pixel_y = unique_rows[i]  # Row (Y)
                pixel_x = unique_cols[j]  # Col (X)
                world_x = pixel_x * self.grid_resolution  # Direct conversion: image x to physical x
                world_y = pixel_y * self.grid_resolution  # Direct conversion: image y to physical y
                world_coords.append([world_x, world_y])
        
        # Calculate centroid from world coordinates
        world_coords = np.array(world_coords)
        center_world_x = np.mean(world_coords[:, 0])
        center_world_y = np.mean(world_coords[:, 1])
        
        # Ensure coordinates are within bounds (0.0m to 2.0m)
        center_world_x = np.clip(center_world_x, 0.0, 2.0)
        center_world_y = np.clip(center_world_y, 0.0, 2.0)
        
        print(f"   🔍 Goal center calculation (YOUR coordinate system):")
        print(f"      Goal pixels found: {len(goal_pixels[0])} pixels")
        print(f"      Goal pixel coordinates:")
        for i in range(len(goal_pixels[0])):
            pixel_y, pixel_x = goal_pixels[0][i], goal_pixels[1][i]
            world_x, world_y = world_coords[i]
            print(f"         Image({pixel_x},{pixel_y}) → Physical({world_x:.1f}m, {world_y:.1f}m)")
        print(f"      Physical centroid: ({center_world_x:.1f}m, {center_world_y:.1f}m)")
        
        return np.array([center_world_x, center_world_y, 1.0])  # Z fixed at 1m
    
    def world_to_image_coords(self, world_pos):
        """Convert world coordinates to image pixel coordinates using YOUR coordinate system"""
        # YOUR SYSTEM: Physical [x*0.2, (x+1)*0.2] × [y*0.2, (y+1)*0.2] → Image (x,y)
        # Physical: 0.0m to 2.0m → Image: 0 to 9 (continuous)
        # Example: Physical (0.1m, 1.8m) → Image (0, 9)
        # Formula: img_x = world_x / 0.2, img_y = world_y / 0.2
        img_x = world_pos[0] / self.grid_resolution  # Direct conversion: physical x to image x
        img_y = world_pos[1] / self.grid_resolution  # Direct conversion: physical y to image y
        
        # Clamp to image bounds for interpolation purposes
        # Note: Out-of-bounds detection is handled separately in check_area_membership_interpolated()
        img_x = np.clip(img_x, 0, self.img_width - 1)
        img_y = np.clip(img_y, 0, self.img_height - 1)
        
        return img_x, img_y
    
    def world_to_image_coords_discrete(self, world_pos):
        """Convert world coordinates to discrete image pixel coordinates using YOUR coordinate system"""
        # YOUR SYSTEM: Physical coordinates → Image coordinates
        # Formula: img_x = int(world_x / 0.2), img_y = int(world_y / 0.2)
        img_x = int(world_pos[0] / self.grid_resolution)  # Direct conversion: physical x to image x
        img_y = int(world_pos[1] / self.grid_resolution)  # Direct conversion: physical y to image y
        
        # Clamp to image bounds
        img_x = np.clip(img_x, 0, self.img_width - 1)
        img_y = np.clip(img_y, 0, self.img_height - 1)
        
        return img_x, img_y
    
    def image_to_world_coords(self, img_x, img_y):
        """Convert image pixel coordinates to world coordinates using YOUR coordinate system"""
        # YOUR SYSTEM: Image (x,y) → Physical [x*0.2, (x+1)*0.2] × [y*0.2, (y+1)*0.2]
        # Image: 0 to 9 → Physical: 0.0m to 2.0m
        # Example: Image (0,9) → Physical [0.0~0.2m] × [1.8~2.0m]
        world_x = img_x * self.grid_resolution  # Direct conversion: image x to physical x
        world_y = img_y * self.grid_resolution  # Direct conversion: image y to physical y
        
        return world_x, world_y
    
    def encode_local_grid(self, drone_pos, grid_size=5):
        """
        Encode a local grid around drone position
        
        Args:
            drone_pos: [x, y] drone position in world coordinates
            grid_size: Size of local grid (grid_size × grid_size)
            
        Returns:
            25-dimensional vector: 0=avoid, 1=safe, 2=target
        """
        # print(f"      Debug encode_local_grid: drone_pos={drone_pos}, type={type(drone_pos)}, shape={drone_pos.shape if hasattr(drone_pos, 'shape') else 'no shape'}")
        
        local_grid = np.zeros((grid_size, grid_size))
        
        # print(f"      Debug: Calling world_to_image_coords_discrete...")
        center_x, center_y = self.world_to_image_coords_discrete(drone_pos)
        # print(f"      Debug: Got center_x={center_x}, center_y={center_y}")
        
        # Grid covers area around drone
        grid_radius = int((grid_size - 1) / 2)
        # print(f"      Debug: grid_radius={grid_radius}, img_height={self.img_height}, img_width={self.img_width}")
        
        for i in range(grid_size):
            for j in range(grid_size):
                # print(f"      Debug: Processing grid cell ({i}, {j})")
                
                # Calculate world coordinates for this grid cell
                if (center_y - (grid_radius-i)) < 0:
                    local_grid[i, j] = 0  # Avoid area
                    # print(f"        Debug: Case 1 - y out of bounds")
                elif (center_y - (grid_radius-i)) >= self.img_height:
                    local_grid[i, j] = 0  # Avoid area
                    # print(f"        Debug: Case 2 - y out of bounds")
                elif (center_x - (grid_radius-j)) < 0:
                    local_grid[i, j] = 0  # Avoid area
                    # print(f"        Debug: Case 3 - x out of bounds")
                elif (center_x - (grid_radius-j)) >= self.img_width:
                    local_grid[i, j] = 0  # Avoid area
                    # print(f"        Debug: Case 4 - x out of bounds")
                else:
                    # print(f"        Debug: Accessing arrays at y={center_y + (i - grid_radius)}, x={center_x + (j - grid_radius)}")
                    if self.target_set[center_y + (i - grid_radius), center_x + (j - grid_radius)] == 1:
                        local_grid[i,j]= 2 # Target area
                        # print(f"        Debug: Target area")
                    elif self.safe_set[center_y + (i - grid_radius), center_x + (j - grid_radius)] == 1:
                        local_grid[i,j]= 1 # Safe area
                        # print(f"        Debug: Safe area")
                    else:
                        local_grid[i,j]= 0 # Avoid area
                        # print(f"        Debug: Avoid area")
        
        flattened = local_grid.flatten()  # 25-dimensional vector
        return flattened
    
    def check_position_status(self, world_pos):
        """
        Check if position is in safe, avoid, or target set
        
        Args:
            world_pos: [x, y, z] position in world coordinates
            
        Returns:
            dict with keys: 'status', 'reward', 'terminate', 'message'
        """
        pos_xy = world_pos[:2]
        
        # Check if position is out of bounds first
        if self.is_out_of_bounds(pos_xy):
            return {
                'status': 'avoid',
                'reward': -10.0,
                'terminate': True,
                'message': 'out_of_bounds'
            }
        
        # Use interpolated area membership for more accurate results
        area_values = self.check_area_membership_interpolated(pos_xy)
        
        # Determine the dominant area type
        safe_val = area_values['safe']
        target_val = area_values['target']
        avoid_val = area_values['avoid']
        
        # Use threshold-based classification
        threshold = 0.5  # Threshold for area membership
        
        if avoid_val > threshold:
            # print("Reached avoid", pos_xy)
            return {
                'status': 'avoid',
                'reward': -10.0,
                'terminate': True,
                'message': 'hit_avoid_set'
            }
        elif target_val > threshold:
            
            return {
                'status': 'target',
                'reward': +10.0,
                'terminate': True,
                'message': 'reached_target'
            }
        elif safe_val > threshold:
            # print("Reached safe", pos_xy)
            return {
                'status': 'safe',
                'reward': +0.0,
                'terminate': False,
                'message': 'in_safe_area'
            }
        else:
            # Mixed area - use the highest value
            max_val = max(safe_val, target_val, avoid_val)
            if max_val == safe_val:
                return {
                    'status': 'safe',
                    'reward': +0.0,
                    'terminate': False,
                    'message': 'in_mixed_safe_area'
                }
            elif max_val == target_val:
                return {
                    'status': 'target',
                    'reward': +10.0,
                    'terminate': True,
                    'message': 'reached_target'
                }
            else:
                return {
                    'status': 'avoid',
                    'reward': -10.0,
                    'terminate': True,
                    'message': 'hit_avoid_set'
                }
    def get_position_status_with_cur(self, pos_next, pos_cur, action, step):
        pos_xy_next = pos_next[:2]
        pos_xy_cur = pos_cur[:2]
        pos_status = self.check_position_status(pos_next)
        reward = pos_status['reward']
        done = pos_status['terminate']
        if reward != -10:
            for i in range(10):
                alpha = i / 9.0  # 0.0 to 1.0
                interpolated_pos = pos_cur + alpha * (pos_next - pos_cur)
                pos_status = self.check_position_status(interpolated_pos)
                if pos_status['status']=='avoid':
                    reward -=8.0
                    break
        
        if not done:
            dist_cur = np.linalg.norm(pos_xy_cur - self.goal_center[:2])
            dist_next = np.linalg.norm(pos_xy_next - self.goal_center[:2])
            
            # Reward for making progress towards the target
            reward_progress = (dist_cur - dist_next) * 10.0 # Reward for getting closer, penalize for moving away
            
            # Small penalty for time to encourage a direct path
            reward_time_penalty = -0.01 
            
            # Small penalty for action magnitude to encourage smooth movements
            reward_action_penalty = -1 * np.linalg.norm(action) 
            
            # Total reward for the step
            reward += reward_action_penalty  # + reward_progress + reward_time_penalty 
            # dist_cur = np.linalg.norm(pos_cur - target_position)
            # dist_next = np.linalg.norm(pos_next - target_position)
            # survive_reward_val = 0.0
            # distance_reward = float(- dist_next)
            # action_penalty = - 1.0 * np.linalg.norm(action) 
            # reward +=  action_penalty # distance_reward + survive_reward_val + action_penalty
            
            height_violation = pos_next[2] < 0.01  # Decreased from 0.05
            last_step = (step == EPISODE_LENGTH - 1)
            
            done = bool(
                height_violation or
                last_step
            )
            
        return reward, done

    def check_area_membership_interpolated(self, world_pos):
        """
        Check area membership using bilinear interpolation for non-integer coordinates
        
        Args:
            world_pos: [x, y] position in world coordinates
            
        Returns:
            dict with interpolated values for safe, target, and avoid sets
        """
        # Convert to image coordinates (can be non-integer)
        img_x, img_y = self.world_to_image_coords(world_pos)
        
        # Check if position is completely out of bounds (before clamping)
        # Calculate raw image coordinates without clamping
        raw_img_x = world_pos[0] / self.grid_resolution
        raw_img_y = world_pos[1] / self.grid_resolution
        
        # If completely out of bounds, return avoid area
        if (raw_img_x < 0 or raw_img_x >= self.img_width or 
            raw_img_y < 0 or raw_img_y >= self.img_height):
            return {
                'safe': 0.0,
                'target': 0.0,
                'avoid': 1.0  # Out of bounds = avoid area
            }
        
        # Get the four surrounding pixels for bilinear interpolation
        x0, y0 = int(np.floor(img_x)), int(np.floor(img_y))
        x1, y1 = min(x0 + 1, self.img_width - 1), min(y0 + 1, self.img_height - 1)
        
        # Calculate interpolation weights
        wx = img_x - x0
        wy = img_y - y0
        
        # Ensure we don't go out of bounds
        x0 = max(0, min(x0, self.img_width - 1))
        y0 = max(0, min(y0, self.img_height - 1))
        x1 = max(0, min(x1, self.img_width - 1))
        y1 = max(0, min(y1, self.img_height - 1))
        
        # Bilinear interpolation for each set
        safe_val = (1 - wx) * (1 - wy) * self.safe_set[y0, x0] + \
                   wx * (1 - wy) * self.safe_set[y0, x1] + \
                   (1 - wx) * wy * self.safe_set[y1, x0] + \
                   wx * wy * self.safe_set[y1, x1]
        
        target_val = (1 - wx) * (1 - wy) * self.target_set[y0, x0] + \
                     wx * (1 - wy) * self.target_set[y0, x1] + \
                     (1 - wx) * wy * self.target_set[y1, x0] + \
                     wx * wy * self.target_set[y1, x1]
        
        avoid_val = (1 - wx) * (1 - wy) * self.avoid_set[y0, x0] + \
                    wx * (1 - wy) * self.avoid_set[y0, x1] + \
                    (1 - wx) * wy * self.avoid_set[y1, x0] + \
                    wx * wy * self.avoid_set[y1, x1]
        
        return {
            'safe': safe_val,
            'target': target_val,
            'avoid': avoid_val
        }
    
    def is_out_of_bounds(self, world_pos):
        """
        Check if a world position is completely out of the image bounds
        
        Args:
            world_pos: [x, y] position in world coordinates
            
        Returns:
            bool: True if out of bounds, False if within bounds
        """
        # Calculate raw image coordinates without clamping
        raw_img_x = world_pos[0] / self.grid_resolution
        raw_img_y = world_pos[1] / self.grid_resolution
        
        # Check if completely out of bounds
        return (raw_img_x < 0 or raw_img_x > self.img_width or 
                raw_img_y < 0 or raw_img_y > self.img_height)
    
    def get_target_position(self):
        """Get the target position from the goal center"""
        return self.goal_center.copy()
    
    def debug_coordinate_mapping(self):
        """Debug coordinate mapping for the 10×10 image using YOUR coordinate system"""
        print("\n🔍 Debug Coordinate Mapping (YOUR System):")
        print("   Image Grid Layout (10×10):")
        print("   YOUR Coordinate System: Image(x,y) → Physical[x*0.2,(x+1)*0.2]×[y*0.2,(y+1)*0.2]")
        
        # Show image grid with coordinates
        for y in range(self.img_height):
            row_str = "   "
            for x in range(self.img_width):
                world_x, world_y = self.image_to_world_coords(x, y)
                
                # Determine area type
                if self.target_set[y, x] > 0:
                    symbol = "🟦"  # Target
                elif self.safe_set[y, x] > 0:
                    symbol = "⬛"  # Safe (black)
                else:
                    symbol = "🟥"  # Avoid (red)
                
                row_str += f"{symbol}({world_x:.1f},{world_y:.1f}) "
            print(row_str)
        
        # Test some specific coordinates
        test_positions = [
            [0.1, 1.6],  # Goal center (leftmost column)
            [0.1, 0.1],  # Bottom-left corner
            [1.9, 1.9],  # Top-right corner
            [1.0, 1.0],  # Center
        ]
        
        print("\n   Test Coordinate Mapping:")
        for pos in test_positions:
            img_coords = self.world_to_image_coords(pos)
            area_values = self.check_area_membership_interpolated(pos)
            
            print(f"   Physical({pos[0]:.1f}m, {pos[1]:.1f}m) → Image({img_coords[0]:.2f}, {img_coords[1]:.2f})")
            print(f"     Safe: {area_values['safe']:.3f}, Target: {area_values['target']:.3f}, Avoid: {area_values['avoid']:.3f}")
        
        # Show target position
        print(f"\n   🎯 Target Position: {self.goal_center[:2]}")
        target_img_coords = self.world_to_image_coords(self.goal_center[:2])
        print(f"     Maps to image coordinates: ({target_img_coords[0]:.2f}, {target_img_coords[1]:.2f})")
    
    def show_pixel_values(self):
        """Show the actual pixel values from the PNG file for debugging"""
        print(f"\n🔍 Raw Pixel Values from {self.world_name}.png:")
        print("   Red Channel (Avoid Areas):")
        for y in range(self.img_height):
            row_str = "   "
            for x in range(self.img_width):
                val = int(self.avoid_set[y, x] * 255)
                row_str += f"{val:3d} "
            print(row_str)
        
        print("\n   Blue Channel (Target Areas):")
        for y in range(self.img_height):
            row_str = "   "
            for x in range(self.img_width):
                val = int(self.target_set[y, x] * 255)
                row_str += f"{val:3d} "
            print(row_str)
        
        print("\n   Safe Areas (Black):")
        for y in range(self.img_height):
            row_str = "   "
            for x in range(self.img_width):
                val = int(self.safe_set[y, x] * 255)
                row_str += f"{val:3d} "
            print(row_str)
    
    def print_map(self, current_pos):
        """Print a simple ASCII representation of the world map using YOUR coordinate system"""
        print(f"\n🗺️  World Map: {self.world_name}.png ({self.img_width}×{self.img_height})")
        print("   Legend: 🟥=Avoid, 🟦=Target, ⬛=Safe, 🟢=Current, ⭐=Goal")
        print("   YOUR Coordinate System: Image(x,y) → Physical[x*0.2,(x+1)*0.2]×[y*0.2,(y+1)*0.2]")
        print("   Grid: 0.2m per pixel, Physical: 0.0m to 2.0m")
        print()
        
        # Print the map with symbols
        for y in range(self.img_height):
            row_str = f"{y:2d} "
            for x in range(self.img_width):
                world_x, world_y = self.image_to_world_coords(x, y)
                
                # Determine area type
                if self.target_set[y, x] > 0:
                    symbol = "🟦"  # Target
                elif self.safe_set[y, x] > 0:
                    symbol = "⬛"  # Safe
                else:
                    symbol = "🟥"  # Avoid
                
                # Check if this is goal position
                if abs(world_x - self.goal_center[0]) < 0.1 and abs(world_y - self.goal_center[1]) < 0.1:
                    symbol = "⭐"  # Goal
                elif abs(world_x - current_pos[0]) < 0.1 and abs(world_y - current_pos[1]) < 0.1:
                    symbol = "🟢"  # Current Position
                
                row_str += symbol
            print(row_str)
        
        # Print X coordinates (physical coordinates)
        x_coords = "   "
        for x in range(self.img_width):
            world_x = x * self.grid_resolution  # YOUR system: direct conversion
            x_coords += f"{world_x:3.1f}"
        print(x_coords)
        
        # Print Y coordinates (physical coordinates)
        y_coords = "   "
        for y in range(self.img_height):
            world_y = y * self.grid_resolution  # YOUR system: direct conversion
            y_coords += f"{world_y:3.1f}"
        print(y_coords)
        
        print(f"\n   ⭐ Goal:  {self.goal_center[:2]} (Physical coordinates)")
        print(f"   Grid: {self.grid_resolution}m per pixel")
        print(f"   Physical range: [0.0m, 2.0m] × [0.0m, 2.0m]")

        encode_local_grid = self.encode_local_grid(current_pos)
        local_grid = encode_local_grid.reshape(5,5)
        for y in range(5):
            row_str = f"{y:2d} "
            for x in range(5):
                # Determine area type
                if local_grid[y, x] == 2:
                    symbol = "🟦"  # Target
                elif local_grid[y, x] == 1:
                    symbol = "⬛"  # Safe
                else:
                    symbol = "🟥"  # Avoid
                
                row_str += symbol
            print(row_str) 

    def test_position(self):
        for y in range(100):
            for x in range(100):
                world_x, world_y = x*0.02, y*0.02
                aa= self.check_position_status([world_x, world_y, 1.0])
                if aa['status'] == 'target':
                    print("!",end="")
                elif aa['status'] == 'safe':
                    print(".",end="")
                else:
                    print("*",end="")
            print()
            
