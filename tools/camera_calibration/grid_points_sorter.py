import numpy as np
from sklearn.decomposition import PCA
import random

# Finds the order of the grid points, using PCA, given a (N,2) array of noisy_pts. Sorted in the order left to right, then top to bottom .

class GridPointsSorter:
    def __init__(self, noisy_points):
        
        self.noisy_points = noisy_points
        self.pca = None

    def run_PCA(self):
        # PCA projection
        self.pca = PCA(n_components=2)
        self.projected = self.pca.fit_transform(self.noisy_points)

        # Randomly flip PCA axes
        PC1_flip = random.choice([-1, 1])
        self.projected[:, 0] *= PC1_flip
        self.pca.components_[0] *= PC1_flip

        PC2_flip = random.choice([-1, 1])
        self.projected[:, 1] *= PC2_flip
        self.pca.components_[1] *= PC2_flip
        return 

    def sort_group_by_PC_value(self, top_left_pt_projected, PC_index=1, group_size=4):
        N = self.projected.shape[0]

        new_order = np.argsort(self.projected[:, PC_index])
        self.noisy_points = self.noisy_points[new_order]
        self.projected = self.projected[new_order]

        top_left_pt_projected_found = False
        resort_reverse = False

        row_num = 0
        for i in range(0, N, group_size):
            row_num += 1
            row = self.projected[i:i+group_size]
            if top_left_pt_projected in row:
                top_left_pt_projected_found = True
                if row_num == 1:
                    print(f"Good. The top left point is found in row 1. No need to re-sort")
                elif row_num == (int(N / group_size)):
                    resort_reverse = True
                    print(f"Good. The top left point is found in the last row. Re-sort needed")
                else:
                    print(f"Warning: The top left point is found in row {row_num} and not in row 1 or last row.")
                break

        if resort_reverse:
            new_order = np.argsort(self.projected[:, PC_index])[::-1]
            self.noisy_points = self.noisy_points[new_order]
            self.projected = self.projected[new_order]

        if not top_left_pt_projected_found:
            print("Warning: The top left point is not found in any of the rows.")
        
        return new_order, self.projected, self.noisy_points

    def sort_within_group(self, top_left_pt_projected, PC_index=0, group_size=4):
        warning_detected = False

        row = self.projected[:group_size]
        indices_row = np.argsort(row[:, PC_index])
        sorted_row = row[indices_row]

        if np.array_equal(top_left_pt_projected, sorted_row[0]):
            print("\n\n Good. Top left pt is found in the left most position of the row")
            sort_by_ascending = True
        elif np.array_equal(top_left_pt_projected, sorted_row[-1]):
            print("\n\nGood. Top left pt is found in the right most position of the row")
            #indices_row = indices_row[::-1]
            sort_by_ascending = False
        else:
            print("Warning: Top left point is neither found as the first or last element after sorting along PC2 in row 1")
            warning_detected = True

        if not warning_detected:
            self.projected, self.noisy_points = self.reorder_each_chunk(self.projected, self.noisy_points, group_size, sort_by_ascending, PC_index)
        return self.projected, self.noisy_points

    def reorder_each_chunk(self, pts, pts2, chunk_size, sort_by_ascending, sort_axis):
        reordered_pts = []
        reordered_pts2 = []

        # Split both pts and pts2 into chunks
        chunks1 = [pts[i:i+chunk_size] for i in range(0, len(pts), chunk_size)]
        chunks2 = [pts2[i:i+chunk_size] for i in range(0, len(pts2), chunk_size)]

        for chunk1, chunk2 in zip(chunks1, chunks2):
            # Get indices that would sort chunk1 by the chosen coordinate
            indices = sorted(
                range(len(chunk1)),
                key=lambda i: chunk1[i][sort_axis],
                reverse=not sort_by_ascending
            )

            # Apply same order to both chunks
            sorted_chunk1 = [chunk1[i] for i in indices]
            sorted_chunk2 = [chunk2[i] for i in indices]

            reordered_pts.extend(sorted_chunk1)
            reordered_pts2.extend(sorted_chunk2)

        return np.array(reordered_pts), np.array(reordered_pts2)

    def print_sorted_results(self, group_size):
        N = self.projected.shape[0]
        for i in range(0, N, group_size):
            row_num = i // group_size + 1
            row_pts = self.projected[i:i+group_size]
            print(f"row {row_num} has pts: {', '.join(str(pt) for pt in row_pts)}")
            
  
            
if __name__ == "__main__":
    
    from calibration_board import Board, plt

    # Simulate Grid
    BDB6_dict_simulation = {'x0':10, 'y0':10, 'm':5, 'n':4, 'offset_x':2, 'offset_y':1, 'offset_x_odd':1 }
    BDB6 = Board(BDB6_dict_simulation)
    grid_points = BDB6.compute_2D_gridpoints()

    # Add rotation then noise
    noise_level = 0
    angle_deg = np.random.uniform(0, 360)
    angle_rad = np.radians(angle_deg)
    noisy_points = BDB6.simulate_noisy_2D_gridpoints(angle_rad, noise_level)
    
    # Run PCA    
    grid_order_finder = GridPointsSorter(noisy_points)
    grid_order_finder.run_PCA()
    
    pca = grid_order_finder.pca
    projected = grid_order_finder.projected

    # assuming that top_left_idx is known from the noisy rotated picture, get the projected coordinates of that pt
    top_left_idx = np.lexsort((grid_points[:, 0], grid_points[:, 1]))[0]
    top_left_pt_projected = projected[top_left_idx]
    print(top_left_pt_projected)
    
    # # Simulate the randomness of the pts being given
    # shuffle_idx = np.random.permutation(len(projected))
    # projected = projected[shuffle_idx]
    # noisy_points = noisy_points[shuffle_idx]
    
    # # Find the new top_left_idx(used for ploting) after the shuffle
    # matches = np.all(projected == top_left_pt_projected, axis=1)
    # top_left_idx = np.where(matches)[0][0]
    # grid_points = grid_points[shuffle_idx]
    
    # Parameter for sorting
    group_size = 4
    
    # Sort (partially) by grouping the points using PC2 value
    _ , sorted_projected, sorted_noisy_points = grid_order_finder.sort_group_by_PC_value(top_left_pt_projected, PC_index=1, group_size=4)
    grid_order_finder.print_sorted_results(group_size)
    
    # Sort within the groups to produce the final fully sorted points.
    sorted_projected, sorted_noisy_points = grid_order_finder.sort_within_group(top_left_pt_projected, PC_index=0, group_size=4)
    grid_order_finder.print_sorted_results(group_size)
    
    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Original staggered grid
    axs[0].scatter(grid_points[:, 0], grid_points[:, 1], color='lightblue')
    axs[0].scatter(*grid_points[top_left_idx], color='red', s=100, label='Top-Left')
    for i, (x_, y_) in enumerate(grid_points):
        axs[0].text(x_, y_, str(i), fontsize=8, ha='center', va='center')
    axs[0].invert_yaxis()
    axs[0].set_title("1. Staggered Grid (Image Coordinates)")
    axs[0].axis('equal')
    axs[0].grid(True)
    axs[0].legend()

    # Rotated + noisy
    axs[1].scatter(noisy_points[:, 0], noisy_points[:, 1], color='gray')
    axs[1].scatter(*noisy_points[top_left_idx], color='lime', s=100, edgecolor='black', label='Top-Left')
    for i, (x_, y_) in enumerate(noisy_points):
        axs[1].text(x_, y_, str(i), fontsize=8, ha='center', va='center')
    axs[1].invert_yaxis()

    # ➕ PCA axes
    origin = noisy_points.mean(axis=0)
    scale = 2.0
    for i, color in zip(range(2), ['orange', 'purple']):
        vec = scale * pca.components_[i]
        axs[1].arrow(origin[0], origin[1], vec[0], vec[1],
                        head_width=0.2, head_length=0.2, fc=color, ec=color, linewidth=2,
                        label=f'PC{i+1}')

    axs[1].set_title(f"2. Rotated + Noisy + PCs ({angle_deg:.1f}°)")
    axs[1].axis('equal')
    axs[1].grid(True)
    axs[1].legend()


    # PCA projected
    axs[2].scatter(projected[:, 0], projected[:, 1], color='skyblue')
    axs[2].scatter(*projected[top_left_idx], color='red', s=100, edgecolor='black', label='Top-Left in PCA')
    for i, (x_, y_) in enumerate(projected):
        axs[2].text(x_, y_, str(i), fontsize=8, ha='center', va='center')
    axs[2].set_title("3. PCA Projected")
    axs[2].set_xlabel("PC1")
    axs[2].set_ylabel("PC2")
    axs[2].axhline(0, color='k', linestyle='--', alpha=0.3)
    axs[2].axvline(0, color='k', linestyle='--', alpha=0.3)
    axs[2].axis('equal')
    axs[2].grid(True)
    axs[2].legend()

    plt.suptitle("Staggered Grid → Rotated → PCA Projected", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    