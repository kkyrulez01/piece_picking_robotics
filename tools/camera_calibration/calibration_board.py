# Get X,Y,Z of the circles centre on the calibration board in terms of the TCP axis
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.cm as cm

# Define board dimensions in mm.  (x0,y0) = top_left_pt, m rows, n cols
BDB6_dict = {'x0': -(150 - 27.5), 'y0': - (38 + 76 + 35 + 105), 'm':5, 'n':4, 'offset_x': 70, 'offset_y': 35, 'offset_x_odd':35 }

class Board:
    def __init__(self, board_dict=BDB6_dict):
        # Board configuration
        self.m  = board_dict['m'] 
        self.n  = board_dict['n']  
        self.x0 = board_dict['x0'] 
        self.y0 = board_dict['y0']  
        self.offset_x = board_dict['offset_x']
        self.offset_y = board_dict['offset_y']
        self.offset_x_odd = board_dict['offset_x_odd']
        
        self.grid_points = None        # (N,2)
        self.board_coordinates = None # (N,3)
        self.worldpoints =  None       # (N,3)
    
    def compute_2D_gridpoints(self):

        origin_x, origin_y = self.x0, self.y0
        grid_points = []

        # along y direction of the tcp (y direction corresponds to rows in BDB6)
        for j in range(self.m):
            y = origin_y  + j * self.offset_y
            x_start = origin_x + (self.offset_x_odd if j % 2 == 1 else 0)
            
            # along x direction of the tcp (x direction corresponds to cols in BDB6), compute the x,y of each of the cols along a specific y
            for i in range(self.n):
                x = x_start + i * self.offset_x
                grid_points.append((x, y))


        self.grid_points = np.array(grid_points, dtype=np.float32)
        return self.grid_points
    
    def simulate_noisy_2D_gridpoints(self, angle_rad, noise_level):
  
        center = self.grid_points.mean(axis=0)
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])
        rotated = (self.grid_points - center) @ R.T + center

        # Add Gaussian noise
        noisy_points = rotated + np.random.normal(scale=noise_level, size=rotated.shape)
        return noisy_points
    
    def variance(self, pts):
        var_x = np.var(pts[:, 0])
        var_y = np.var(pts[:, 1])
        print(f"Variance in X: {var_x:.4f}")
        print(f"Variance in Y: {var_y:.4f}")
        return
    
    def compute_board_coordinates(self):
        if self.board_coordinates is None:
            self.compute_2D_gridpoints()
        self.board_coordinates = np.hstack((self.grid_points, np.zeros((self.grid_points.shape[0], 1))))
        return self.board_coordinates

    def compute_worldpoints(self, R, t):
        """Apply R(Rotation Matrix: 3x3) , t(translation vector: 3x1 ) to each of the pts on the board"""
        
        t = t.reshape(3,) # from (3,1) col vector to (3,) for broadcasting 
        
        if self.board_coordinates is None:
            self.compute_board_coordinates()
        self.worldpoints = self.board_coordinates @ R.T + t  
        return self.worldpoints
    
    def visualise_2D_points(self, pts, fixed_interval=True, padding=0.05):
        plt.figure(figsize=(8, 5))
        plt.scatter(pts[:, 0], pts[:, 1], color='blue')

        for i, (x, y) in enumerate(pts):
            plt.text(x, y, str(i), fontsize=9, ha='right', va='bottom', color='red')

        ax = plt.gca()
        ax.set_aspect('equal')

        if fixed_interval:
            x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
            y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            span = max(x_max - x_min, y_max - y_min) / 2

            # Apply padding
            span *= (1 + padding)

            ax.set_xlim(center_x - span, center_x + span)
            ax.set_ylim(center_y - span, center_y + span)

        plt.title("Grid Pattern with Indices")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()
        return


    def visualize_3d_grid_with_index(self, points_3d, marker='o', title='3D Grid Points', zlim=None):
        """
        Visualize a (M, N, 3) array of 3D points.
        
        Each row is plotted in a unique color. Each point is labeled with its index in the row.

        Parameters:
        - points_3d: numpy array of shape (M, N, 3)
        - marker: marker style for scatter plot
        - title: plot title
        - zlim: optional tuple (zmin, zmax) to set Z-axis limits
        """
        points_3d = np.array(points_3d)
        if points_3d.ndim != 3 or points_3d.shape[2] != 3:
            raise ValueError("Input must be a (M, N, 3) array of 3D points.")

        M, N, _ = points_3d.shape
        colors = cm.get_cmap('tab10', M)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for row_idx in range(M):
            row = points_3d[row_idx]
            xs, ys, zs = row[:, 0], row[:, 1], row[:, 2]
            ax.scatter(xs, ys, zs, color=colors(row_idx), marker=marker, label=f'Row {row_idx}')
            # for pt_idx, (x, y, z) in enumerate(row):
            #     ax.text(x, y, z, f'{pt_idx}', fontsize=8, color=colors(row_idx))

        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if zlim:
            ax.set_zlim(zlim)

        ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    
    def visualize_mulitple_3d_points(self, grid_points1, grid_points2=None, num_of_subplots=1, marker='o', title='3D Point Visualization', zlim=None):
        """
        Visualize one or two sets of 3D grid points across multiple subplots.

        - If only grid_points1 is given: each row in grid_points1 is plotted in a different color.
        - If grid_points2 is also given: grid_points1 is plotted in one color, grid_points2 in another.

        Parameters:
        - grid_points1: list of lists of 3D points (shape M x N x 3)
        - grid_points2: (optional) second list of lists of 3D points (shape K x N x 3)
        - num_of_subplots: number of horizontal subplots to split the rows
        """
        grid1 = np.array(grid_points1)
        if grid1.shape[2] != 3:
            raise ValueError("Each point in grid_points1 must have 3 dimensions (x, y, z).")

        grid2 = None
        if grid_points2 is not None:
            grid2 = np.array(grid_points2)
            if grid2.shape[2] != 3:
                raise ValueError("Each point in grid_points2 must have 3 dimensions (x, y, z).")

        chunks1 = np.array_split(grid1, num_of_subplots, axis=0)
        chunks2 = np.array_split(grid2, num_of_subplots, axis=0) if grid2 is not None else [None] * num_of_subplots

        fig = plt.figure(figsize=(12 * num_of_subplots, 10))

        for i in range(num_of_subplots):
            ax = fig.add_subplot(1, num_of_subplots, i + 1, projection='3d')
            chunk1 = chunks1[i]

            if grid2 is None:
                # Plot each row of grid_points1 in a unique color
                num_rows1 = chunk1.shape[0]
                colors = cm.get_cmap('tab10', num_rows1)
                for row_idx, row in enumerate(chunk1):
                    xs, ys, zs = row[:, 0], row[:, 1], row[:, 2]
                    ax.scatter(xs, ys, zs, color=colors(row_idx), marker=marker, label=f'Row {row_idx}')
            else:
                # Plot all rows of group 1 in the same color
                for row in chunk1:
                    xs, ys, zs = row[:, 0], row[:, 1], row[:, 2]
                    ax.scatter(xs, ys, zs, color='blue', marker=marker, label='Group 1')

                # Plot all rows of group 2 in the same color
                chunk2 = chunks2[i]
                if chunk2 is not None:
                    for row in chunk2:
                        xs, ys, zs = row[:, 0], row[:, 1], row[:, 2]
                        ax.scatter(xs, ys, zs, color='orange', marker=marker, alpha=0.7, label='Group 2')

            ax.set_title(f'{title} - Part {i + 1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            if zlim is not None:
                ax.set_zlim(zlim)
                
            # Clean legend: avoid duplicates
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            
            ax.legend(
                unique.values(), unique.keys(),
                loc='center left', bbox_to_anchor=(1.2, 0.5),
                borderaxespad=0.
            )
            plt.tight_layout()
            plt.show()

# Example usage:
if __name__ == "__main__":
    
    import pandas as pd
    import os
    import random
    from scipy.spatial.transform import Rotation as R
    BDB6 = Board()
    
    # Get 2D grid points
    gridpoints = BDB6.compute_2D_gridpoints()
    print(gridpoints)
    BDB6.visualise_2D_points(gridpoints)
    
    # Plot
    calibration_folder_path = os.path.join("data", "calibration_auto_07may")
    poses_image_folder_path = os.path.join(calibration_folder_path,"images")
    poses_csvfilepath = os.path.join(calibration_folder_path, "poses_recorded.csv")
    df = pd.read_csv(poses_csvfilepath)
    df[['X', 'Y', 'Z']] = df[['X', 'Y', 'Z']] * 1000

    points_3D_1 = []
    points_3D_2 = []
    points_3D_3 = []
    
    poses_with_rotations = ["pose21", "pose22", "pose23", "pose24","pose25","pose26"] 
    poses_with_rotations =  poses_with_rotations+ ["pose6", "pose7", "pose8", "pose9", "pose10", "pose11"]
    poses_with_rotations = poses_with_rotations + ["pose36", "pose37", "pose38", "pose39", "pose40", "pose41"]


    filenames = df['filename'].tolist()
    for text_filename in filenames:
        
        # # select layers
        # if int(text_filename.replace(".png",'').replace("pose",'')) > 15 and int(text_filename.replace(".png",'').replace("pose",'')) <= 30:
        #     pass
        # else:
        #     print("skip")
        #     continue
        
        # # skip 
        # if (text_filename.replace(".png",'') in ["pose19","pose20", "pose27"]):
        #     #print("skip")
        #     continue
        
        # skip rotation
        if (text_filename.replace(".png",'') in poses_with_rotations):
            #print("skip")
            continue
        
        row = df[df['filename'] == text_filename.replace(".txt", '.png')].iloc[0]
        x, y, z, Rx, Ry, Rz = row[['X', 'Y', 'Z','Rx', 'Ry', 'Rz']].values
        
        R_tcp2base = R.from_euler('xyz', [Rx, Ry, Rz]).as_matrix()  # 3x3 rotation matrix
        translation = np.array([x, y, z])
        
        #
        R_board2tcp = R.from_euler('z', 43 + 90 , degrees=True).as_matrix()  # 3x3 rotation matrix

        
        # Apply transformation matrix to BDB6 grid points
        points_tcp = BDB6.compute_board_coordinates()
        #print(points_tcp)
        points_3D_1.append(points_tcp)
    
        points_base_3D = BDB6.compute_worldpoints(R_tcp2base, [0,0,0])
        points_3D_2.append(points_base_3D)
        
        points_base_3D = BDB6.compute_worldpoints(R_tcp2base @ R_board2tcp,  translation)
        points_3D_3.append(points_base_3D)
    
    BDB6.visualize_3d_grid_with_index(points_3D_3, marker='o', title='3D Point Visualization', zlim=(0,400))
