
import os
import pandas as pd
import re
import numpy as np
import cv2

class CalibrationFolderManager:
    def __init__(self, calibration_folder_path):
        self.base_folder = calibration_folder_path
        self.images_folder = os.path.join(calibration_folder_path, "images")
        self.depth_folder = os.path.join(calibration_folder_path, "images_depth")
        self.record_path = os.path.join(calibration_folder_path, "poses_recorded.csv")
        
        self.imgpoints_folder = os.path.join(calibration_folder_path, "imgpoints")
        
        self.setup_folders()
        self.clean_pose_records()

    def setup_folders(self):
        os.makedirs(self.images_folder, exist_ok=True)
        os.makedirs(self.depth_folder, exist_ok=True)

    def clean_pose_records(self):
        
        if os.path.exists(self.record_path):
            
            df = pd.read_csv(self.record_path)
            # try:
            #     df = pd.read_csv(self.record_path)
            # except pd.errors.EmptyDataError:
            #     os.remove(self.record_path)
            #     return
            recorded_filenames = df['filename'].tolist()
            df_updated = pd.DataFrame()

            # for each image found, check if it exists in the csv and its corresponding depth image is found.
            for filename in os.listdir(self.images_folder):
                depth_filename = filename.replace("pose", "poseDepth").replace(".png", ".tiff")
                image_path = os.path.join(self.images_folder, filename)
                depth_path = os.path.join(self.depth_folder, depth_filename)

                
                if os.path.isfile(depth_path) and (filename in recorded_filenames):
                    df_updated = pd.concat([df_updated, df.loc[df['filename'] == filename]],axis=0)
                    #print(f"Record: {filename} is found and corresponding Depth image is found.")
                    
                else:
                    print(f"Record: {filename} is not found or corresponding Depth image is not found. Removing...")
                    if os.path.isfile(depth_path):
                        os.remove(depth_path)
                    if os.path.isfile(image_path):
                        os.remove(image_path)
            
            # remove any depth image without corresponding image
            for depth_filename in os.listdir(self.depth_folder):
                depth_path = os.path.join(self.depth_folder, depth_filename)
                image_filename = depth_filename.replace('Depth','').replace('.tiff','.png')
                
                if not (image_filename in os.listdir(self.images_folder)):
                    print(f"cleaning up depth folder: removing file:{image_filename}")
                    os.remove(depth_path)
            
            # remove csv if no image found at all, i.e num of rows in dataframe=0, to prevent empty dataframe read error
            if len(df_updated) == 0:
                os.remove(self.record_path)
            else:
                df_updated.to_csv(self.record_path, index=False)

    def get_next_filenames(self):
        def get_highest_pose_number():
            max_num = 0
            pattern = re.compile(r'pose(\d+)\.png$', re.IGNORECASE)
            for filename in os.listdir(self.images_folder):
                match = pattern.match(filename)
                if match:
                    num = int(match.group(1))
                    max_num = max(max_num, num)
            return max_num

        num_saved = len(os.listdir(self.images_folder))
        if num_saved:
            highest_num = get_highest_pose_number()
            return f"pose{highest_num + 1}.png", f"poseDepth{highest_num + 1}.tiff"
        else:
            return "pose1.png", "poseDepth1.tiff"
        
    def save_robot_pose(self, pose_data, filename):
        """
        Save robot TCP pose with respect to base frame into the poses record CSV.
        Inputs: pose_data (list[float]): List of 6 floats [X, Y, Z, Rx, Ry, Rz].
        """
        if len(pose_data) != 6:
            raise ValueError("Pose data must be a list of 6 values: [X, Y, Z, Rx, Ry, Rz]")

        row_data = [filename] + pose_data
        df_new = pd.DataFrame([row_data], columns=['filename', 'X', 'Y', 'Z', 'Rx', 'Ry', 'Rz'])

        if not os.path.exists(self.record_path):
            df_new.to_csv(self.record_path, index=False)
        
        else:
            df_old = pd.read_csv(self.record_path)
        
            if filename in df_old['filename'].values:
                df_old = df_old[df_old['filename'] != filename]  # Remove all old instances(may be multiple)
                df_old = pd.concat([df_old, df_new], axis=0)
                df_old.to_csv(self.record_path, index=False)
            else:
                df_old = pd.concat([df_old, df_new], axis=0)
                df_old.to_csv(self.record_path, index=False)
                
    def is_robot_pose_in_record(self, X, Y, Z, Rx, Ry, Rz):
        if os.path.exists(self.record_path):
            
            df = pd.read_csv(self.record_path)
            target = np.array([X, Y, Z, Rx, Ry, Rz])
            tolerance = 1e-3  # Adjust as needed
            mask = np.isclose(df[['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']], target, atol=tolerance).all(axis=1)  # Check each row
            matching_rows = df[mask]

            # Check if any match found
            is_inside = len(matching_rows) > 0
            return is_inside
            
        else:
            return False
                     
    def get_calibration_data(self, refined=False, exclude=[]):
        """Return imgpoints, tcp_pose, filename, image_paths, image_depth_paths"""
        
        if not (os.path.exists(self.record_path)):
            raise RuntimeError("Must get the tcp pose data as csv first.")
        else:
            df = pd.read_csv(self.record_path)
            
        list_of_imgpoints = []
        list_of_depths = []
        list_of_tcp_pose = []
        filenames = []
        image_paths = []
        image_depth_paths = []

        text_filenames = [i for i in os.listdir(self.imgpoints_folder) if i.endswith(".txt")] 
            
        for text_filename in text_filenames:
            
            if int(text_filename.replace(".txt",'').replace("pose", '')) in exclude:
                continue
            
            # get tcp 
            row = df[df['filename'] == text_filename.replace(".txt", '.png')].iloc[0]
            x, y, z, Rx, Ry, Rz = row[['X', 'Y', 'Z','Rx', 'Ry', 'Rz']].values
            x, y, z = x*1000, y*1000, z*1000
            list_of_tcp_pose.append([x,y,z,Rx,Ry,Rz])
            
            # get imgpoints for the image
            imgpoints = np.loadtxt(os.path.join(self.imgpoints_folder, text_filename))
            list_of_imgpoints.append(imgpoints)
            
            # get file paths for (image, depth)
            filenames.append(text_filename.replace(".txt", ''))
            image_paths.append(os.path.join(self.images_folder, text_filename.replace(".txt", ".png")))
            depth_filepath = os.path.join(self.depth_folder, text_filename.replace(".txt", ".tiff").replace("pose","poseDepth"))
            image_depth_paths.append(depth_filepath)
            
            # get the depth values
            img = cv2.imread(depth_filepath, cv2.IMREAD_UNCHANGED) 
            depths = []
            for imgpoint in imgpoints: 
                x,y = int(imgpoint[0]), int(imgpoint[1])
                depth = img[y, x]
                depths.append(depth)
            list_of_depths.append(depths)
        
        
        print(f"\n{len(text_filenames)} poses with grid correctly detected. {len(list_of_imgpoints)} loaded after excluding {len(exclude)}.")
        return(list_of_imgpoints, list_of_depths, list_of_tcp_pose, filenames, image_paths, image_depth_paths)
        
    
            

            
            