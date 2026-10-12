import subprocess

# folder_names = ["./data/mechmind_logMleft_25jun"]
folder_names = ["/home/support/unseen_sku_ws/tools/camera_calibration/calibration_data/realsense_D455f_260922"]

def run_scripts(calibration_folder_path):

    cmds = [#f"python find_camera_parameters.py {calibration_folder_path}", 
            f"python3 tcp_to_board_alignment.py {calibration_folder_path}", 
            f"python3 find_extrinsic.py {calibration_folder_path}", 
            #f"python compute_show_error.py {calibration_folder_path} reprojections extrinsic.npz", 
            f"python3 optimise.py {calibration_folder_path}", 
            f"python3 compute_show_error.py {calibration_folder_path} reprojections_opt optimised.npz", 
            ]
    
    for cmd in cmds:
        print(f"\n\n ................................Running {cmd}................................")
        subprocess.run(cmd.split() ,check=True)
        
if __name__ == "__main__":
    
    for calibration_folder_path in folder_names:
        run_scripts(calibration_folder_path)