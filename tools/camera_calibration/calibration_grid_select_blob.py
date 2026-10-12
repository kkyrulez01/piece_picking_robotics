import cv2
import numpy as np
import os
from grid_points_sorter import GridPointsSorter

calibration_data_dir_path = "/home/support/unseen_sku_ws/tools/camera_calibration/calibration_data"
calibration_folder_path = os.path.join(calibration_data_dir_path, "realsense_D455f_260922")


image_folder_path = os.path.join(calibration_folder_path, "images")
labelled_image_folder_path = os.path.join(calibration_folder_path, "images_grid_detected")
imgpoints_folder_path = os.path.join(calibration_folder_path, "imgpoints")

if not os.path.exists(labelled_image_folder_path):
    os.mkdir(labelled_image_folder_path)
    
if not os.path.exists(imgpoints_folder_path):
    os.mkdir(imgpoints_folder_path)
    
# Initialise 
disp_img_width = 900
disp_img_height = 600

num_of_points_calibration_grid = 20

image_names = os.listdir(image_folder_path)

for image_name in image_names:
    
    # if label is already done skip
    txt_save_path = os.path.join(imgpoints_folder_path, image_name.replace(".png", ".txt"))
    if os.path.exists(txt_save_path):
        continue

    image_path = os.path.join(image_folder_path, image_name)

    # Load high-res image
    img_high = cv2.imread(image_path)
    height, width, channels = img_high.shape
    scale = disp_img_height/height
  
    img_disp = cv2.resize(img_high, (0, 0), fx=scale, fy=scale)
    circles_original = []                                                                                                                   
    circles_rescaled = []

    # 1. Detect circles in original high-res image
    gray = cv2.cvtColor(img_high, cv2.COLOR_BGR2GRAY)
    

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200
    params.filterByArea, params.minArea, params.maxArea  = True, 30, 10_000
    params.filterByCircularity, params.minCircularity = True, 0.6 # Default 0.7
    params.filterByConvexity, params.minConvexity = True, 0.8
    params.filterByInertia, params.minInertiaRatio = True, 0.45 # Default 0.5
    params.filterByColor, params.blobColor = True, 255
    detector = cv2.SimpleBlobDetector_create(params)
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)
    
    scaled_keypoints = []

    for kp in keypoints:
        x, y = kp.pt
        # Create a new KeyPoint with scaled coordinates and original size/angle/etc.
        scaled_kp = cv2.KeyPoint(
            x * scale,
            y * scale,
            kp.size * scale,
            kp.angle,
            kp.response,
            kp.octave,
            kp.class_id
        )
        scaled_keypoints.append(scaled_kp)
    
    circles_original = [kp.pt for kp in keypoints]

    circles_rescaled = [(int(round(kp.pt[0])), int(round(kp.pt[1]))) for kp in scaled_keypoints]


    
    
    # if detected is not None:
    #     detected = np.uint16(np.around(detected[0]))
    #     for (x, y, r) in detected:
    #         circles_original.append((x, y, r))

    # 2. Scale circles for display
    # circles_rescaled = [(int(x * scale), int(y * scale), int(r * scale)) for (x, y, r) in circles_original]

    
    # 3. Draw scaled circles on resized image
    img_show_all_points = img_disp.copy()
    for (x, y) in circles_rescaled:
        cv2.circle(img_show_all_points, (x, y), 2, (0, 0, 255), 3)

    

    
    def mouse_event(event, x, y, flags, param):
        global drawing, roi_start, roi_end, img_show_all_points, mode
        global selected_pt_rescaled, selected_pt_original
        global selected_circles_rescaled, selected_circles_original
        global sorted_selected_circles_rescaled

        img_temp = img_show_all_points.copy()

        if mode=='roi':
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                roi_start = (x, y)                                                                                 

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    roi_end = (x, y)  # ← crucial update!
                    cv2.rectangle(img_temp, roi_start, roi_end, (0, 0, 255), 2)
                    cv2.imshow(f"{image_name}", img_temp)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                roi_end = (x, y)
                cv2.rectangle(img_temp, roi_start, roi_end, (0, 0, 255), 2)

                # Now process the ROI selection
                x1, y1 = roi_start
                x2, y2 = roi_end
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)

                # list of 2D tuples [(x1,y1)...., (xn,yn)]
                selected_circles_rescaled = []
                selected_circles_original = []
                for (xr, yr), (xo, yo) in zip(circles_rescaled, circles_original):
                    if x_min <= xr <= x_max and y_min <= yr <= y_max:
                        selected_circles_rescaled.append((xr, yr))
                        selected_circles_original.append((xo, yo))
                        
                        # show only the selected circles in green
                        cv2.circle(img_temp, (xr, yr), 5, (0, 255, 0), -1)
                                
                cv2.imshow(f"{image_name}", img_temp)
            
                # Do not move to the next mode if the roi is not drawn correctly 
                if len(selected_circles_rescaled) == num_of_points_calibration_grid:
                    print("Please select the top left point")
                    mode = "point"
                else:
                    print(f"{len(selected_circles_rescaled)} pts selected. Strictly require: {num_of_points_calibration_grid} pts. Redraw rect or skip next image (Space).")
        
        if mode == 'point':
            if event == cv2.EVENT_LBUTTONDOWN:
                # Find the closest point in roi_centers_all
                distances = np.linalg.norm(selected_circles_rescaled - np.array([x, y]), axis=1)
                idx = np.argmin(distances)
                
                # tuples
                selected_pt_rescaled =  tuple(selected_circles_rescaled[idx])
                selected_pt_original =  tuple(selected_circles_original[idx])
                
                cv2.circle(img_temp, selected_pt_rescaled, 5, (0, 255, 255), -1)
                cv2.imshow(f"{image_name}", img_temp)
                
                # Show the order of points
                selected_pt_rescaled, selected_pt_original = np.array(selected_pt_rescaled), np.array(selected_pt_original)
                selected_circles_rescaled = np.array(selected_circles_rescaled)
                # selected_circles_original = np.array(selected_circles_original)

                # Sort the points and visualise
                grid_order_finder = GridPointsSorter(selected_circles_rescaled)
                grid_order_finder.run_PCA()
                projected = grid_order_finder.projected
                matches = np.all(selected_circles_rescaled == selected_pt_rescaled, axis=1)
                top_left_idx = np.where(matches)[0][0]
                top_left_pt_projected = projected[top_left_idx]
                # Group the points using PC2 value
                _ , sorted_projected, sorted_selected_circles_rescaled = grid_order_finder.sort_group_by_PC_value(top_left_pt_projected, PC_index=1, group_size=4)
                # print("\n")
                # grid_order_finder.print_sorted_results(group_size=4)

                # Sort within the groups to produce the final fully sorted points.
                sorted_projected, sorted_selected_circles_rescaled = grid_order_finder.sort_within_group(top_left_pt_projected, PC_index=0, group_size=4)
                # print("\n")
                # grid_order_finder.print_sorted_results(group_size=4)

                for idx, (x, y) in enumerate(sorted_selected_circles_rescaled):
                    cv2.putText(img_temp, str(idx), (int(x), int(y)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.7, color=(0, 0, 255), thickness=1)
                    
                cv2.imshow(f"{image_name}", img_temp)
                mode = 'sorting_done'
        
        if mode == 'sorting_done':
            # provide option for user to reset 
            if event == cv2.EVENT_RBUTTONDOWN:
                mode = "roi"
                cv2.imshow(f"{image_name}", img_temp)


            
    def apply_same_sort_order(lst_to_be_sorted, lst_sorted, lst_before_sorting):
        """Determines the sort order from lst_before_sorting to lst_sorted, and apply the sort order to be lst_to_be_sorted"""
        def to_tuple(row): return tuple(row.tolist()) if isinstance(row, np.ndarray) else tuple(row)
        value_to_index = {to_tuple(val): idx for idx, val in enumerate(lst_before_sorting)}
        indices = [value_to_index[to_tuple(val)] for val in lst_sorted]
        return lst_to_be_sorted[indices]
            
    # Global vars
    drawing = False
    mode = "roi"
    selected_pt_rescaled, selected_pt_original = None, None
    selected_circles_rescaled , selected_circles_original = [], []
    
    # Show image and set callback
    cv2.imshow(f"{image_name}", img_show_all_points)
    cv2.setMouseCallback(f"{image_name}", mouse_event)
    key = cv2.waitKey(0) & 0xFF
    if key == 32:  # ASCII for spacebar
        cv2.destroyAllWindows()

    # if not empty then save the circles coordinates after space key is pressed
    if selected_circles_original and mode == "sorting_done":

        selected_circles_original = np.array(selected_circles_original)
        sorted_selected_circles_original = apply_same_sort_order(selected_circles_original, sorted_selected_circles_rescaled, selected_circles_rescaled)

        # save the coordinates of the centres of the selected circles
        np.savetxt(txt_save_path, sorted_selected_circles_original)
        
        # save the image for visual checking
        for idx, (x, y) in enumerate(sorted_selected_circles_original):
            
            cv2.putText(img_high, str(idx), (int(x), int(y)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 255, 255), thickness=1)
            
            cv2.circle(img_high, (int(x), int(y)), 2, (0, 0, 255), -1)
        
        cv2.imwrite(os.path.join(labelled_image_folder_path, image_name), img_high)