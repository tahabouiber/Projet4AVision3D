import cv2
import numpy as np
import glob

# Define the chessboard size
chessboard_size = (9, 6)
frame_size = (640, 480)

# Prepare object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []
imgpoints1 = []
imgpoints2 = []

# Load images
images1 = glob.glob('captures/left/*.jpg')
images2 = glob.glob('captures/right/*.jpg')

for img1, img2 in zip(images1, images2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, chessboard_size, None)

    if ret1 and ret2:
        objpoints.append(objp)
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)

# Calibrate the cameras
ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, frame_size, None, None)
ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, frame_size, None, None)

print("Camera 1 Matrix (mtx1):\n", mtx1)
print("Camera 2 Matrix (mtx2):\n", mtx2)

# Stereo calibration
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, frame_size, criteria=criteria, flags=flags)

print("Stereo Calibration Results:\n")
print("R:\n", R)
print("T:\n", T)
print("E:\n", E)
print("F:\n", F)

# Rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, frame_size, R, T, alpha=0)

# Load rectified images
imgL = cv2.imread('left.jpg', 0)
imgR = cv2.imread('right.jpg', 0)

# Create stereo matcher
stereo = cv2.StereoSGBM_create(
    minDisparity=1,
    numDisparities=16 * 3,
    blockSize=6,
    P1=8 * 11 ** 2,
    P2=16 * 11 ** 2,
    disp12MaxDiff=75,
    uniquenessRatio=1,
    speckleWindowSize=100,
    speckleRange=128


)

# Compute disparity map
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

# Calculate depth map
focal_length = mtx1[0, 0]
baseline_cm = 5.0  # Distance between the two cameras in cm
baseline_mm = baseline_cm * 10  # Convert baseline to mm

depth_map = np.zeros(disparity.shape, np.float32)
depth_map[disparity > 0] = (focal_length * baseline_mm) / (disparity[disparity > 0])

# Print depth map
print("Depth Map:\n", depth_map)

# Normalize disparity for visualization
disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity_vis = np.uint8(disparity_vis)

# Load the original left image
original_imgL = cv2.imread('left.jpg')

# Function to display depth at cursor position
def show_depth(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        if x < original_imgL.shape[1]:
            depth = depth_map[y, x]
        else:
            depth = depth_map[y, x - original_imgL.shape[1]]
        
        depth_cm = depth / 10  # Convert mm to cm for display
        
        img_with_text = original_imgL.copy()
        disparity_with_text = cv2.cvtColor(disparity_vis, cv2.COLOR_GRAY2BGR).copy()
        
        # Draw depth text on both images
        if x < original_imgL.shape[1]:
            cv2.putText(img_with_text, f"{depth_cm:.2f} cm", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            combined_img = np.hstack((img_with_text, cv2.cvtColor(disparity_vis, cv2.COLOR_GRAY2BGR)))
        else:
            cv2.putText(disparity_with_text, f"{depth_cm:.2f} cm", (x - original_imgL.shape[1], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            combined_img = np.hstack((img_with_text, disparity_with_text))
        
        cv2.imshow('Original and Disparity', combined_img)

# Combine images for initial display
combined_img = np.hstack((original_imgL, cv2.cvtColor(disparity_vis, cv2.COLOR_GRAY2BGR)))

# Show the combined image and set mouse callback
cv2.imshow('Original and Disparity', combined_img)
cv2.setMouseCallback('Original and Disparity', show_depth)
cv2.waitKey(0)
cv2.destroyAllWindows()
