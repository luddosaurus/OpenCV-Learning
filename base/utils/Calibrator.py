import numpy as np
import cv2


class Calibrator:
    # Calibration Settings
    chessboard_y = 5+1
    chessboard_x = 8+1
    chessboard_size = (chessboard_x, chessboard_y)

    # termination criteria for finding good corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_y * chessboard_x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_x, 0:chessboard_y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.

    distortion_coefficients = []
    rotation_vectors = []
    translation_vectors = []
    camera_matrix = []

    def reset_settings(self):
        self.obj_points = []
        self.img_points = []
        self.chessboard_y = 5
        self.chessboard_x = 8
        self.chessboard_size = (self.chessboard_x, self.chessboard_y)

    # finds the 2d and 3d points of the chessboard and stores the points
    def update_chessboard_points(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        found_pattern, corners = cv2.findChessboardCorners(
            image=gray_img,
            patternSize=self.chessboard_size
        )

        if found_pattern:
            self.obj_points.append(self.objp)
            # find 'clean' corner
            clean_corners = cv2.cornerSubPix(
                image=gray_img,
                corners=corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=self.criteria
            )
            self.img_points.append(clean_corners)
            cv2.drawChessboardCorners(
                image=img,
                patternSize=self.chessboard_size,
                corners=clean_corners,
                patternWasFound=found_pattern
            )
            return img

        else:
            return img

    def calibrate(self, img, debug=False):
        if not self.obj_points:
            return Exception("Find chessboard points first")

        height, width = img.shape[:2]
        image_size = (width, height)

        ret, camera_matrix, distortion, rotation_vectors, translation_vectors = cv2.calibrateCamera(
            objectPoints=self.obj_points,
            imagePoints=self.img_points,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None)

        if debug:
            print("Camera Calibrated: ", ret)
            print("Camera Matrix: ", camera_matrix)
            print("Camera Distortion: ", distortion)
            print("Camera Rotation: ", rotation_vectors)
            print("Camera Translation: ", translation_vectors)
            print("Object Points: ", len(self.obj_points))

        optimised_matrix, roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=camera_matrix,
            distCoeffs=distortion,
            imageSize=image_size,
            alpha=1,
            newImgSize=image_size
        )

        image_undistorted = cv2.undistort(
            src=img,
            cameraMatrix=camera_matrix,
            distCoeffs=distortion,
            dst=None,
            newCameraMatrix=optimised_matrix
        )
        x, y, w, h, = roi
        image_undistorted = image_undistorted[y:y+h, x:x+w]

        self.camera_matrix = optimised_matrix
        self.distortion_coefficients = distortion
        self.rotation_vectors = rotation_vectors
        self.translation_vectors = translation_vectors

        return image_undistorted

    def find_error(self):
        # Reprojection Error
        mean_error = 0

        for i in range(len(self.obj_points)):
            projectedPoints, _ = cv2.projectPoints(
                self.obj_points[i],
                self.rotation_vectors[i],
                self.translation_vectors[i],
                self.camera_matrix,
                self.distortion_coefficients
            )
            error = cv2.norm(self.img_points[i], projectedPoints, cv2.NORM_L2) / len(projectedPoints)
            mean_error += error

        print("total error: {}".format(mean_error / len(self.obj_points)))

    def save_calibration(self, path):
        np.savez(
            file=path,
            matrix=self.camera_matrix,
            distortion=self.distortion_coefficients,
            rotation_vectors=self.rotation_vectors,
            translation_vectors=self.translation_vectors,
        )

    def load_calibration(self, path):
        with np.load(f"{path}.npz") as X:
            self.camera_matrix, self.distortion_coefficients, self.rotation_vectors, self.translation_vectors = \
                [X[i] for i in ('matrix', 'distortion', 'rotation_vectors', 'translation_vectors')]

