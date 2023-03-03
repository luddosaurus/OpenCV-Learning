import cv2

from base.utils.Calibrator import Calibrator
from base.utils.Camera import Camera


camera = Camera()
cb = Calibrator()

# Run camera
while True:
    # Run Camera
    image = camera.fetch()
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    # Do Stuff
    chess_image = cb.update_chessboard_points(image)

    # Show Image
    cv2.imshow("Video", chess_image)

# Calibrate
calibrated_image = cb.calibrate(img=image)
cv2.imshow("Calibrated", calibrated_image)
cv2.waitKey(0)
print("Calibration pre-save ", cb.camera_matrix)

# Save n Load calibration matrix
save_path = "calibrations/matrix2"

cb.save_calibration(path=save_path)

cb.find_error()

camera.stop()
cv2.destroyAllWindows()
