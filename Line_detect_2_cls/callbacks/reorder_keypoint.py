import numpy as np
from shapely.geometry import Point
from shapely.affinity import rotate

class KeypointReorder:
    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height

    def reorder_keypoints(self, keypoints, rotation_angle):
        """
        Reorder keypoints to ensure that for each line, the keypoint on the right or the top is first,
        even after rotation.
        """
        if keypoints.shape[0] % 2 != 0:
            raise ValueError("Keypoints should be in pairs of two")

        # Apply rotation to keypoints
        center = (self.img_width / 2, self.img_height / 2)
        rotated_keypoints = self._rotate_keypoints(keypoints, rotation_angle, center)

        # Reorder rotated keypoints
        reordered_keypoints = self._sort_keypoints(rotated_keypoints)

        return reordered_keypoints

    def _rotate_keypoints(self, keypoints, angle, center):
        """
        Rotate keypoints by the given angle around the center.
        """
        rotated_keypoints = []
        for pt in keypoints:
            point = Point(pt)
            rotated_point = rotate(point, angle, origin=center)
            rotated_keypoints.append([rotated_point.x, rotated_point.y])
        return np.array(rotated_keypoints)

    def _sort_keypoints(self, keypoints):
        """
        Sort keypoints such that for each line, the keypoint on the right or the top is first.
        """
        sorted_keypoints = []

        for i in range(0, len(keypoints), 2):
            pt1, pt2 = keypoints[i], keypoints[i+1]
            
            if pt1[0] > pt2[0]:  # pt1 is to the right of pt2
                sorted_keypoints.extend([pt1, pt2])
            elif pt1[0] < pt2[0]:  # pt1 is to the left of pt2
                sorted_keypoints.extend([pt2, pt1])
            else:  # Vertical line
                if pt1[1] > pt2[1]:  # pt1 is above pt2
                    sorted_keypoints.extend([pt1, pt2])
                else:
                    sorted_keypoints.extend([pt2, pt1])

        return np.array(sorted_keypoints)

