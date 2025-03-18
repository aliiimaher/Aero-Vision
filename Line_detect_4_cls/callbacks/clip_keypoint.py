from shapely.geometry import LineString

class KeypointClipper:
    def __init__(self, img_width, img_height):
        # Initialize with the width and height of the image
        self.img_width = img_width
        self.img_height = img_height

    def find_intersection_with_border(self, point1, point2):
        """
        Find the intersection of the line segment defined by point1 and point2 with the image borders.
        """
        # Define the borders of the image as LineString objects
        border_lines = [
            LineString([(0, 0), (self.img_width, 0)]),  # Top border
            LineString([(0, 0), (0, self.img_height)]),  # Left border
            LineString([(self.img_width, 0), (self.img_width, self.img_height)]),  # Right border
            LineString([(0, self.img_height), (self.img_width, self.img_height)])  # Bottom border
        ]

        # Create a LineString for the line segment defined by point1 and point2
        line = LineString([point1, point2])

        for border in border_lines:
            if line.intersects(border):
                intersection = line.intersection(border)
                if intersection.geom_type == 'Point':
                    # Return the intersection point as a tuple (x, y)
                    return (intersection.x, intersection.y)
        return None

    def clip_keypoints_to_image(self, keypoints):
        """
        Clip the keypoints to the image borders, adjusting them if they fall outside the image.
        """
        # Copy the list of keypoints to avoid modifying the original list
        clipped_keypoints = keypoints.copy()

        for i in range(0, len(keypoints) - 1, 2):
            pt1 = keypoints[i]
            pt2 = keypoints[i + 1]

            # Check if pt1 is outside the image borders
            if not (0 <= pt1[0] <= self.img_width and 0 <= pt1[1] <= self.img_height):
                intersection = self.find_intersection_with_border(pt1, pt2)
                if intersection:
                    # Adjust pt1 to the intersection point
                    clipped_keypoints[i] = intersection

            # Check if pt2 is outside the image borders
            if not (0 <= pt2[0] <= self.img_width and 0 <= pt2[1] <= self.img_height):
                intersection = self.find_intersection_with_border(pt2, pt1)
                if intersection:
                    # Adjust pt2 to the intersection point
                    clipped_keypoints[i + 1] = intersection

        return clipped_keypoints