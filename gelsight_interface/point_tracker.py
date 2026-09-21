from scipy.spatial import cKDTree
import numpy as np
import cv2

#How to use it:
#tracker = PointTracker()
#circles = cv2.HoughCircles(gray_frame_1, cv2.HOUGH_GRADIENT, 1.5, 30, param1=40, param2=40, minRadius=40, maxRadius=75)
#tracked_points = tracker.update(circles[0,:,:2])
#tracker.plot_tracks(frame, tracked_points)

class PointTracker:
    def __init__(self, distance_threshold=50.0, max_occlusion_time=15):
        self.active_tracks = {}  # Active points being tracked
        self.inactive_tracks = {}  # Points temporarily occluded
        self.next_id = 0  # ID counter for new points
        self.initial_locations = {}  # Store initial locations of points
        self.distance_threshold = distance_threshold  # Max distance for matching 
        self.max_occlusion_time = max_occlusion_time  # Frames to keep inactive points

    def update(self, points):
        """
        Update the tracker with a new set of points.

        Parameters:
        points (ndarray): Nx2 array of new points.

        Returns:
        dict: Updated tracks with point IDs as keys and coordinates as values.
        """
        points = self.order_points(np.array(points))
        new_active_tracks = {}
        unmatched_points = set(range(len(points)))

        # Step 1: Match current points to active tracks
        if len(self.active_tracks) > 0:
            previous_points = np.array(list(self.active_tracks.values()))
            previous_ids = list(self.active_tracks.keys())

            tree = cKDTree(previous_points)
            distances, indices = tree.query(points)

            for i, (dist, idx) in enumerate(zip(distances, indices)):
                if dist < self.distance_threshold:
                    point_id = previous_ids[idx]
                    new_active_tracks[point_id] = points[i]
                    unmatched_points.discard(i)

        # Step 2: Reactivate points from inactive tracks
        if len(self.inactive_tracks) > 0:
            previous_points = np.array([data['position'] for data in self.inactive_tracks.values()])
            previous_ids = list(self.inactive_tracks.keys())

            tree = cKDTree(previous_points)
            distances, indices = tree.query(points)

            for i, point_idx in enumerate(unmatched_points.copy()):  # Iterate over unmatched_points safely
                dist = distances[point_idx]
                idx = indices[point_idx]
                if dist < self.distance_threshold:
                    point_id = previous_ids[idx]
                    new_active_tracks[point_id] = points[point_idx]
                    unmatched_points.discard(point_idx)
                    del self.inactive_tracks[point_id]

        # Step 3: Add unmatched points as new tracks
        for i in unmatched_points:
            new_active_tracks[self.next_id] = points[i]
            if self.next_id not in self.initial_locations:
                self.initial_locations[self.next_id] = points[i]
            self.next_id += 1

        # Step 4: Update inactive tracks
        for point_id, data in list(self.inactive_tracks.items()):
            data['time_inactive'] += 1
            if data['time_inactive'] > self.max_occlusion_time:
                del self.inactive_tracks[point_id]  # Remove expired tracks

        # Step 5: Move unmatched active tracks to inactive
        for point_id in self.active_tracks.keys() - new_active_tracks.keys():
            self.inactive_tracks[point_id] = {'position': self.active_tracks[point_id], 'time_inactive': 1}

        # Update active tracks
        self.active_tracks = new_active_tracks
        return self.active_tracks

    def get_initial_locations(self):
        """
        Retrieve the initial locations of all points.

        Returns:
        dict: A dictionary where keys are point IDs and values are their initial locations.
        """
        return self.initial_locations
    
    def plot_tracks(self, frame, circles):
        if circles is not None:
            for point_id, (x, y) in circles.items():
                x, y = int(round(x)), int(round(y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), 4)
                cv2.putText(frame, str(point_id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    def order_points(self, points):
        """
        Orders noisy grid-like points from top-left to bottom-right without modifying their values.

        Args:
        - points: List of (x, y) tuples.
        - x_tol, y_tol: Tolerance to group points into approximate rows and columns.

        Returns:
        - Ordered list of (x, y) tuples.
        """
        points = np.array(points)

        # Sort points by y first (descending), then by x (ascending)
        sorted_points = sorted(points, key=lambda p: (-p[1], p[0]))

        # Group points into approximate rows
        rows = []
        while sorted_points:
            row = [sorted_points.pop(0)]
            to_remove = []
            for p in sorted_points:
                if abs(p[1] - row[0][1]) <= self.distance_threshold:  # If within the same row
                    row.append(p)
                    to_remove.append(p)
            for p in to_remove:
                sorted_points.remove(p)

            # Sort the row by x ascending
            row.sort(key=lambda p: p[0])
            rows.append(row)

        # Flatten the list
        return [p for row in rows for p in row]