import os
import sys
import json
import numpy as np
import trimesh
import re
from scipy.spatial.transform import Rotation as R

sys.path.append("/FoundationPose")

from estimater import *
from datareader import *

from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from rclpy.node import Node
from cv_bridge import CvBridge
import rclpy
import message_filters
from message_filters import ApproximateTimeSynchronizer
from rclpy.executors import MultiThreadedExecutor


class MarkerArrayPublisher(Node):
    def __init__(self):
        super().__init__('marker_array_publisher')
        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/rgb')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth')
        self.segmentation_sub = message_filters.Subscriber(self, Image, '/camera/semantic_segmentation')
        self.camera_info = message_filters.Subscriber(self, CameraInfo, '/camera/camera_info')
        self.semantic_labels = self.create_subscription(String, '/camera/semantic_labels', self.semantic_labels_callback, 10)

        # Publisher for MarkerArray
        self.marker_pub = self.create_publisher(MarkerArray, '/object_marker_array', 10)

        # Synchronize topics
        self.ats = ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub, self.segmentation_sub, self.camera_info], queue_size=10, slop=0.1
        )
        self.ats.registerCallback(self.sync_callback)

        self.semantic_labels_dict = {}  # Store parsed labels
        self.mesh_directory = "/FoundationPose/demo_data/ycb"
        self.est_refine_iter = 5

    def semantic_labels_callback(self, msg):
        try:
            self.semantic_labels_dict = json.loads(msg.data)
            self.get_logger().info(f"Parsed semantic labels: {self.semantic_labels_dict}")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse semantic labels: {e}")

    def sync_callback(self, rgb_msg, depth_msg, segmentation_msg, camera_info_msg):
        if not self.semantic_labels_dict:
            self.get_logger().error("Semantic labels are not available.")
            return

        try:
            # Convert messages to OpenCV format
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            segmentation = self.bridge.imgmsg_to_cv2(segmentation_msg, desired_encoding='passthrough')
            intrinsic_matrix = np.array(camera_info_msg.k).reshape(3, 3)

            marker_array = MarkerArray()

            # Iterate through semantic labels
            for index, label_data in self.semantic_labels_dict.items():
                try:
                    label_index = int(index)
                except ValueError:
                    self.get_logger().warning(f"Invalid label key: {index}, skipping...")
                    continue

                class_name = label_data.get("class", "UNKNOWN")
                self.get_logger().info(f"Processing label {label_index}: {class_name}")

                # Create mask for the label
                obj_mask = (segmentation == label_index).astype(np.uint8)
                if np.sum(obj_mask) == 0:
                    self.get_logger().info(f"No mask found for label {class_name}")
                    continue

                # Match directory
                matched_dir = self.match_directory(class_name)
                if not matched_dir:
                    self.get_logger().error(f"No matching directory found for {class_name}")
                    continue

                mesh_path = os.path.join(matched_dir, "google_16k", "textured.obj")
                if not os.path.exists(mesh_path):
                    self.get_logger().error(f"Mesh file not found: {mesh_path}")
                    continue

                # Load mesh and estimate pose
                mesh = trimesh.load(mesh_path)
                to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
                bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

                scorer = ScorePredictor()
                refiner = PoseRefinePredictor()
                glctx = dr.RasterizeCudaContext()

                est = FoundationPose(
                    model_pts=mesh.vertices,
                    model_normals=mesh.vertex_normals,
                    mesh=mesh,
                    scorer=scorer,
                    refiner=refiner,
                    debug_dir="/FoundationPose/debug",
                    debug=1,
                    glctx=glctx
                )

                pose = est.register(
                    K=intrinsic_matrix,
                    rgb=rgb,
                    depth=depth,
                    ob_mask=obj_mask,
                    iteration=self.est_refine_iter
                )
                pose = pose @ np.linalg.inv(to_origin)

                # Extract position and orientation
                position = pose[:3, 3]
                rotation_matrix = pose[:3, :3]
                quaternion = R.from_matrix(rotation_matrix).as_quat()

                # --- 여기서 header를 camera_info_msg의 header로 가져옴 ---
                marker = Marker()
                marker.header.frame_id = camera_info_msg.header.frame_id
                marker.header.stamp = camera_info_msg.header.stamp
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.id = label_index
                marker.pose.position.x = float(position[0])
                marker.pose.position.y = float(position[1])
                marker.pose.position.z = float(position[2])
                marker.pose.orientation.x = float(quaternion[0])
                marker.pose.orientation.y = float(quaternion[1])
                marker.pose.orientation.z = float(quaternion[2])
                marker.pose.orientation.w = float(quaternion[3])
                marker.scale.x = extents[0]
                marker.scale.y = extents[1]
                marker.scale.z = extents[2]
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.text = class_name

                marker_array.markers.append(marker)

            # Publish marker array
            self.marker_pub.publish(marker_array)

        except Exception as e:
            self.get_logger().error(f"Error in sync_callback: {e}")


    def match_directory(self, label_name):
        """Match the label name to the directory, tolerating prefixes like 'ycb_'."""
        # Remove any prefix from the label name
        clean_label_name = re.sub(r'^ycb_', '', label_name)

        # Match the cleaned label name against the directory
        for dir_name in os.listdir(self.mesh_directory):
            if clean_label_name in dir_name:
                return os.path.join(self.mesh_directory, dir_name)
        return None


def main(args=None):
    try:
        rclpy.init(args=args)
        node = MarkerArrayPublisher()
        executor = MultiThreadedExecutor()
        rclpy.spin(node, executor)
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
