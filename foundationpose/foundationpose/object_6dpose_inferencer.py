import argparse
import os
import sys

sys.path.append("/FoundationPose")

from estimater import *
from datareader import *

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
from message_filters import ApproximateTimeSynchronizer
# import marker and TF
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster
import tf2_ros
import tf2_geometry_msgs
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String

class ObjectPoseEstimater(Node):
    def __init__(self):
        super().__init__('object_pose_estimater')

        self.bridge = CvBridge()
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/rgb')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth')
        self.segmentation_sub = message_filters.Subscriber(self, Image, '/camera/semantic_segmentation')
        self.camera_info = message_filters.Subscriber(self, CameraInfo, '/camera/camera_info')
        self.semantic_labels = self.create_subscription(String, '/camera/semantic_labels', self.semantic_labels_callback, 10)

        # publish marker
        self.marker_pub = self.create_publisher(Marker, '/object_marker', 10)

        self.ats = ApproximateTimeSynchronizer([self.image_sub, self.depth_sub, self.segmentation_sub, self.camera_info], 10, 0.1)
        self.ats.registerCallback(self.sync_callback)

        #set foundationpose args
        self.mesh_file = '/home/hyeonsu/Documents/docker/moveit2_docker/FoundationPose/demo_data/mustard0/mesh/textured_simple.obj'
        self.mesh = trimesh.load(self.mesh_file)
        self.to_origin, self.extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-self.extents/2, self.extents/2], axis=0).reshape(2,3)
        self.est_refine_iter = 5
        self.track_refine_iter = 2

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals, mesh=self.mesh, scorer=scorer, refiner=refiner, debug_dir='/home/hyeonsu/Documents/docker/moveit2_docker/FoundationPose/debug', debug=1, glctx=glctx)
        self.counter = 0
        self.semantic_labels_msg = None

    def semantic_labels_callback(self, msg):
        self.semantic_labels_msg = msg

    def sync_callback(self, rgb_msg, depth_msg, segmentation_msg, camera_info_msg):
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        #normalize depth to 11
        # depth = depth / 255.0 * 7.0
        if ((self.counter == 0) & (self.semantic_labels_msg is not None)):
            self.intrinsic_matrixK = camera_info_msg.k
            self.intrinsic_matrixK = np.array(self.intrinsic_matrixK).reshape(3,3)
            segmentation = self.bridge.imgmsg_to_cv2(segmentation_msg, desired_encoding='passthrough')
            mesh_index = 6
            mask = np.where(segmentation == mesh_index, 1, 0).astype(np.uint8)
            pose = self.est.register(K=self.intrinsic_matrixK, rgb=rgb, depth=depth, ob_mask=mask, iteration=self.est_refine_iter)
            self.counter += 1
        elif self.counter > 0:
            pose = self.est.track_one(rgb=rgb, depth=depth, K=self.intrinsic_matrixK, iteration=self.track_refine_iter)
            # publish marker
            marker = Marker()
            marker.header.frame_id = 'Camera'
            marker.header.stamp = rgb_msg.header.stamp
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.id = 0
            marker.pose.position.x = float(pose[0,3])
            marker.pose.position.y = float(pose[1,3])
            marker.pose.position.z = float(pose[2,3])
            marker.pose.orientation.x = float(pose[0,0])
            marker.pose.orientation.y = float(pose[1,1])
            marker.pose.orientation.z = float(pose[2,2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.bbox[1,1]*2
            marker.scale.y = self.bbox[1,2]*2
            marker.scale.z = self.bbox[1,0]*2
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0


            self.marker_pub.publish(marker)
            center_pose = pose@np.linalg.inv(self.to_origin)
            vis = draw_posed_3d_box(self.intrinsic_matrixK, img=rgb, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(rgb, ob_in_cam=center_pose, scale=0.1, K=self.intrinsic_matrixK, thickness=3, transparency=0, is_input_rgb=True)
            #save image every 10 frames
            if self.counter % 10 == 0:
                cv2.imwrite(f'/home/hyeonsu/Documents/docker/moveit2_docker/FoundationPose/debug/track_vis/{self.counter}.png', vis)
            self.counter += 1
        else:
            pass

def main(args=None):
    rclpy.init(args=args)
    opest = ObjectPoseEstimater()
    executor = MultiThreadedExecutor()
    rclpy.spin(opest, executor)
    opest.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
