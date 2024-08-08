import sys
import os
import cv2
import numpy as np
import trimesh
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import json
import torch

sys.path.append("/FoundationPose")

from estimater import *
from datareader import *

app = FastAPI()

@app.on_event("startup")
def startup_event():
    global code_dir
    code_dir = os.path.dirname(os.path.realpath(__file__))

@app.on_event("shutdown")
def shutdown_event():
    torch.cuda.empty_cache()
    logging.info("GPU memory has been freed.")

@app.get("/realtime")
def real_time_video():
    pass

@app.post("/image")
async def image_upload(
    rgb: UploadFile = File(...),
    depth: UploadFile = File(...),
    mask: UploadFile = File(...),
    semantic_labels: str = Form(...),
    K_matrix: str = Form(...),
    debug: int = Form(3),
    est_refine_iter: int = Form(5),
):
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    set_logging_format()
    set_seed(0)

    # 파일 읽기
    rgb_data = await rgb.read()
    depth_data = await depth.read()
    mask_data = await mask.read()

    # RGB 이미지 디코딩
    rgb_image = cv2.imdecode(np.frombuffer(rgb_data, np.uint8), cv2.IMREAD_COLOR)
    if rgb_image is None:
        raise HTTPException(status_code=400, detail="RGB image decoding failed")

    # Depth 이미지 디코딩
    depth_image = cv2.imdecode(np.frombuffer(depth_data, np.uint8), cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise HTTPException(status_code=400, detail="Depth image decoding failed")
    depth_image = depth_image.astype(np.float32)

    # Mask 이미지 디코딩
    mask_image = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_UNCHANGED)/1e3
    if mask_image is None:
        raise HTTPException(status_code=400, detail="Mask image decoding failed")

    depth_image[(depth_image<0.1) | (depth_image>=np.inf)] = 0

    for c in range(3):
        if mask_image[...,c].sum()>0:
            mask_image = mask_image[...,c]
            break

    # JSON 문자열 파싱
    semantic_labels = json.loads(semantic_labels)
    K_matrix = np.array(json.loads(K_matrix))

    vis = rgb_image.copy()

    for idx, semantic_label in enumerate(semantic_labels):
        mesh = trimesh.load(f'/FoundationPose/demo_data/ycb/{semantic_label}/google_16k/textured.obj')
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

        est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=code_dir,
            debug=debug,
            glctx=glctx
        )

        logging.info(f"Estimator initialization done for {semantic_label}")

        obj_mask = mask_image.astype(bool)
        pose = est.register(
            K=K_matrix,
            rgb=rgb_image,
            depth=depth_image,
            ob_mask=obj_mask,
            iteration=est_refine_iter
        )

        center_pose = pose @ np.linalg.inv(to_origin)

        print(pose, center_pose)

        vis = draw_posed_3d_box(K_matrix, img=vis, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K_matrix, thickness=3, transparency=0, is_input_rgb=True)

    output_path = "output.png"
    success = cv2.imwrite(output_path, vis)
    if not success:
        logging.error("Failed to save the output image using cv2.imwrite")
        raise HTTPException(status_code=500, detail="Failed to save the output image using cv2.imwrite")

    if not os.path.exists(output_path):
        logging.error(f"Output file does not exist: {output_path}")
        raise HTTPException(status_code=500, detail=f"Output file does not exist: {output_path}")

    return FileResponse(output_path, media_type="image/png", filename='output.png')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5678)
