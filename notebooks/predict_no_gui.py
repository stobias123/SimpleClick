from dataclasses import dataclass, asdict
from typing import List
import numpy as np

import boto3 as boto3

from interactive_demo.server_controller import ServerController
from isegm.inference import utils
import torch
import cv2

BUCKET=''

@dataclass
class Config:
    device: str = "cpu"
    checkpoint: str = "sbd_vit_huge-002.pth"
    interactive_models_path: str = "../weights"

@dataclass
class PredictorParams:
    net_clicks_limit: int = 8

@dataclass
class ZoomInParams:
    skip_clicks: int = -1
    target_size: tuple = (448, 448)
    expansion_ratio: float = 1.4

@dataclass
class State:
    zoomin_params: ZoomInParams
    predictor_params: PredictorParams
    brs_mode: str = 'NoBRS'
    prob_thresh: float = 0.5
    lbfgs_max_iters: int = 20
    alpha_blend: float = 0.5
    click_radius: int = 3

def update_image_callback():
    print('callback')

def download_from_s3(boto3_session, image_s3_url):
    """
    Download image from s3
    :param image_s3_url: The s3 url of the image. Example: s3://bucket/path/to/image.jpg
    :return: a numpy array representing the image
    """
    s3 = boto3.resource('s3')
    bucket, key = image_s3_url.replace("s3://", "").split("/", 1)
    obj = s3.Object(bucket, key)
    response = obj.get()
    file_stream = response['Body']
    image = np.asarray(bytearray(file_stream.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def get_mask_from_points(clicks: List[tuple], image_s3_url):
    pass

def get_points(points_array):
    return_arr = []
    i = 0
    while i < len(points_array):
        x = points_array[i+0]
        y = points_array[i+1]
        i+=2
        return_arr.append((x,y))
    return return_arr

def get_mask_from_image(client, controller, image_s3_url, shapes_for_frame,predictor_params, state):
    bucket, key = image_s3_url.replace("s3://", "").split("/", 1)
    print(f"Downloading image from {image_s3_url}")
    client.download_file(BUCKET, key, key)
    image = cv2.cvtColor(cv2.imread(key), cv2.COLOR_BGR2RGB)
    controller.reset_predictor(predictor_params)
    controller.set_image(image)
    img = controller.get_visualization(state.alpha_blend,state.click_radius)
    for instance in shapes_for_frame:
        for point in get_points(instance['points']):
            controller.add_click(point[0],point[1], True)
        controller.finish_object()
    img2 = controller.get_visualization(state.alpha_blend,state.click_radius)
    cv2.imwrite(f"mask/{key}", img2)

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    config = Config()
    state = State(zoomin_params=ZoomInParams(),predictor_params=PredictorParams())
    checkpoint_path = utils.find_checkpoint(config.interactive_models_path, config.checkpoint)
    model = utils.load_is_model(checkpoint_path, config.device, False, cpu_dist_maps=True)
    predictor_params = {
            'brs_mode': state.brs_mode,
            'prob_thresh': state.prob_thresh,
            'zoom_in_params': asdict(state.zoomin_params),
            'predictor_params': {
                'net_clicks_limit': state.predictor_params.net_clicks_limit,
                'max_size': 800 ## Default
            },
            'brs_opt_func_params': {'min_iou_diff': 1e-3},
            'lbfgs_params': {'maxfun': state.lbfgs_max_iters}
        }
    controller = ServerController(model, config.device,
                                  predictor_params=predictor_params,
                                  update_image_callback=update_image_callback)

    FNAME = '/Users/steven.tobias/Desktop/image.jpg'
    image = cv2.cvtColor(cv2.imread(FNAME), cv2.COLOR_BGR2RGB)
    controller.reset_predictor(predictor_params)
    controller.set_image(image)
    #controller.reset_last_object(update_image=False)
    #controller.reset_init_mask()

    img = controller.get_visualization(state.alpha_blend,state.click_radius)

    controller.add_click(756,787, True)
    controller.add_click(368,575, True)
    controller.finish_object()
    controller.add_click(635,534, True)
    controller.add_click(514,232, True)
    controller.finish_object()

    img2 = controller.get_visualization(state.alpha_blend,state.click_radius)

    cv2.imwrite('test.png', img)
    cv2.imwrite('test2.png', img2)
