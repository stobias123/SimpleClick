import torch
import numpy as np
import logging
from tkinter import messagebox
import cv2

from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks


class ServerController:
    def __init__(self, net, device, predictor_params, update_image_callback, prob_thresh=0.5):
        self.net = net
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()
        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_mask = None
        self._init_mask = None

        self.image = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.reset_predictor()

    def set_image(self, image):
        logging.debug(f"set_image {image.shape}")
        self.image = image
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        self.object_count = 0
        self.reset_last_object(update_image=False)
        self.update_image_callback()

    def set_mask(self, mask):
        logging.debug('I am setting a mask now!')
        if self.image.shape[:2] != mask.shape[:2]:
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)
        self.clicker.click_indx_offset = 1

    # TODO: This can be an endpoint for adding clicks.
    def add_click(self, x, y, is_positive):
        logging.debug(f"add_click {x} {y} {is_positive}")
        state = self.clicker.get_state()
        self.states.append({
            'clicker': state,
            'predictor': self.predictor.get_states()
        })
        logging.debug(f"state is {state}")

        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)
        pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
        if self._init_mask is not None and len(self.clicker) == 1:
            pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)

        torch.cuda.empty_cache()

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))

        self.update_image_callback()

    def undo_click(self):
        logging.debug('undo_click')
        if not self.states:
            return

        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])
        self.probs_history.pop()
        if not self.probs_history:
            self.reset_init_mask()
        self.update_image_callback()

    def partially_finish_object(self):
        logging.debug('partially_finish_object')
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.probs_history.append((object_prob, np.zeros_like(object_prob)))
        self.states.append(self.states[-1])

        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        self.update_image_callback()

    def finish_object(self):
        logging.debug('finish_object')
        if self.current_object_prob is None:
            return

        self._result_mask = self.result_mask
        self.object_count += 1
        self.reset_last_object()

    def reset_last_object(self, update_image=True):
        logging.debug(f"reset_last_object {update_image}")
        self.states = []
        self.probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        if update_image:
            self.update_image_callback()

    def reset_predictor(self, predictor_params=None):
        logging.debug('reset_predictor')
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        logging.debug('reset_init_mask')
        self._init_mask = None
        self.clicker.click_indx_offset = 0

    def get_points(self,points_array):
        return_arr = []
        i = 0
        while i < len(points_array):
            x = points_array[i+0]
            y = points_array[i+1]
            i+=2
            return_arr.append((x,y))
        return return_arr

    def parse_image(self, client, bucket, image_s3_url, shapes_for_frame, predictor_params, state):
        """
        parse_image will do the following.
            1. Download the image from image_s3_url
            2. Get all the points from the cvat shape,
            3. "Finish object" - Pass it through the net and set final mask.
            4. Return the mask
        :return: np.array(uint.8)
        """
        bucket, key = image_s3_url.replace("s3://", "").split("/", 1)
        logging.info(f"Downloading image from {image_s3_url}")
        client.download_file(bucket, key, key)
        image = cv2.cvtColor(cv2.imread(key), cv2.COLOR_BGR2RGB)
        self.reset_predictor(predictor_params)
        self.set_image(image)
        img = self.get_visualization(state.alpha_blend,state.click_radius)
        for instance in shapes_for_frame:
            for point in self.get_points(instance['points']):
                self.add_click(point[0],point[1], True)
            self.finish_object()
        return self.result_mask

    def write_pretty_mask(self, key, alpha_blend, click_radius, s3_client = None, mask_path: str = 'pretty_mask', bucket_name: str = 'bird-mlflow-bucket'):
        """
        write_pretty_mask returns the nice mask like you'd want to see in a GUI.
        It will optionally upload it to s3 if an s3_client is provided
        """
        img = self.get_visualization(alpha_blend, click_radius)
        logging.info(f"Writing pretty to {mask_path}/{key}")
        cv2.imwrite(f"{mask_path}/{key}", img)
        if s3_client is not None:
            logging.info(f"Uploading image to {bucket_name}/{mask_path}/{key}")
            s3_client.upload_file(f"{mask_path}/{key}",bucket_name, f"{mask_path}/{key}")

    def write_simple_mask(self, key, s3_client = None, mask_path: str = 'binary_mask', bucket_name: str = 'bird-mlflow-bucket'):
        """
        write_simple_mask returns the a binary ish mask, that could be useful for downstream operations.
        It will optionally upload it to s3 if an s3_client is provided
        """
        logging.info(f"Writing image to {mask_path}/{key}")
        img = self.result_mask
        if img.max() < 256:
            mask = img.astype(np.uint8)
            mask *= 255 // mask.max()
        cv2.imwrite(f"{mask_path}/{key}", img)
        if s3_client is not None:
            logging.info(f"Uploading image to {bucket_name}/{mask_path}/{key}")
            s3_client.upload_file(f"{mask_path}/{key}",bucket_name, f"{mask_path}/{key}")

    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        if self.probs_history:
            result_mask[self.current_object_prob > self.prob_thresh] = self.object_count + 1
        return result_mask

    def get_visualization(self, alpha_blend, click_radius):
        logging.info('get_visualization')
        if self.image is None:
            return None

        results_mask_for_vis = self.result_mask
        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius)
        if self.probs_history:
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend)

        return vis
