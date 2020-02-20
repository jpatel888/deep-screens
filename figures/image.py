from utils.image_utils import concatenate_images_by_width
import numpy as np


class Image:
    def __init__(self, config, input_imgs, label=None, logit=None):
        self.config = config
        self.baseline_img = input_imgs[:, :, :3]
        self.current_img = input_imgs[:, :, 3:]
        self.label = label
        self.logit = logit

    def apply_box(self, image, bounding_box, color):
        """

        :param image:
        :param bounding_box: (left_x, right_x, top_y, bottom_y)
        :param color:
        :return:
        """
        line_width = self.config.figure.line_width
        try:
            image[bounding_box[1] - line_width:bounding_box[1], bounding_box[0]:bounding_box[2]] = color
            image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[0] + line_width] = color
            image[bounding_box[3] - line_width:bounding_box[3], bounding_box[0]:bounding_box[2]] = color
            image[bounding_box[1]:bounding_box[3], bounding_box[2]:bounding_box[2] + line_width] = color
        except:
            pass
        return image

    def get_bounding_box(self, y_idx, x_idx, yxhw):
        num_grid_cells_width = self.config.model_output_size[1]
        num_grid_cells_height = self.config.model_output_size[0]
        image_width = self.config.input_shape[1]
        image_height = self.config.input_shape[0]
        grid_cell_width = image_width / num_grid_cells_width
        grid_cell_height = image_height / num_grid_cells_height
        my, mx, h, w = yxhw[0], yxhw[1], yxhw[2], yxhw[3]
        #print(yxhw)
        mx = (mx * grid_cell_width) + (x_idx * grid_cell_width)
        my = (my * grid_cell_height) + (y_idx * grid_cell_height)
        h = h * grid_cell_height
        w = w * grid_cell_width
        #print(mx, my, h, w)
        lx = mx - (w/2)
        rx = mx + (w/2)
        ty = my - (h/2)
        by = my + (h/2)
        #print(lx, rx, ty, by)
        ret = int(lx), int(ty), int(rx), int(by)
        return ret

    def get_color(self, categories):
        category = np.argmax(categories)
        return self.config.figure.color_map[category]

    def apply_label(self, image, label):
        for _y in range(label.shape[0]):
            for _x in range(label.shape[1]):
                has_defect = label[_y, _x, 0]
                if has_defect > 0.5:
                    bounding_box = self.get_bounding_box(_y, _x, label[_y, _x, 5:])
                    color = self.get_color(label[_y, _x, 1:5])
                    image = self.apply_box(image, bounding_box, color)
        return image

    def get_log_image(self):
        first_image = self.baseline_img
        second_image = self.apply_label(self.current_img, self.label) if self.label is not None else None
        third_image = self.apply_label(self.current_img, self.logit) if self.logit is not None else None
        all_images = filter(lambda el: el is not None, [first_image, second_image, third_image])
        return concatenate_images_by_width(list(all_images))
