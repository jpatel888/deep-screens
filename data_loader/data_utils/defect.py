import numpy as np


class Defect:
    def __init__(self, config, defect, image_height, image_width):
        self.all_defect_types = config.defect_types
        self.defect_type = defect.defect_type
        self.defect_midpoint_x = defect.location.midpoint_x
        self.defect_midpoint_y = defect.location.midpoint_y
        self.defect_height = defect.location.height
        self.defect_width = defect.location.width
        self.image_height = image_height
        self.image_width = image_width

    def get_y_index(self, grid_height):
        """
        using grid_height, self.defect_height, and self.image_height, figure out the vertical index
        :param grid_height:
        :return:
        """
        return int(self.defect_midpoint_y / self.image_height * grid_height)

    def get_x_index(self, grid_width):
        """
        using grid_width and self.image_width, figure out the vertical index
        :param grid_width:
        :return:
        """
        return int(self.defect_midpoint_x / self.image_width * grid_width)

    def get_y(self, grid_height):
        """
        using grid_height and self.image_height, figure out % of the cell
        :param grid_height:
        :return:
        """
        cell_height = self.image_height / grid_height
        y = (self.defect_midpoint_y % cell_height) / cell_height
        return y

    def get_x(self, grid_width):
        """
        using grid_width and self.image_width, figure out % of the cell
        :param grid_width:
        :return:
        """
        cell_width = self.image_width / grid_width
        x = (self.defect_midpoint_x % cell_width) / cell_width
        return x

    def get_h(self, grid_height):
        """
        using grid_width and self.image_width, figure out % of the cell
        :param grid_height:
        :return:
        """
        cell_height = self.image_height / grid_height
        h = self.defect_height / cell_height
        return h

    def get_w(self, grid_width):
        """
        :param grid_width:
        :return:
        """
        cell_width = self.image_width / grid_width
        w = self.defect_width / cell_width
        return w

    def add_to_grid(self, grid):
        grid_height, grid_width, grid_depth = grid.shape
        softmax_defects = self.to_softmax_bin(self.defect_type, self.all_defect_types).tolist()
        y_idx = self.get_y_index(grid_height)
        x_idx = self.get_x_index(grid_width)
        y = [self.get_y(grid_height)]
        x = [self.get_x(grid_width)]
        h = [self.get_h(grid_height)]
        w = [self.get_w(grid_width)]
        vector = np.array([1] + softmax_defects + y + x + h + w)
        try:
            grid[y_idx, x_idx] = vector
        except Exception:
            print("Couldn't add to grid, probably out of bounds")
        return grid

    @staticmethod
    def to_softmax_bin(el, el_list):
        return (el == np.array(el_list, object)).astype(int)


class Defects:
    def __init__(self, config, json_defects, image_height, image_width):
        self.defects = [Defect(config, json_defect, image_height, image_width)
                        for json_defect in json_defects]

    def generate_grid(self, grid):
        for defect in self.defects:
            grid = defect.add_to_grid(grid)
        return grid
