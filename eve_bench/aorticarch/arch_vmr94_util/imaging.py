from typing import List, Tuple
from PIL import Image, ImageDraw, ImageChops
import numpy as np
from .simulation import Simulation


class Imaging:
    def __init__(
        self,
        intervention: Simulation,
        image_size: Tuple[int, int],
        dimension_to_omit: str = "y",
    ) -> None:
        self.intervention = intervention
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.dimension_to_omit = dimension_to_omit
        self._dim_del = 1
        self.image_mode = "L"
        self.low = 0
        self.high = 255

        tracking_high = self.intervention.tracking_space.high
        tracking_low = self.intervention.tracking_space.low
        tracking_high = np.delete(tracking_high, self._dim_del, axis=-1)
        tracking_low = np.delete(tracking_low, self._dim_del, axis=-1)
        maze_size_x = tracking_high[0] - tracking_low[0]
        maze_size_y = tracking_high[1] - tracking_low[1]
        x_factor = self.image_size[0] / (maze_size_x)
        y_factor = self.image_size[1] / (maze_size_y)
        self.maze_to_image_factor = min(x_factor, y_factor)
        self.vessel_offset = np.array([-tracking_low[0], -tracking_low[1]])
        x_image_offset = (
            self.image_size[0] - maze_size_x * self.maze_to_image_factor
        ) / 2
        y_image_offset = (
            self.image_size[1] - maze_size_y * self.maze_to_image_factor
        ) / 2
        self.image_offset = np.array([x_image_offset, y_image_offset])

    def get_image(self):
        # Noise is around colour 128.
        noise_image = Image.effect_noise(size=self.image_size, sigma=5)
        physics_image = self._render()
        image = ImageChops.darker(physics_image, noise_image)
        return image

    def _render(self) -> None:
        physics_image = Image.new(mode=self.image_mode, size=self.image_size, color=255)
        tracking = np.delete(self.intervention.tracking, self._dim_del, axis=-1)
        diameter = int(
            np.round(self.intervention.device_diameter * self.maze_to_image_factor)
        )

        self._draw_lines(
            physics_image,
            tracking,
            int(diameter),
            colour=40,
        )
        return physics_image

    def _draw_lines(
        self,
        image: Image.Image,
        point_cloud: np.ndarray,
        width=1,
        colour=0,
    ):

        draw = ImageDraw.Draw(image)
        point_cloud_image = self._coord_transform_tracking_to_image(point_cloud)
        draw.line(point_cloud_image, fill=colour, width=width, joint="curve")
        return image

    def _coord_transform_tracking_to_image(
        self, coords: np.ndarray
    ) -> List[Tuple[float, float]]:

        coords_image = (coords + self.vessel_offset) * self.maze_to_image_factor
        coords_image += self.image_offset
        coords_image = np.round(coords_image, decimals=0).astype(np.int64)
        coords_image[:, 1] = -coords_image[:, 1] + self.image_size[1]
        coords_image = [(coord[0], coord[1]) for coord in coords_image]
        return coords_image
