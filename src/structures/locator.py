from __future__ import annotations

import cv2
import math
import numpy as np
import scipy.ndimage
from typing import List, Union, Optional
from pycocotools import mask as masktools


class Locator:
    """
    Generic class to indicate a localization, e.g. a Point, a Box, a Mask ...
    """
    def __init__(self, *args, **kwargs):
        self.coordinates = None  # A list of integers/floats
        self.center = None       # A Point indicating the coarse 2D localization of the locator
        self.box = None          # A Box indicating the coarse 4D localization of the locator

    @property
    def area(self) -> float:
        raise NotImplementedError("The method area() is not implemented yet!")

    def merge(self, others: List[Locator], **kwargs):
        """
        A method to merge multiples Locators
        """
        # This is specific to a locator
        raise NotImplementedError("The method merge() is not implemented yet!")


class Point(Locator):
    """
    A class used to defined a 2D-point.
    """

    def __init__(self, x: float, y: float):
        """
        :param x: horizontal coordinate
        :param y: vertical coordinate
        """
        super().__init__()

        self.x = x
        self.y = y

        self.coordinates = [self.x, self.y]
        self.center = self
    
    @property
    def area(self) -> float:
        return 1.0

    def d1(self, other: Locator) -> float:
        """
        L1 distance with another Point/Box/Mask based on their center
        Values of d1 distance from the point x :
            |4|3|2|3|4|
            |3|2|1|2|3|
            |2|1|x|1|2|

        :param other: another locator (Point/Box/Mask)
        """

        # Case of a Point-(Box/Mask) L1 distance
        # It that case, the other reference is the center of the Box/Mask
        if not isinstance(other, Point):
            return self.d1(other.center)
        # Case of a Point-Point L1 distance
        else:
            return abs(self.x - other.x) + abs(self.y - other.y)

    def d2(self, other: Locator) -> float:
        """
        L2 distance with another Point/Box/Mask based on their center
        :param other: another locator (Point/Box/Mask)
        """

        # Case of a Point-(Box/Mask) L2 distance
        # It that case, the other reference is the center of the Box/Mask
        if not isinstance(other, Point):
            return self.d2(other.center)
        # Case of a Point-Point L2 distance
        else:
            return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def dWindow(self, other: Locator) -> float:
        """
        Distance between two points based on a window distance
        :param other: another locator (Point/Box/Mask)

        Values of dWindow distance from the point x :
            |2|2|2|2|2|
            |2|1|1|1|2|
            |2|1|x|1|2|
        """
        if not isinstance(other, Point):
            return self.dWindow(other.center)
        # Case of a Point-Point L2 distance
        return max(abs(self.x - other.x), abs(self.y - other.y))

    def projection(self, img_width: int, img_height: int) -> Optional[Point]:
        """
        Returns the Point corresponding to a projection if inside the image, otherwise None
        This function is useful when the method shift() is used

        :param img_width: width of the image
        :param img_height: height of the image
        """

        if (self.x < 0) or (self.x >= img_width) or (self.y < 0) or (self.y >= img_height):
            return None
        else:
            return self

    def shift(self, direction: Point, img_width: int, img_height: int) -> Optional[Point]:
        """
        Returns the Point corresponding to a displacement if inside the image, otherwise None

        :param direction: the displacement as a 2D-vector, represented by a Point
        :param img_width: width of the image
        :param img_height: height of the image
        """

        point = self + direction
        return point.projection(img_width, img_height)

    def get_tl(self, points: List[Point]) -> Point:
        """
        Returns the top-left Point between multiple Point

        :param points: list of Point
        """

        tl_x = min([self.x] + [point.x for point in points])
        tl_y = min([self.y] + [point.y for point in points])

        return Point(x=tl_x, y=tl_y)

    def get_br(self, points: List[Point]) -> Point:
        """
        Returns the bottom-right Point between multiple Point

        :param points: list of Point
        """

        br_x = max([self.x] + [point.x for point in points])
        br_y = max([self.y] + [point.y for point in points])

        return Point(x=br_x, y=br_y)

    def merge(self, others: List[Point], **kwargs) -> Point:
        """
        Returns the average Point between multiple Point as a merged version

        :param others: list of Point
        """

        n_points = len(others) + 1

        x_tot = self.x + sum([p.x for p in others])
        y_tot = self.y + sum([p.y for p in others])

        return Point(x=x_tot/n_points, y=y_tot/n_points)

    def __str__(self) -> str:
        text = f"The point is located @({self.x}, {self.y})"
        return text

    def __add__(self, other: Point) -> Point:
        return Point(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(x=self.x - other.x, y=self.y - other.y)

    def __mul__(self, coeff: float) -> Point:
        return Point(x=self.x*coeff, y=self.y*coeff)

    def __truediv__(self, coeff: float) -> Point:
        return Point(x=self.x/coeff, y=self.y/coeff)

    def __eq__(self, other: Optional[Point]) -> bool:
        if not isinstance(other, Point):
            return False
        else:
            return (self.x - other.x < 1e-5) & (self.y - other.y < 1e-5)

    def __lt__(self, other: Point) -> bool:
        """
        Non total binary relation
        """
        if self.x == other.x:
            return self.y < other.y
        elif self.y == other.y:
            return self.x < other.x
        else:
            return (self.x < other.x) & (self.y < other.y)

    def __le__(self, other: Point) -> bool:
        return (self.x <= other.x) & (self.y <= other.y)

    def __gt__(self, other: Point) -> bool:
        """
        Non total binary relation
        """
        if self.x == other.x:
            return self.y > other.y
        elif self.y == other.y:
            return self.x > other.x
        else:
            return (self.x > other.x) & (self.y > other.y)

    def __ge__(self, other: Point) -> bool:
        return (self.x >= other.x) & (self.y >= other.y)


class Box(Locator):
    """
    A class used to defined a 2D-Box aligned with the cartesian axis
    """

    def __init__(self, coordinates: List[Union[int, float]]):
        """
        :param coordinates: list of extreme coordinates [xmin ymin xmax ymax]. This coordinates are included.
        """
        super().__init__()

        assert len(coordinates) == 4, "coordinates does not contain exactly four numbers"

        self.coordinates = coordinates

        # the values are in [xmin, xmax] included
        self.xmin = coordinates[0]
        self.ymin = coordinates[1]
        self.xmax = coordinates[2]
        self.ymax = coordinates[3]

        self.tl = Point(x=self.xmin, y=self.ymin)
        self.br = Point(x=self.xmax, y=self.ymax)

        assert self.tl <= self.br, f"The box @{self.coordinates} is not a correct box."

        self.width = self.xmax - self.xmin + 1
        self.height = self.ymax - self.ymin + 1
        self.center = (self.tl + self.br) / 2
        self.box = self

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_mask(self, img_width: Optional[int] = None, img_height: Optional[int] = None) -> Mask:
        """
        Returns a Mask version of this Box
        Crop it to remove out-of-the-image pixels
        """

        if (img_width is None) and (img_height is None):
            projected_box = self
        else:
            projected_box = self.projection(img_width, img_height)
            assert projected_box is not None, f"The box was originally out of the image : {self}"

        return Mask.from_binmap(tl=projected_box.tl, bin_map=np.ones((projected_box.height, projected_box.width), dtype=bool),
                                img_width=img_width, img_height=img_height)

    def to_binmap(self, img_width: int, img_height: int) -> np.ndarray:
        """
        Returns a binary 2D-array with the provided shape

        :param img_width: width of the image
        :param img_height: height of the image
        """

        bin_map = np.zeros((img_height, img_width), dtype=bool)
        bin_map[int(self.ymin):int(self.ymax + 1), int(self.xmin):int(self.xmax + 1)] = True

        return bin_map

    def projection(self, img_width: int, img_height: int) -> Optional[Box]:
        """
        Returns the part of the Box which is inside an image, otherwise None
        This function is useful when the method shift() and add_noise() are used

        :param img_width: img_width of the image
        :param img_height: img_height of the image
        """

        # Box outside of the image
        if (self.br.x < 0) or (self.tl.x >= img_width) or (self.br.y < 0) or (self.tl.y >= img_height):
            return None
        else:
            xmin, ymin, xmax, ymax = self.coordinates
            if self.xmin < 0:
                xmin = 0
            if self.ymin < 0:
                ymin = 0
            if self.xmax >= img_width:
                xmax = img_width - 1
            if self.ymax >= img_height:
                ymax = img_height - 1

            return Box([xmin, ymin, xmax, ymax])

    def shift(self, direction: Point, img_width: int, img_height: int) -> Optional[Box]:
        """
        Returns the part of the Box corresponding to a displacement if inside the image, otherwise None

        :param direction: the displacement as a 2D-vector as a Point
        :param img_width: width of the image
        :param img_height: height of the image
        """

        tl = self.tl + direction
        br = self.br + direction

        box = Box([tl.x, tl.y, br.x, br.y])

        return box.projection(img_width, img_height)

    def merge(self, others: List[Box], **kwargs) -> Box:
        """
        Return the smallest box that contains all the boxes
        """

        tl = self.tl.get_tl([box.tl for box in others])
        br = self.br.get_br([box.br for box in others])

        return Box([tl.x, tl.y, br.x, br.y])

    def d1(self, other: Locator) -> float:
        """
        L1 distance with another Point/Box/Mask

        :param other: another locator (Point/Box/Mask)
        """
        return self.center.d1(other.center)

    def d2(self, other: Locator) -> float:
        """
        L2 distance with another Point/Box/Mask

        :param other: another locator (Point/Box/Mask)
        """
        return self.center.d2(other.center)

    def IoU(self, other: Optional[Union[Box, Mask]]) -> float:
        """
        Computes the intersection over union (IOU, aka Jaccard index) at the box level
        The IoU is a value between 0 and 1
        """

        if other is None:
            return 0.0

        if isinstance(other, Mask):
            return self.IoU(other.box)

        assert isinstance(other, Box), f"The other in not a Box: {type(other)}"

        # Intersection
        xmin = max(self.xmin, other.xmin)
        ymin = max(self.ymin, other.ymin)
        xmax = min(self.xmax, other.xmax)
        ymax = min(self.ymax, other.ymax)

        intersection_area = (xmax - xmin + 1) * (ymax - ymin + 1) if (xmin <= xmax) & (ymin <= ymax) else 0.0

        union_area = self.area + other.area - intersection_area

        return float(intersection_area) / float(union_area)

    def mIoU(self, other: Optional[Union[Box, Mask]]) -> float:
        """
        Computes the mask Intersection over Union (mIoU) with another locator.
        """

        if other is None:
            return 0.0

        if isinstance(other, Box):
            return self.IoU(other)
        elif isinstance(other, Mask):
            return other.mIoU(self)
            
    def buffered(self, b: float) -> Box:
        """
        Return a Box whose width and height have been inflated by a coefficient b
        This is iseful to compute BIoU
        """
        return Box(coordinates=[self.box.xmin-b*self.box.width/2, 
                                self.box.ymin-b*self.box.height/2,
                                self.box.xmax+b*self.box.width/2, 
                                self.box.ymax+b*self.box.height/2])
                                         
    def BIoU(self, other: Optional[Box], b: float) -> float:
        """
        Computes the Buffered Intersection over Union (BIoU) at the box level
        ref: https://arxiv.org/abs/2211.14317
        """
        
        if other is None:
            return 0.0
        
        buffered_self = self.buffered(b=b)
        buffered_other = other.box.buffered(b=b)
        return buffered_self.IoU(buffered_other)

    def GIoU(self, other: Optional[Box]) -> float:
        """
        Computes the Generalized Intersection over Union (GIoU) at the box level
        ref: https://arxiv.org/abs/1902.09630
        """

        if other is None:
            return 0.0

        # Calculate intersection coordinates
        intersection_xmin = max(self.xmin, other.xmin)
        intersection_ymin = max(self.ymin, other.ymin)
        intersection_xmax = min(self.xmax, other.xmax)
        intersection_ymax = min(self.ymax, other.ymax)

        # Calculate intersection area
        intersection_area = max(0, intersection_xmax - intersection_xmin + 1) * max(0, intersection_ymax - intersection_ymin + 1)

        # Calculate union area
        union_area = self.area + other.area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area

        # Smallest covering box 
        smallest_covering_area = (max(self.xmax, other.xmax) - min(self.xmin, other.xmin) + 1) * (max(self.ymax, other.ymax) - min(self.ymin, other.ymin) + 1)

        # Calculate GIoU
        giou = iou - (smallest_covering_area - union_area) / smallest_covering_area

        return giou
    
    def DIoU(self, other: Optional[Box]) -> float:
        """
        Computes the Distance-IoU (DIoU) at the box level
        ref: https://arxiv.org/abs/1911.08287
        """

        if other is None:
            return 0.0
        
        # Calculate intersection coordinates
        intersection_xmin = max(self.xmin, other.xmin)
        intersection_ymin = max(self.ymin, other.ymin)
        intersection_xmax = min(self.xmax, other.xmax)
        intersection_ymax = min(self.ymax, other.ymax)

        # Calculate intersection area
        intersection_area = max(0, intersection_xmax - intersection_xmin + 1) * max(0, intersection_ymax - intersection_ymin + 1)

        # Calculate union area
        union_area = self.area + other.area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area

        # Smallest covering box 
        smallest_covering_box = Box([min(self.xmin, other.xmin), min(self.ymin, other.ymin), max(self.xmax, other.xmax), max(self.ymax, other.ymax)])

        # Calculate DIoU
        diou = iou - self.center.d2(other.center) ** 2 / (smallest_covering_box.width ** 2 + smallest_covering_box.height ** 2)

        return diou

    def sIoU(self, other: Optional[Union[Box, Mask]]) -> float:
        """
        Computes the signed intersection over union at the box level
        Ref : https://arxiv.org/abs/1905.12365
        """

        if other is None:
            return 0.0

        # Intersection
        xmin = max(self.xmin, other.xmin)
        ymin = max(self.ymin, other.ymin)
        xmax = min(self.xmax, other.xmax)
        ymax = min(self.ymax, other.ymax)
        
        # Intersection area that can be either positive or negative
        intersection_width = xmax - xmin + 1
        intersection_height = ymax - ymin + 1
        abs_intersection_area = abs(intersection_width) * abs(intersection_height)

        if intersection_width >= 0 and intersection_height >= 0:
            return abs_intersection_area / (self.area + other.area - abs_intersection_area)
        else:
            return -abs_intersection_area / (self.area + other.area + abs_intersection_area)
    
    def to_box(self):
        return self

    def warp(self, flow: np.ndarray) -> Optional[Box]:
        """
        Return the warped version of a Box if it exists, None otherwise

        :param flow: optical flow of shape (height, width, channel)
        """

        # Transform the mask into an image
        height, width, _ = flow.shape
        whole_bin_map = self.to_binmap(width, height).astype(np.float32)  # a rectangular box in the whole image

        # Remap the flow
        flow = -flow
        flow[:, :, 0] += np.arange(width)
        flow[:, :, 1] += np.arange(height)[:, np.newaxis]

        warped_bin_map = cv2.remap(whole_bin_map, flow, None, cv2.INTER_LINEAR)

        # Get the warped mask
        if warped_bin_map.any():
            where_true = np.where(warped_bin_map)
            ymin, xmin = np.min(where_true[0]), np.min(where_true[1])
            ymax, xmax = np.max(where_true[0]), np.max(where_true[1])
            return Box([xmin, ymin, xmax, ymax])
        else:
            return None

    def add_noise(self, sigma: float, img_width: int, img_height: int) -> Optional[Box]:
        """
        Return a Box within the image which a noisy version of itself, None if the perturbed version is out the image

        :param sigma: standard deviation of the white Gaussian noise
        :param img_width: width of the image
        :param img_height: height of the image
        """

        xmin = self.xmin + np.random.normal(scale=sigma * self.width,  size=1).astype(int)[0]
        xmax = self.xmax + np.random.normal(scale=sigma * self.width,  size=1).astype(int)[0]
        ymin = self.ymin + np.random.normal(scale=sigma * self.height, size=1).astype(int)[0]
        ymax = self.ymax + np.random.normal(scale=sigma * self.height, size=1).astype(int)[0]

        return Box(coordinates=[xmin, ymin, xmax, ymax]).projection(img_width, img_height)

    def __add__(self, other: Union[Point, Box]) -> Box:

        if isinstance(other, Box):
            add_coord = [i + j for i, j in zip(self.coordinates, other.coordinates)]
        elif isinstance(other, Point):
            add_coord = [self.xmin + other.x, self.ymin + other.y, self.xmax + other.x, self.ymax + other.y]

        return Box(coordinates=add_coord)

    def __mul__(self, coeff: float) -> Box:
        return Box([i*coeff for i in self.coordinates])

    def __truediv__(self, coeff: float) -> Box:
        return Box([i/coeff for i in self.coordinates])

    def __eq__(self, other: Optional[Box]) -> bool:
        if not isinstance(other, Box):
            return False
        else:
            return (self.tl == other.tl) & (self.br == other.br)

    def __str__(self):
        text = f"Box located @[{self.xmin} {self.ymin} {self.xmax} {self.ymax}]. Dimension of ({self.width} x {self.height}) for an area of {self.area}."
        return text


class Mask(Locator):
    """
    A class used to defined a 2D binary mask. A Mask is defined by its RLE (run-length encoding) and the size of the image
    NB: a Mask is linked to a image dimension. When the Mask is shifted outside of the image, it disappears (None is returns).

    Two API to initialize a Mask:
        >>> mask1 = Mask(rle='0120', img_width=2, img_height=2)                                                                    # initialized with a rle code
        >>> mask2 = Mask.from_binmap(tl=Point(0, 0), bin_map=np.array([[True, False], [False, True]]), img_width=2, img_height=2)  # initialized with a binary map
        >>> mask1 == mask2  # True
    """

    def __init__(self, rle: str, img_width: int, img_height: int):
        """
        :param rle: compressed run-length encoding of the mask with pycocotools
        :param img_width: width of the image
        :param img_height: height of the image
        """
        super().__init__()

        self.tl = Point(0, 0)
        self.bin_map = np.ascontiguousarray(masktools.decode({'size': (img_height, img_width), 'counts': rle.encode('utf-8')}).astype(bool))  # full size bin_map, with a lot of 0s at the border
        self.__cut_empty_border()  # modify self.bin_map to delete 0s at the border
        self.rle = rle
        self.img_height = img_height   # size of the image
        self.img_width = img_width     # size of the image

        self.height, self.width = self.bin_map.shape
        self.br = self.tl + Point(x=self.width-1, y=self.height-1)

        # the values are in [xmin, xmax] included
        self.xmin, self.ymin = self.tl.x, self.tl.y
        self.xmax, self.ymax = self.br.x, self.br.y
        self.coordinates = [self.xmin, self.ymin, self.xmax, self.ymax]
        self.box = Box(coordinates=self.coordinates)

        # The center of mass. Its center defined by its bounding box is self.box.center
        x_center = (np.sum(self.bin_map, axis=0) / self.area).dot(np.arange(self.width))
        y_center = (np.sum(self.bin_map, axis=1) / self.area).dot(np.arange(self.height))
        self.center = self.tl + Point(x=x_center, y=y_center)
    
    @property
    def area(self) -> float:
        return np.sum(self.bin_map)

    @classmethod
    def from_binmap(cls, tl: Point, bin_map: np.ndarray, img_width: int, img_height: int):
        """
        Creates an instance of the class from a binary map instead of the RLE

        :param tl: The top-left point of the binary map within the image.
        :param bin_map: A 2D numpy array of boolean values representing the binary map.
        :param img_width: width of the image.
        :param img_height: height of the image.

        :return: An instance of the class with the encoded binary map.
        """

        assert isinstance(tl, Point), "tl is not a Point"
        assert bin_map.dtype == 'bool', "The mask is not boolean"
        assert (tl >= Point(0, 0)) and (tl < Point(x=img_width, y=img_height)), f"The Point tl={tl} is out of the image of size ({img_height}x{img_width})"

        if (bin_map.shape[0] != img_height) or (bin_map.shape[1] != img_width):
            bin_map_full = np.zeros((img_height, img_width), dtype=bool)
            height, width = bin_map.shape
            bin_map_full[tl.y:(tl.y + height), tl.x:(tl.x + width)] = bin_map
        else:
            bin_map_full = bin_map

        bin_map_full = np.asfortranarray(bin_map_full)
        rle = masktools.encode(bin_map_full)['counts'].decode("utf-8")

        return cls(rle=rle, img_width=img_width, img_height=img_height)

    def __cut_empty_border(self):
        """
        Removes empty rows or columns that are on a border of the binary map
        This function does not modify the mask, so it is not necessary to update its rle
        This function changes the binary map
        """

        #TODO : optimize it by looking both directions and not the interior
        bin_map_x_axis = np.where(np.sum(self.bin_map, axis=0) != 0)[0]
        bin_map_y_axis = np.where(np.sum(self.bin_map, axis=1) != 0)[0]

        if (len(bin_map_x_axis) == 0) or (len(bin_map_y_axis) == 0):
            # the mask is empty
            self.tl = Point(x=0, y=0)
            self.bin_map = np.array([[True]])  # one pixel to avoid any error

        else:

            width = np.max(bin_map_x_axis) - np.min(bin_map_x_axis) + 1
            height = np.max(bin_map_y_axis) - np.min(bin_map_y_axis) + 1

            min_x = np.min(bin_map_x_axis)
            min_y = np.min(bin_map_y_axis)

            self.tl = Point(x=self.tl.x + min_x, y=self.tl.y + min_y)
            self.bin_map = self.bin_map[int(min_y):int(min_y+height), int(min_x):int(min_x+width)].copy()

    def projection(self, img_width: int, img_height: int) -> Optional[Mask]:
        """
        Returns the part of the Mask which is inside an image, otherwise None
        This function is useful when the method shift() is used

        :param img_width: width of the image
        :param img_height: height of the image
        """

        # Mask englobes the whole image
        if (self.tl <= Point(0, 0)) and (self.br >= Point(img_width-1, img_height-1)):
            return Mask.from_binmap(tl=Point(x=0, y=0), bin_map=self.bin_map[int(-self.ymin):int(img_height-self.ymin), int(-self.xmin):int(img_width-self.xmin)],
                                    img_width=img_width, img_height=img_height)

        # Mask outside of the image
        elif (self.br.x < 0) or (self.tl.x >= img_width) or (self.br.y < 0) or (self.tl.y >= img_height):
            return None

        # Mask inside or a part outside
        else:
            xmin, ymin = self.coordinates[:2]  # coordinates of the top-left point of the bounding box related to the Mask
            x1, y1, x2, y2 = 0, 0, self.bin_map.shape[1], self.bin_map.shape[0]  # coordinates inside the box
            if self.xmin < 0:
                xmin = 0
                x1 -= self.xmin  # reduce the start position
            if self.ymin < 0:
                ymin = 0
                y1 -= self.ymin  # reduce the start position
            if self.xmax >= img_width:
                x2 = img_width - self.xmax - 1  # negative coordinate
            if self.ymax >= img_height:
                y2 = img_height - self.ymax - 1  # negative coordinate

            return Mask.from_binmap(tl=Point(x=xmin, y=ymin), bin_map=self.bin_map[int(y1):int(y2), int(x1):int(x2)],
                                    img_width=img_width, img_height=img_height)

    def shift(self, direction: Point, img_width: int, img_height: int) -> Optional[Mask]:
        """
        Returns the part of the Mask corresponding to a displacement if inside the image, otherwise None

        :param direction: the displacement as a 2D-vector
        :param img_width: width of the image
        :param img_height: height of the image
        """
        mask = Mask.from_binmap(tl=self.tl + direction, bin_map=self.bin_map,
                                img_width=img_width, img_height=img_height)

        return mask.projection(img_width, img_height)

    def to_binmap(self, img_width: int, img_height: int) -> np.ndarray:
        """
        Returns a binary 2D-array with the provided shape

        :param img_width: width of the image
        :param img_height: height of the image
        """

        bin_map = np.zeros((img_height, img_width), dtype=bool)
        bin_map[self.tl.y:(self.tl.y + self.height), self.tl.x:(self.tl.x + self.width)] = self.bin_map

        return bin_map

    def reshape(self, width_or: int, height_or: int, width: int, height: int) -> np.ndarray:
        """
        Reshape a bin_map from the size (width_or, height_or) to (width, height)
        Bilinear interpolation (order = 1)
        """

        bin_map = self.to_binmap(img_width=width_or, img_height=height_or).astype(float)
        return scipy.ndimage.zoom(bin_map, (height/height_or, width/width_or), order=1) > 0.5

    def merge(self, others: List[Mask], **kwargs) -> Mask:
        """
        Merge multiple Masks by considerating all pixels from all Masks
        """

        # The whole area of interest
        # It is faster to only consider this part of the image instead of the whole image (cf to_binmap())
        tl = self.tl.get_tl([other.tl for other in others])
        br = self.br.get_br([other.br for other in others])

        width, height = br.x - tl.x + 1, br.y - tl.y + 1

        fixed_size_bin_map_self = np.zeros((height, width), dtype=bool)
        fixed_size_bin_map_other = np.zeros((height, width), dtype=bool)

        fixed_size_bin_map_self[(self.tl.y - tl.y):(self.tl.y - tl.y + self.height), (self.tl.x - tl.x):(self.tl.x-tl.x+self.width)] = self.bin_map

        bin_map = fixed_size_bin_map_self
        for other in others:
            fixed_size_bin_map_other[(other.tl.y - tl.y):(other.tl.y - tl.y + other.height), (other.tl.x - tl.x):(other.tl.x-tl.x+other.width)] = other.bin_map
            bin_map |= fixed_size_bin_map_other

        return Mask.from_binmap(tl=tl, bin_map=bin_map,
                                img_width=kwargs.get('img_width'), img_height=kwargs.get('img_height'))

    def mIoU(self, other: Optional[Locator]) -> float:
        """
        Computes the mask-intersection over union (mIOU, aka Jaccard index) at the pixel level (between 0 and 1)
        """

        if other is None:
            return 0.0

        if isinstance(other, Box):
            return self.mIoU(other.to_mask(img_height=self.img_height, img_width=self.img_width))

        assert isinstance(other, Mask), f"The other is not a Mask but a {type(other)}"

        # Shortcut to make computation faster
        if self.box.IoU(other.box) == 0:
            return 0.0

        # Pixels that are out of this region are necessarely 'False'
        tl = self.tl.get_tl([other.tl])
        br = self.br.get_br([other.br])

        width, height = br.x - tl.x + 1, br.y - tl.y + 1

        fixed_size_bin_map_self = np.zeros((height, width), dtype=bool)
        fixed_size_bin_map_other = np.zeros((height, width), dtype=bool)

        fixed_size_bin_map_self[(self.tl.y - tl.y):(self.tl.y - tl.y + self.height), (self.tl.x - tl.x):(self.tl.x-tl.x+self.width)] = self.bin_map
        fixed_size_bin_map_other[(other.tl.y - tl.y):(other.tl.y - tl.y + other.height), (other.tl.x - tl.x):(other.tl.x-tl.x+other.width)] = other.bin_map

        intersection_area = np.sum(fixed_size_bin_map_self & fixed_size_bin_map_other)
        union_area = np.sum(fixed_size_bin_map_self | fixed_size_bin_map_other)

        return intersection_area / union_area

    def IoU(self, other: Optional[Locator]) -> float:
        """
        Computes the intersection over union (IOU, aka Jaccard index) at the box level (between 0 and 1)
        """

        if other is None:
            return 0.0

        assert isinstance(other, (Box, Mask)), f"The other is not a Mask or a Box but a {type(other)}"
        return self.box.IoU(other.box)

    def mIoM(self, other: Optional[Union[Box, Mask]]) -> float:
        """
        Computes the mask Intersection over Minimum
        ref : https://arxiv.org/abs/2007.03200
        """

        if other is None:
            return 0.0

        # Shortcut to make computation faster, especially for non linked Masks
        if self.box.IoU(other.box) == 0:
            return 0

        # Here, both Masks share some common parts at box-level
        if isinstance(other, Box):
            other = other.to_mask(img_height=None, img_width=None)

        # Pixels that are out of this region are necessarely 'False'
        tl = self.tl.get_tl([other.tl])
        br = self.br.get_br([other.br])

        width, height = br.x - tl.x + 1, br.y - tl.y + 1

        fixed_size_bin_map_self = np.zeros((height, width), dtype=bool)
        fixed_size_bin_map_other = np.zeros((height, width), dtype=bool)

        fixed_size_bin_map_self[(self.tl.y - tl.y):(self.tl.y - tl.y + self.height), (self.tl.x - tl.x):(self.tl.x-tl.x+self.width)] = self.bin_map
        fixed_size_bin_map_other[(other.tl.y - tl.y):(other.tl.y - tl.y + other.height), (other.tl.x - tl.x):(other.tl.x-tl.x+other.width)] = other.bin_map

        intersection_area = np.sum(fixed_size_bin_map_self & fixed_size_bin_map_other)
        min_area = min(np.sum(fixed_size_bin_map_self),  np.sum(fixed_size_bin_map_other))

        return intersection_area / min_area
        
    def BIoU(self, other: Optional[Union[Box, Mask]], b: float) -> float:
        return self.box.BIoU(other.box, b=b)
    
    def GIoU(self, other: Optional[Union[Box, Mask]]) -> float:
        return self.box.GIoU(other.box)   
        
    def DIoU(self, other: Optional[Union[Box, Mask]]) -> float:
        return self.box.DIoU(other.box)

    def d1(self, other: Union[Point, Box, Mask]) -> float:
        return self.center.d1(other.center)

    def d2(self, other: Union[Point, Box, Mask]) -> float:
        return self.center.d2(other.center)

    def warp(self, flow: np.ndarray) -> Optional[Mask]:
        """
        Return the warped version of a Mask if it exists, None otherwise

        :param flow: optical flow of shape (height, width, channel)
        """

        # Transform the mask into an image
        img_height, img_width, _ = flow.shape
        whole_bin_map = self.to_binmap(img_width, img_height).astype(np.float32)

        # Remap the flow
        flow = -flow
        flow[:, :, 0] += np.arange(img_width)
        flow[:, :, 1] += np.arange(img_height)[:, np.newaxis]

        warped_bin_map = cv2.remap(whole_bin_map, flow, None, cv2.INTER_LINEAR)

        # Get the warped mask
        if warped_bin_map.any():
            return Mask.from_binmap(Point(0, 0), bin_map=warped_bin_map.astype(bool),
                                    img_width=img_width, img_height=img_height)
        else:
            return None

    def to_box(self) -> Box:
        return Box([self.xmin, self.ymin, self.xmax, self.ymax])

    def to_mask(self, **kwargs) -> Mask:
        return self

    def contours(self) -> List[Point]:
        """
        Returns the list of Points that are in the contours of this Mask
        """

        bin_map = self.bin_map.astype('uint8')
        contours, _ = cv2.findContours(bin_map, mode=1, method=cv2.CHAIN_APPROX_NONE)
        # NB: The method cv2.CHAIN_APPROX_NONE stores absolutely all the contour points and not a sample of them

        list_point_contours = []
        for contour in contours:
            for point in contour:
                list_point_contours += [Point(x=self.tl.x + point[0][0], y=self.tl.y + point[0][1])]

        return list_point_contours

    def show(self, size=5):

        import seaborn as sns
        import matplotlib.pyplot as plt

        print(self)

        # Adjust the labels to get the absolute position of the binary map
        x_step = (self.xmax - self.xmin) // 10
        y_step = (self.ymax - self.ymin) // 10
        xlabels = [x if x % x_step == 0 else None for x in np.arange(self.xmin, self.xmax, 1)]
        ylabels = [y if y % y_step == 0 else None for y in np.arange(self.ymin, self.ymax, 1)]

        plt.figure(figsize=(size * self.width/self.height, size))
        sns.heatmap(self.bin_map, cbar=False, xticklabels=xlabels, yticklabels=ylabels)
        plt.title(f"Mask : area = {self.area}, coverage = {self.area / self.box.area:0.3f}")
        plt.plot()

    def __add__(self, other: Point, img_width: int, img_height: int) -> Mask:

        assert isinstance(other, Point), f"Cannot add a Mask to a {other.__class__}"

        return Mask.from_binmap(tl=self.tl+other, bin_map=self.bin_map,
                                img_width=img_width, img_height=img_height)

    def __str__(self) -> str:
        text = f"Mask @{self.box.coordinates} of size ({self.width}, {self.height}) centered @({self.center.x}, {self.center.y})"
        return text

    def __eq__(self, other: Optional[Mask]) -> bool:
        if not isinstance(other, Mask):
            return False
        else:
            return self.rle == other.rle