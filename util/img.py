import numpy as np
import scipy.misc
import scipy.signal
import skimage.draw
import math
from PIL import Image, ImageDraw

# =============================================================================
# General image processing functions
# =============================================================================

def resize(*args):
    im = args[0]
    if im.ndim == 3 and im.shape[2] > 3:
        res = args[1]
        new_im = np.zeros((res[0], res[1], im.shape[2]), np.float32)
        for i in range(im.shape[2]):
            if im[:,:,i].max() > 0:
                new_im[:,:,i] = resize(im[:,:,i], res)
        return new_im
    else:
        return scipy.misc.imresize(*args).astype(np.float32) / 255

def get_transform(center, scale, res, rot=0):
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0, as_int=True):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    if as_int:
        return new_pt[:2].astype(int)
    else:
        return new_pt[:2]

def crop(img, center, scale, res, rot=0):
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that context is included when rotated
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    return resize(new_img, res)

def nms(img):
    # Do non-maximum suppression on a 2D array
    win_size = 3
    domain = np.ones((win_size, win_size))
    maxes = scipy.signal.order_filter(img, domain, win_size ** 2 - 1)
    diff = maxes - img
    result = img.copy()
    result[diff > 0] = 0
    return result

# =============================================================================
# Drawing and visualization
# =============================================================================

def gauss(x, a, b, c, d=0):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

def color_heatmap(x):
    color = np.zeros((x.shape[0],x.shape[1],3))
    color[:,:,0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
    color[:,:,1] = gauss(x, 1, .5, .3)
    color[:,:,2] = gauss(x, 1, .2, .3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color

def visualize_heatmap(inp, hm, part_idx=0):
    # Assumes an RGB input image, and multi-channel heatmap
    # Upsample heatmap to input size
    tmp_hm = resize(hm[:,:,part_idx], inp.shape[:2])
    tmp_hm = color_heatmap(tmp_hm)
    tmp_img = inp * .3 + tmp_hm * .7
    return tmp_img

def line(img, pt1, pt2, color, width):
    # Draw a line on an image
    # Make sure dimension of color matches number of channels in img
    pt1 = np.array(pt1,dtype=int) if type(pt1) == list else pt1.astype(int)
    pt2 = np.array(pt2,dtype=int) if type(pt2) == list else pt2.astype(int)
    tmp_img = Image.fromarray(img)
    tmp_draw = ImageDraw.Draw(tmp_img)
    tmp_draw.line((pt1[0],pt1[1],pt2[0],pt2[1]), fill=color, width=width)
    return np.array(tmp_img)

def draw_gaussian(img, pt, sigma):
    # Draw a 2D gaussian

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    sub_img = img[img_y[0]:img_y[1], img_x[0]:img_x[1]]
    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(sub_img, g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return img

def draw_text(img, pt, text, color=(0,0,0), draw_bg=True):
    pt = pt.astype(int)
    tmp_img = Image.fromarray(img)
    tmp_draw = ImageDraw.Draw(tmp_img)
    if draw_bg:
        tmp_draw.rectangle([pt[0],pt[1],pt[0]+7*len(text),pt[1]+15],fill=(255,255,255))
    tmp_draw.text([pt[0]+3,pt[1]+3],text,color)
    return np.array(tmp_img)

def draw_circle(img, pt, color, radius):
    # Draw a circle
    # Mostly a convenient wrapper for skimage.draw.circle
    rr, cc = skimage.draw.circle(pt[1], pt[0], radius, img.shape)
    img[rr, cc] = color
    return img

def draw_polygon(img, pts, color):
    x_vals = [int(pts[i][0]) for i in range(len(pts))]
    y_vals = [int(pts[i][1]) for i in range(len(pts))]
    rr, cc = skimage.draw.polygon(y_vals, x_vals, img.shape)
    img[rr, cc] = color
    return img

def draw_bbox(img, pts, color, width):
    img = line(img, [pts[0], pts[1]], [pts[0], pts[3]], color, width)
    img = line(img, [pts[0], pts[1]], [pts[2], pts[1]], color, width)
    img = line(img, [pts[0], pts[3]], [pts[2], pts[3]], color, width)
    img = line(img, [pts[2], pts[1]], [pts[2], pts[3]], color, width)
    return img
