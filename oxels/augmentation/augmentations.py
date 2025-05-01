import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


##
# All classes here are callables that accept an image (np.array of floats) and return an *aligned* image of the same shape
##

class HueShift:
    def __init__(self, max_shift = 0.5):
        self.max_shift = max_shift

    def __call__(self, image):

        """
        Apply a random linear “phase shift” (gradient) to the hue channel of an RGB image.

        Parameters
        ----------
        image : ndarray, shape (H, W, 3), dtype=float (0 to 1)
            Input image in RGB color space.
        max_shift : float
            Maximum hue‐shift fraction in [0,1]. 1.0 corresponds to a full wrap of the hue wheel.

        Returns
        -------
        out_rgb : ndarray, shape (H, W, 3), dtype=float
            The hue‐shifted RGB image.
        """

        hsv = cv2.cvtColor((image*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        h_norm = h / 179.0       # normalize hue to [0,1)

        # 2) Generate random gradient parameters
        H, W = h.shape
        intercept = np.random.uniform(-self.max_shift, self.max_shift)
        slope_x    = np.random.uniform(-self.max_shift, self.max_shift)
        slope_y    = np.random.uniform(-self.max_shift, self.max_shift)

        # 3) Build 2D linear gradient in [−max_shift, +max_shift]
        xv = np.linspace(0, 1, W, dtype=np.float32)
        yv = np.linspace(0, 1, H, dtype=np.float32)
        grad = intercept + slope_x * xv[np.newaxis, :] + slope_y * yv[:, np.newaxis]

        # 4) Apply gradient to normalized hue, wrap around with modulo
        h_shifted = (h_norm + grad) % 1.0
        h_new     = (h_shifted * 179.0).astype(np.float32)

        # 5) Merge back and convert HSV → RGB
        hsv[..., 0] = h_new
        hsv[..., 1] = s
        hsv[..., 2] = v

        out_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return out_rgb/255
    
class MotionBlur:
    def __init__(self, L_min=4, L_max=33, s_min=1, s_max=11):
        self.L_min = L_min
        self.L_max = L_max
        self.s_min = s_min
        self.s_max = s_max

    def get_motion_kernel(self):
        L = np.random.randint(self.L_min, self.L_max)
        x = np.cumsum(np.cumsum(np.random.normal(0,1,L)))
        y = np.cumsum(np.cumsum(np.random.normal(0,1,L)))

        x -= np.min(x)
        y -= np.min(y)
        wf = int(np.max(x)) + 1
        hf = int(np.max(y)) + 1

        sigma = np.random.randint(self.s_min,self.s_max)
        F = np.zeros((hf+10*sigma, wf+10*sigma))

        F[y.astype(int)+5*sigma, x.astype(int)+5*sigma] = 1
        
        K = cv2.resize(gaussian_filter(F, sigma), dsize = (0, 0), fx = 0.1, fy=0.1)

        X,Y = np.meshgrid(np.arange(K.shape[1]), np.arange(K.shape[0]))
        cx = np.sum(K*X)/np.sum(K)
        cy = np.sum(K*Y)/np.sum(K)

        py = int(np.round(K.shape[0]/2 - cy))
        px = int(np.round(K.shape[1]/2 - cx))

        return np.pad(K, ((max(0,py),max(0,-py)), (max(0,px),max(0,-px))))/np.sum(K)


    def __call__(self, im):
        K = self.get_motion_kernel()
        return cv2.filter2D(im, -1, K)
    
class RandomKernel:
    def __init__(self, max_size=3):
        self.max_size = max_size

    def __call__(self, im):
        immin, immax = im.min(), im.max()
        sizex, sizey = np.random.randint(1, self.max_size+1, 2)

        #kernel = np.random.random((sizex, sizey))-0.5
        kernel = np.random.randint(-2,3,(sizex, sizey)).astype(float)
        
        if np.sum(np.abs(kernel)) == 0:
            return im

        # Normalize kernel to prevent overflow or weird brightness
        kernel /= np.sum(np.abs(kernel))

        # Apply the kernel using filter2D
        im = cv2.filter2D(im, -1, kernel)
        im -= im.min()
        im /= im.max()
        im *= (immax - immin)
        im += immin
        return im
    
class RandomBrightnessContrast:
    def __init__(self, contrast_range=(0.5, 1.5), brighntess_range=(-0.2,0.2)):
        self.contrast_range = contrast_range
        self.brighntess_range = brighntess_range

    def __call__(self, im):
        contrast = np.random.uniform(*self.contrast_range)
        brightness = np.random.uniform(*self.brighntess_range)

        return np.clip(im*contrast + brightness, 0, 1)
    
class PixelNoise:
    def __init__(self, sigma=0.05):
        self.sigma = sigma

    def __call__(self, im):
        return np.clip(im + np.random.normal(0, self.sigma, im.shape), 0, 1)
    
class Sobel():
    def __call__(self, im):
        return np.sqrt(cv2.Sobel(im, cv2.CV_32F, 0, 1)**2 + cv2.Sobel(im, cv2.CV_32F, 1, 0)**2)
    
class Scharr():
    def __call__(self, im):
        return np.sqrt(cv2.Scharr(im, cv2.CV_32F, 0, 1)**2 + cv2.Scharr(im, cv2.CV_32F, 1, 0)**2)