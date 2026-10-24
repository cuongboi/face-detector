"""numpy BGR frames -> CGImage for Apple Vision, without an encode/decode round trip."""

import cv2
import numpy as np
import Quartz


def bgr_to_cgimage(frame_bgr: np.ndarray):
    """Wrap a BGR uint8 frame as an RGBX CGImage (the pixels are copied once into CFData)."""
    h, w = frame_bgr.shape[:2]
    rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
    provider = Quartz.CGDataProviderCreateWithCFData(rgba.tobytes())
    return Quartz.CGImageCreate(
        w, h, 8, 32, w * 4,
        Quartz.CGColorSpaceCreateDeviceRGB(),
        Quartz.kCGImageAlphaNoneSkipLast | Quartz.kCGBitmapByteOrderDefault,
        provider, None, False, Quartz.kCGRenderingIntentDefault,
    )
