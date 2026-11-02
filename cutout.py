"""立绘抠图：从白底（含淡灰地面阴影）里把角色抠成透明 PNG。

思路：
    1. 生成"背景候选"掩膜（近白 或 低饱和浅灰）
    2. 用连通域标记，只保留「与画面四边连通」的那一块作为背景 ——
       角色身上那块白围裙被深色描边包住，不属于外边连通域，不会被误删
    3. alpha 边缘做 1px 收缩 + 轻微模糊，去掉白边

用法：
    python cutout.py 输入.png 输出.png
"""
import sys

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


def cutout(src: str, dst: str, white_thr: int = 232, gray_thr: int = 203,
           sat_thr: int = 26) -> None:
    im = Image.open(src).convert("RGB")
    a = np.asarray(im).astype(np.int16)

    mn = a.min(axis=2)
    sat = a.max(axis=2) - mn

    cand = (mn > white_thr) | ((mn > gray_thr) & (sat < sat_thr))

    lab, n = ndimage.label(cand)
    border_ids = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    border_ids = border_ids[border_ids != 0]
    bg = np.isin(lab, border_ids)

    alpha = np.where(bg, 0, 255).astype(np.uint8)
    al = Image.fromarray(alpha).filter(ImageFilter.MinFilter(3))
    al = al.filter(ImageFilter.GaussianBlur(0.7))

    out = im.convert("RGBA")
    out.putalpha(al)
    bbox = out.getbbox()
    if bbox:
        out = out.crop(bbox)
    out.save(dst)
    print(f"ok  {dst}  {out.size[0]}x{out.size[1]}  "
          f"背景 {bg.sum()} px ({bg.sum() / bg.size:.1%})")


if __name__ == "__main__":
    cutout(sys.argv[1], sys.argv[2])
