import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class DeltaXYZWLHRBBoxCoder(BaseBBoxCoder):
    """Bbox Coder for 3D boxes.

    Args:
        code_size (int): The dimension of boxes to be encoded.
    """

    def __init__(self, code_size=7):
        super(DeltaXYZWLHRBBoxCoder, self).__init__()
        self.code_size = code_size

    @staticmethod
    def encode(src_boxes, dst_boxes):
        """Get box regression transformation deltas (xt, yt, zt, dxt, dyt, dzt,
        rt, dv*) that can be used to transform the `src_boxes` into the
        `target_boxes`.

        Args:
            src_boxes (torch.Tensor): source boxes, e.g., object proposals.
            dst_boxes (torch.Tensor): target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas.
        """
        box_ndim = src_boxes.shape[-1]
        cas, cgs, cts = [], [], []
        if box_ndim > 7:
            xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(
                src_boxes, 1, dim=-1)
            xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(
                dst_boxes, 1, dim=-1)
            cts = [g - a for g, a in zip(cgs, cas)]
        else:
            xa, ya, za, dxa, dya, dza, ra = torch.split(src_boxes, 1, dim=-1)
            xg, yg, zg, dxg, dyg, dzg, rg = torch.split(dst_boxes, 1, dim=-1)
        za = za + dza / 2
        zg = zg + dzg / 2
        diagonal = torch.sqrt(dya**2 + dxa**2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        rt = rg - ra
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, rt, *cts], dim=-1)

    @staticmethod
    def decode(anchors, deltas):
        """Apply transformation `deltas` (dx, dy, dz, dw, dh, dl, dr, dv*) to
        `boxes`.

        Args:
            anchors (torch.Tensor): Parameters of anchors with shape (N, 7).
            deltas (torch.Tensor): Encoded boxes with shape
                (N, 7+n) [x, y, z, w, l, h, r, velo*].

        Returns:
            torch.Tensor: Decoded boxes.
        """
        cas, cts = [], []
        box_ndim = anchors.shape[-1]
        if box_ndim > 7:
            xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(
                anchors, 1, dim=-1)
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(
                deltas, 1, dim=-1)
        else:
            xa, ya, za, dxa, dya, dza, ra = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, dxt, dyt, dzt, rt = torch.split(deltas, 1, dim=-1)

        za = za + dza / 2
        diagonal = torch.sqrt(dya**2 + dxa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza
        rg = rt + ra
        zg = zg - dzg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)
