def generate_anchors(feature_size: int, image_size: float, scales: list[float], aspect_ratios: list[float]) -> list[list[float]]:
    """
    Returns a list of [x1, y1, x2, y2] anchor boxes.
    """
    stride = (image_size/feature_size)
    anchors=[]

    for i in range(feature_size):
        for j in range(feature_size):
            cy = (i+.5)*stride
            cx = (j+.5)*stride
            for s in scales:
                for a in aspect_ratios:
                    w = s * (a)**.5
                    h = s/(a)**.5
                    anchors.append([cx-(w/2),cy-(h/2), cx+(w/2), cy+(h/2)])

    return anchors
                    