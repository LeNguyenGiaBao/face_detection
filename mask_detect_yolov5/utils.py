def get_bbox(x1, y1, x2, y2, img_width, img_height, padding_ratio):
    width_box = x2 - x1
    height_box = y2 - y1 

    padding_width = width_box * padding_ratio
    padding_height = height_box * padding_ratio

    x = x1 - padding_width
    if x < 0:
        x = 0

    y = y1 - padding_height
    if y < 0:
        y = 0

    width = width_box + 2 * padding_width
    height = height_box + 2 * padding_height

    x2 = x2 + padding_width
    if x2 > img_width:
        x2 = img_width - 1