def get_bbox(x1, y1, x2, y2, img_width, img_height, padding_ratio):
    width_box = x2 - x1
    height_box = y2 - y1 

    padding_width = width_box * padding_ratio
    padding_height = height_box * padding_ratio

    x = int(x1 - padding_width)
    if x < 0:
        x = 0

    y = int(y1 - padding_height)
    if y < 0:
        y = 0

    x2 = x2 + padding_width
    if x2 >= img_width:
        x2 = img_width - 1

    y2 = y2 + padding_height
    if y2 >= img_height:
        y2 = img_height -1 

    width = int(x2 - x) 
    height = int(y2 - y) 
    return x, y, width, height

def get_landmark(landmark, x_box, y_box):
    landmark[:, 0] = landmark[:, 0] - x_box
    landmark[:, 1] = landmark[:, 1] - y_box

    landmark_flatten = landmark.flatten()
    landmark_list = list(landmark_flatten)
    landmark_string = ','.join(map(str,landmark_list))

    return landmark_string