import cv2

def draw_boxes(img, boxes, labels, probs, class_names):
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 1)
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.putText(img, label,
                    (int(box[0]) + 10, int(box[1]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    (127, 0, 127),
                    2) # line type

    return img