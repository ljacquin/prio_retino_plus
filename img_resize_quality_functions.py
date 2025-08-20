import cv2
import tempfile
import imquality.brisque as brisque
from tensorflow.keras.utils import load_img


def get_img_brisque_score(filename, img_size_qual_):
    img = load_img(filename, target_size=(int(img_size_qual_), int(img_size_qual_)))
    return brisque.score(img)


def resize_image(filename, desired_size_=1024):
    img = cv2.imread(f"{filename}")
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    image, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contours)

    if w > 200 and h > 200:
        new_img = img[y:y + h, x:x + w]
        height, width, _ = new_img.shape

        if max([height, width]) > desired_size_:
            ratio = float(desired_size_ / max([height, width]))
            new_img = cv2.resize(new_img,
                                 tuple([int(width * ratio), int(height * ratio)]),
                                 interpolation=cv2.INTER_CUBIC)

        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cv2.imwrite(tmpfile.name, new_img)
        return tmpfile.name
    else:
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cv2.imwrite(tmpfile.name, img)
        return tmpfile.name


def compute_resize_quality_img(filename, img_size_qual=150, desired_size=1024):
    temp_resize_img = resize_image(filename, desired_size_=desired_size)
    img_qual_score = get_img_brisque_score(filename, img_size_qual_=img_size_qual)
    return {'temp_resize_img': temp_resize_img,
            'img_qual_score': img_qual_score}

