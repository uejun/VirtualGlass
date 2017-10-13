import os
import glob

from PIL import Image
import click
import cv2
import dlib

VIRTUAL_FIT_IMAGE_SIZE = (713, 401)

@click.group(chain=True)
def cmd():
    pass


@cmd.command()
@click.option('--dir', '-d', default='World')
@click.option('--out', '-o', help='output_dir_path')
def resize_no_glass_images(dir: str, out: str):
    for imgpath in glob.glob(dir + "/**/*.png"):
        basename = imgpath.split('/')[-1]
        people = imgpath.split('/')[-2]
        output_name = people + "_" + basename
        output_imgpath = out + "/" + output_name

        img = Image.open(imgpath)
        resized_img = img.resize(VIRTUAL_FIT_IMAGE_SIZE)
        resized_img.save(output_imgpath, 'PNG')

@cmd.command()
@click.option('--dir', '-d', default='World')
def create_shimlink(dir):
    for imgpath in glob.glob(dir + "/**/*.png"):
        print(imgpath)
    pass

@cmd.command()
@click.option('--dir', '-d', default='World')
def rename(dir: str):
    for imgpath in glob.glob(dir + "/**/*.png"):
        diname = os.path.dirname(imgpath)
        basename = os.path.basename(imgpath)
        basename_noext, ext = os.path.splitext(basename)
        parts = basename_noext.split('_')

        angle_no = int(parts[-1])
        if angle_no == 0:
            modified_angle_no = 12
        else:
            modified_angle_no = angle_no - 1
        parts[-1] = str(modified_angle_no)
        modified_name = "_".join(parts) + ext

        new_path = diname + "/" + modified_name
        print(new_path)
        os.rename(imgpath, new_path)

@cmd.command()
@click.option('--dir', '-d', default='World')
def rename2(dir: str):
    for imgpath in glob.glob(dir + "/**/*.png"):
        diname = os.path.dirname(imgpath)
        basename = os.path.basename(imgpath)
        basename_noext, ext = os.path.splitext(basename)
        parts = basename_noext.split('_')

        modified_name = parts[2] + '_' + parts[0] + "_" + parts[1] + ext

        new_path = diname + "/" + modified_name
        print(new_path)
        os.rename(imgpath, new_path)


def extract_feature(detector, predictor, imgpath):
    img = cv2.imread(imgpath)
    return extract_feature_with_img(detector, predictor, img)


def extract_feature_with_img(detector, predictor, img):
    dets = detector(img, 1)
    landmarks = predictor(img, dets[0]).parts()

    features = []
    for i, landmark in enumerate(landmarks):
        if i in [6, 8, 10, 30, 31, 35, 48, 51, 54, 57]:
            features.append(landmark.x)
            features.append(landmark.y)
    return features

@cmd.command()
@click.option('--dir', '-d', default='World')
def cv(dir: str):
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=1)
    y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    predictor_path = "./resources/dlib/shape_predictor_68_face_landmarks.dat"
    if not predictor_path:
        predictor_path = "./shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)


    noglass_images_dir = "/home/dl-box/デスクトップ/ResizedVirtualFitNoGlassImages"
    a = {}
    for imgpath in glob.glob(noglass_images_dir + "/Ueda" + "_*.png"):
        basename = os.path.basename(imgpath)
        basename_noext, ext = os.path.splitext(basename)
        people = basename_noext.split('_')[0]
        angle_no = basename_noext.split('_')[-1]

        features = extract_feature(detector, predictor, imgpath)
        a[angle_no] = features

    Xtrue = [a["00"], a["01"], a["02"], a["03"], a["04"], a["05"], a["06"], a["07"], a["08"], a["09"], a["10"], a["11"], a["12"]]
    print(len(Xtrue))
    neigh.fit(Xtrue, y)

    count = 0
    for imgpath in glob.glob(dir + "/*.png"):
        img = cv2.imread(imgpath)
        feature = extract_feature_with_img(detector, predictor, img)
        print(len(feature))
        print(neigh.predict([feature]))

        output_path = "/home/dl-box/デスクトップ/output"
        if count > 200:
            break
        else:
            count += 1

def main():
    cmd()


if __name__ == '__main__':
    main()