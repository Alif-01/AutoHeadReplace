from PIL import Image, ImageDraw, ImageFilter
import face_recognition
import numpy as np


def get_center(points):
    cx, cy = 0, 0

    for p in points:
        cx += p[0]
        cy += p[1]

    cx /= len(points)
    cy /= len(points)

    return cx, cy


def get_feature_points(image):
    landmarks = face_recognition.face_landmarks(np.array(image.convert('RGB')))

    res = []

    for landmark in landmarks:
        pts = []
        for label in landmark:
            label_size = len(landmark[label])
            weight = 1 / label_size
            if label == "chin":
                weight *= 10
            for pos in landmark[label]:
                pts.append((pos[0], pos[1], weight))
        res.append(np.asarray(pts))

    return res


def solve_param(x, y, z, t, k):
    a = np.zeros((3, 3))
    b = np.zeros((3, 1))

    a[0][0] = np.sum(x * x * k)
    a[0][1] = np.sum(y * x * k)
    a[0][2] = np.sum(z * x * k)
    b[0] = np.sum(t * x * k)

    a[1][0] = np.sum(x * y * k)
    a[1][1] = np.sum(y * y * k)
    a[1][2] = np.sum(z * y * k)
    b[1] = np.sum(t * y * k)

    a[2][0] = np.sum(x * z * k)
    a[2][1] = np.sum(y * z * k)
    a[2][2] = np.sum(z * z * k)
    b[2] = np.sum(t * z * k)

    x = np.matmul(np.linalg.inv(a), b)
    return x[0][0], x[1][0], x[2][0]


def get_affine_transform(feature1, feature2):
    a, b, c = solve_param(feature1[:, 0], feature1[:, 1], np.ones(feature1.shape[0]), feature2[:, 0], feature1[:, 2])
    d, e, f = solve_param(feature1[:, 0], feature1[:, 1], np.ones(feature1.shape[0]), feature2[:, 1], feature1[:, 2])

    return a, b, c, d, e, f


def get_feature_dis(feature1, feature2, trans):
    t_00 = trans[0] * feature1[:, 0] * feature1[:, 2]
    t_01 = trans[1] * feature1[:, 1] * feature1[:, 2]
    t_02 = trans[2] * feature1[:, 2]

    t_10 = trans[3] * feature1[:, 0] * feature1[:, 2]
    t_11 = trans[4] * feature1[:, 1] * feature1[:, 2]
    t_12 = trans[5] * feature1[:, 2]

    diff_0 = t_00 + t_01 + t_02 - feature2[:, 0] * feature2[:, 2]
    diff_1 = t_10 + t_11 + t_12 - feature2[:, 1] * feature2[:, 2]

    dis = np.sum(diff_0 * diff_0 + diff_1 * diff_1)

    return dis


def replace_single_face(img1, img2, img3, feature1, feature2, feature3):
    trans2 = get_affine_transform(feature1, feature2)
    trans3 = get_affine_transform(feature1, feature3)

    dis2 = get_feature_dis(feature1, feature2, trans2)
    dis3 = get_feature_dis(feature1, feature3, trans3)

    if dis2 < dis3:
        t_img2 = img2.transform(img1.size, Image.AFFINE, trans2, Image.BICUBIC)
        img1.paste(t_img2, (0, 0), t_img2)
    else:
        t_img3 = img3.transform(img1.size, Image.AFFINE, trans3, Image.BICUBIC)
        img1.paste(t_img3, (0, 0), t_img3)

    return img1


def blend_image(image1, image2, feather_radius):
    im1 = image1.convert('RGB')
    im2 = image2.convert('RGB')
    im3 = Image.new('RGB', im1.size)
    im3.paste(im1, (0, 0), image2)

    im3 = im3.filter(ImageFilter.GaussianBlur(5))
    im2 = im2.filter(ImageFilter.GaussianBlur(5))

    alpha = image2.split()[3]
    alpha = alpha.filter(ImageFilter.GaussianBlur(feather_radius))
    ar_alpha = np.clip((np.asarray(alpha)/255-0.5)*2, 0, 1)

    ar0 = np.asarray(image1).astype('float32')
    ar1 = np.asarray(image2).astype('float32')
    ar2 = np.asarray(im2).astype('float32')
    ar2 += 0.01
    ar3 = np.asarray(im3).astype('float32')
    ar3 += 0.01

    res = np.zeros((image2.size[1], image2.size[0], 3))

    for i in range(im2.size[1]):
        for j in range(im2.size[0]):
            if ar1[i][j][3] > 0:
                res[i, j, 0:3] = (ar1[i, j, 0:3] / ar2[i, j] * ar3[i, j]) * ar_alpha[i, j] \
                                + ar0[i, j, 0:3] * (1 - ar_alpha[i, j])
                # res[i][j][3] = ar_alpha[i][j]

    res = np.clip(res, 0, 255).astype('uint8')

    new_image = Image.fromarray(res, 'RGB')

    image1.paste(new_image, (0, 0), image2)

    return image1


def replace_faces(image, new_face, feather_radius=5):
    image = image.convert('RGBA')
    new_face = new_face.convert('RGBA')

    new_face_rev = new_face.transpose(Image.FLIP_LEFT_RIGHT)

    image_feature = get_feature_points(image)
    new_face_feature = get_feature_points(new_face)
    new_face_rev_feature = get_feature_points(new_face_rev)

    if len(new_face_feature) < 1 or len(new_face_rev_feature) < 1:
        print("no face in the new_face image")
        return image

    if len(new_face_feature) > 1 or len(new_face_rev_feature) > 1:
        print("too many faces in the new_face image")
        return image

    print("%d faces detected" % len(image_feature))

    cover_image = Image.new('RGBA', image.size)

    cnt, cur = len(image_feature), 0
    for face in image_feature:
        cur += 1
        print("replacing %d/%d..." % (cur, cnt))
        cover_image = replace_single_face(cover_image, new_face, new_face_rev,
                                          face, new_face_feature[0], new_face_rev_feature[0])

    print("blending...")
    image = blend_image(image, cover_image, feather_radius)

    print("done.")

    return image


# ikisugi! 191919
if __name__ == '__main__':
    test_img1 = Image.open("yajue.jpg")
    test_img2 = Image.open("yajue1.png")

    test_img3 = replace_faces(test_img1, test_img2)

    test_img3.save("senpai.png")
    # test_img3.show()
