import os, cv2, glob
import scipy.io as io

root = '/home/xx/MacCrowdCode/crowdcount/crowdcount-mcnn/data/original/shanghaitech'


def mat2txt():
    part_A_train = os.path.join(root, 'part_A_final/train_data', 'ground-truth')
    part_A_test = os.path.join(root, 'part_A_final/test_data', 'ground-truth')
    part_B_train = os.path.join(root, 'part_B_final/train_data', 'ground-truth')
    part_B_test = os.path.join(root, 'part_B_final/test_data', 'ground-truth')

    sets = [part_A_train, part_A_test, part_B_train, part_B_test]

    for index, path in enumerate(sets):
        gt_list = os.listdir(path)
        for gt_name in gt_list:
            if 'mat' in gt_name:
                name = os.path.join(path, gt_name)
                print(name)
                mat = io.loadmat(name)
                txt_path = name.replace('.mat', '.txt').replace('ground-truth', 'gt_txt_p2p').replace('GT_IMG_', 'IMG_')
                print(txt_path)
                im_path = name.replace('.mat', '.jpg').replace('ground-truth', 'images').replace('GT_IMG_', 'IMG_')
                im = cv2.imread(im_path)
                height, width = im.shape[:2]
                gt = mat["image_info"][0, 0][0, 0][0]
                for i in range(0, len(gt)):
                    if 0 < int(gt[i][1]) < height and 0 < int(gt[i][0]) < width:
                        with open(txt_path, 'a') as f:
                            f.write("{} {}\n".format(int(gt[i][0]), int(gt[i][1])))


def txt2list():
    part_A_train = os.path.join(root, 'part_A_final/train_data', 'gt_txt_p2p')
    part_A_test = os.path.join(root, 'part_A_final/test_data', 'gt_txt_p2p')
    sets = [part_A_train, part_A_test]
    name = ['part_A_train.txt', 'part_A_test.txt']

    for index, path in enumerate(sets):
        txt_list = glob.glob(os.path.join(path, '*.txt'))
        for txt_name in txt_list:
            img_name = txt_name.replace('.txt', '.jpg').replace('gt_txt_p2p', 'images')
            with open(name[index], 'a') as f:
                f.write("{} {}\n".format(img_name, txt_name))


if __name__ == '__main__':
    # mat2txt()
    txt2list()
