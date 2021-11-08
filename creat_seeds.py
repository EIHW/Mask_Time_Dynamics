import pickle as p
import os
import csv
from os.path import join


def exclude_patients(seeds_save_to, paths, patients):
    with open(join(seeds_save_to, 'patient_samples.pkl'), 'rb') as new_pkl:
        pat_samples = p.load(new_pkl)
    samples = []
    [samples.extend(pat_samples['valid'][pa]) for pa in patients]

    sel_paths = [seed for seed in paths if seed[0].split('/')[-1] not in samples]
    not_sel_paths = [seed for seed in paths if seed[0].split('/')[-1] in samples]
    return sel_paths, not_sel_paths


def sel_gender(paths, gender, gender_dict_):
    return [pa for pa in paths if gender_dict_[pa[0].split('/')[-1]] == gender]


def create_seeds(wav_dir, meta_info, save_to):
    filenames = os.listdir(wav_dir)
    data_dict = {}
    gender_dict = {}
    with open(meta_info) as csv_file:
        lines = csv.reader(csv_file)
        next(lines)

        for line in lines:
            file_id, label = line[0], line[1]
            data_dict[file_id] = label
            gender_dict[file_id] = line[3].split('_')[0]

    with open('seeds/gender_dict.pkl', 'wb') as gender_pkl:
        p.dump(gender_dict, gender_pkl)

    tr_paths, dev_paths, test_paths = [], [], []
    trava_paths = []
    for filename in filenames:
        if filename.endswith('.wav'):
            file_path = join(wav_dir, filename)
            label = data_dict[filename]
            if filename.startswith('train_'):
                tr_paths.append([file_path, label])
                trava_paths.append([file_path, label])
            elif filename.startswith('devel_'):
                dev_paths.append([file_path, label])
                trava_paths.append([file_path, label])
            elif filename.startswith('test_'):
                test_paths.append([file_path, label])

    with open(join(save_to, 'train.pkl'), 'wb') as train_pkl:
        p.dump(tr_paths, train_pkl)
    with open(join(save_to, 'valid.pkl'), 'wb') as valid_pkl:
        p.dump(dev_paths, valid_pkl)
    with open(join(save_to, 'test.pkl'), 'wb') as test_pkl:
        p.dump(test_paths, test_pkl)

    # the patients for validation in experiments: 'prob03', 'prob09', 'prob23'.
    # The reasons for this selection is seen in manuscript
    sel_paths, not_sel_paths = exclude_patients(save_to, trava_paths, ['prob03', 'prob09', 'prob23'])
    with open(join(save_to, 'train_more.pkl'), 'wb') as sel_pkl:
        p.dump(sel_paths, sel_pkl)
    with open(join(save_to, 'valid_less.pkl'), 'wb') as not_sel_pkl:
        p.dump(not_sel_paths, not_sel_pkl)

    # female = ['prob02', 'prob07', 'prob11', 'prob17', 'prob32', 'prob07', 'prob27', 'prob32', 'prob03', 'prob09',
    # 'prob13', 'prob19', 'prob28', 'prob05', 'prob10', 'prob16', 'prob22', 'prob30']
    # male = ['prob04', 'prob08', 'prob15', 'prob21', 'prob25', 'prob31', 'prob12', 'prob18', 'prob23', 'prob26',
    # 'prob33', 'prob06', 'prob14', 'prob20', 'prob24', 'prob29']

    sel_paths_male = sel_gender(sel_paths, 'm', gender_dict)
    sel_paths_female = sel_gender(sel_paths, 'f', gender_dict)

    not_sel_paths_male = sel_gender(not_sel_paths, 'm', gender_dict)
    not_sel_paths_female = sel_gender(not_sel_paths, 'f', gender_dict)

    test_paths_male = sel_gender(test_paths, 'm', gender_dict)
    test_paths_female = sel_gender(test_paths, 'f', gender_dict)

    with open(join(save_to, 'train_more_male.pkl'), 'wb') as sel_pkl:
        p.dump(sel_paths_male, sel_pkl)
    with open(join(save_to, 'train_more_female.pkl'), 'wb') as sel_pkl:
        p.dump(sel_paths_female, sel_pkl)
    with open(join(save_to, 'valid_less_male.pkl'), 'wb') as sel_pkl:
        p.dump(not_sel_paths_male, sel_pkl)
    with open(join(save_to, 'valid_less_female.pkl'), 'wb') as sel_pkl:
        p.dump(not_sel_paths_female, sel_pkl)
    with open(join(save_to, 'test_male.pkl'), 'wb') as sel_pkl:
        p.dump(test_paths_male, sel_pkl)
    with open(join(save_to, 'test_female.pkl'), 'wb') as sel_pkl:
        p.dump(test_paths_female, sel_pkl)


if __name__ == '__main__':
    data_root = '/nas/staff/data_work/Adria/ComParE_masks/Data'
    wav_folder = join(data_root, 'wav')  # path to wav folder
    meta = join(data_root, 'Mask_labels_confidential.csv')  # path to labels (csv file)
    pkls_folder = './seeds'
    create_seeds(wav_folder, meta, save_to=pkls_folder)
