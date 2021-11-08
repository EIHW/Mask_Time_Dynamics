# See data distribution, the result should correspond to Table I of manuscript
import pickle as p
from os.path import join

source_dir = './'

train = join(source_dir, 'train_more.pkl')
train_f = join(source_dir, 'train_more_female.pkl')
train_m = join(source_dir, 'train_more_male.pkl')

valid = join(source_dir, 'valid_less.pkl')
valid_f = join(source_dir, 'valid_less_female.pkl')
valid_m = join(source_dir, 'valid_less_male.pkl')

test = join(source_dir, 'test.pkl')
test_f = join(source_dir, 'test_female.pkl')
test_m = join(source_dir, 'test_male.pkl')

gender = join(source_dir, 'gender_dict.pkl')
patients = join(source_dir, 'patient_samples.pkl')

with open(train, 'rb') as tr_file:
    train_seeds = p.load(tr_file)

with open(train_f, 'rb') as tr_file:
    train_f_seeds = p.load(tr_file)

with open(train_m, 'rb') as tr_file:
    train_m_seeds = p.load(tr_file)

with open(valid, 'rb') as va_file:
    valid_seeds_orig = p.load(va_file)

valid_seeds = []
for seed in valid_seeds_orig:
    if seed not in train_seeds:
        valid_seeds.append(seed)

with open(test, 'rb') as te_file:
    test_seeds = p.load(te_file)

with open(test_f, 'rb') as te_file:
    test_f_seeds = p.load(te_file)

with open(test_m, 'rb') as te_file:
    test_m_seeds = p.load(te_file)

with open(gender, 'rb') as gender_file:
    gender_list = p.load(gender_file)

with open(patients, 'rb') as patients_file:
    patients_list = p.load(patients_file)


def convert_samples2patients(patients):
    patients_list = {}

    for sp in ['train', 'valid', 'test']:
        for pa in patients[sp].keys():
            for sa in patients[sp][pa]:
                patients_list[sa] = pa

    return patients_list


patients = convert_samples2patients(patients_list)


def stats(data_set):
    samples_info = []
    for i in data_set:
        sample_id = i[0].split('/')[-1]
        mask = i[1]
        patient = patients[sample_id]
        gender = gender_list[sample_id]

        sample_info = [sample_id, mask, patient, gender]
        samples_info.append(sample_info)

    return samples_info


for split, seeds in zip(['train', 'valid', 'test'], [train_seeds, valid_seeds, test_seeds]):
    mask_count = 0
    mask_count_f = 0
    mask_count_m = 0
    no_mask_count = 0
    no_mask_count_f = 0
    no_mask_count_m = 0
    mask_spks = {}
    no_mask_spks = {}

    for i in stats(seeds):
        if i[1] == 'mask':
            mask_count += 1
            if i[2] not in mask_spks.keys():
                mask_spks[i[2]] = i[3]
            if i[3] == 'f':
                mask_count_f += 1
            elif i[3] == 'm':
                mask_count_m += 1

        elif i[1] == 'clear':
            no_mask_count += 1
            if i[2] not in no_mask_spks.keys():
                no_mask_spks[i[2]] = i[3]
            if i[3] == 'f':
                no_mask_count_f += 1
            elif i[3] == 'm':
                no_mask_count_m += 1

    mask_f_spks = []
    mask_m_spks = []
    no_mask_f_spks = []
    no_mask_m_spks = []
    for spk, gender in mask_spks.items():
        if gender == 'm':
            mask_m_spks.append(spk)
        elif gender == 'f':
            mask_f_spks.append(spk)

    for spk, gender in no_mask_spks.items():
        if gender == 'm':
            no_mask_m_spks.append(spk)
        elif gender == 'f':
            no_mask_f_spks.append(spk)

    print('-------------------------  {}  ------------------------'.format(split))
    print(len(no_mask_f_spks), len(no_mask_m_spks), len(no_mask_spks))
    print(no_mask_count_f, no_mask_count_m, no_mask_count)
    print(mask_count_f, mask_count_m, mask_count)
    print(mask_count_f + no_mask_count_f, mask_count_m + no_mask_count_m, mask_count + no_mask_count)
