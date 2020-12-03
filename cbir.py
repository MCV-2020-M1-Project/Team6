from libs import descriptors as desc
from libs import distance_metrics as dists


def get_top_k_multi(query, db_descriptor_list, descriptor_method_list, weights, measure_name, similarity, k,
                    hier_desc_dict=None, desc_check=False):
    if desc_check and desc.painting_in_db(query, db_descriptor_list):  # TODO: Before or after text???
        return [-1]

    shorter_list = []
    # Filter out by hierarchy
    if hier_desc_dict is not None:
        for desc_name, thresh in hier_desc_dict.items():
            # Gazapo: Will only work with text
            # print('HEY', query[desc_name])
            for d in db_descriptor_list:
                if dists.gestalt(query[desc_name], d[desc_name]) <= thresh:
                    shorter_list.append(d)

    if len(shorter_list) < 1:
        shorter_list = db_descriptor_list

    # get top k
    distances_dict = {}
    for db_point in shorter_list:
        # print('db:', db_point['author'])
        img_idx = db_point['idx']
        # print(img_idx)
        distances_dict[img_idx] = 0

        for d, w in zip(descriptor_method_list, weights):
            if measure_name == 'bfm' or measure_name == 'flann':
                distances = dists.get_all_measures(query[d], db_point[d], mode='kp')
                # print('measure=',distances[measure],'db_point=',db_point[d])
                distances_dict[img_idx] += (distances[measure_name])
            else:
                distances = dists.get_all_measures(query[d], db_point[d])
                distances_dict[img_idx] += w * abs(distances[measure_name])

    result = sorted(distances_dict, key=distances_dict.get, reverse=similarity)[:k]

    if len(result) < k:
        result += [0] * (k - len(result))

    return result
