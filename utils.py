import tensorflow as tf
import numpy as np
from skimage.measure import compare_ssim


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def extract_paths_from_deep_dict(d, prefix=[]):
    if type(d) is not dict:
        return [prefix]
    else:
        return sum([extract_paths_from_deep_dict(d[key], prefix + [key]) for key in d], [])


def get_from_deep_dict(d, keys):
    for k in keys:
        d = d[k]
    return d


def ssim(tar, gen, masks=None):
    tar = np.array(tar)
    gen = np.array(gen)
    if masks is not None:
        masks = np.array(masks)
    data_range = max(gen.max(), tar.max()) - min(gen.min(), tar.min())
    ssims = []
    fg_ssims = []
    bg_ssims = []
    for t, g, m in zip(tar, gen, masks):
        fgc = np.sum(m)
        bgc = m.shape[0] * m.shape[1] - fgc
        s, si = compare_ssim(t, g, multichannel=True, data_range=data_range, full=True)
        ssims.append(s)
        fg_ssims.append(np.sum(si * m) / fgc / si.shape[-1])
        bg_ssims.append(np.sum(si * (1 - m)) / bgc / si.shape[-1])
    if masks is not None:
        return np.mean(ssims), np.mean(fg_ssims), np.mean(bg_ssims)
    else:
        return np.mean(ssims)


def make_pretrained_weight_loader(pretrained_path, loaded_scope, checkpoint_scope, excluded_parts, replace_names):
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=loaded_scope)
    var_dict = {v.op.name[v.op.name.index(checkpoint_scope):]: v for v in var_list}
    var_dict = {k: v for k, v in var_dict.items() if not any(excl in k for excl in excluded_parts)}
    for fr, to in replace_names:
        var_dict = {k.replace(fr, to): v for k, v in var_dict.items()}
    saver = tf.train.Saver(var_list=var_dict)

    def init_fn(sess):
        saver.restore(sess, pretrained_path)

    return init_fn


def extend_spatial_sizes(t):
    return tf.pad(t, [[0, 0]] + [[0, 1]] * (len(t.shape) - 2) + [[0, 0]])


def reduce_spatial_sizes(t):
    for i in range(1, len(t.shape) - 1):
        t = tf.gather(t, list(range(1, int(t.shape[i]))), axis=i)
    return t
