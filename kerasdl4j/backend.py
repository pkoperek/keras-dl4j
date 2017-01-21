import h5py
import tempfile
import new
import numpy
import xxhash

from keras import backend as K
from os import path, mkdir
from py4j.java_gateway import JavaGateway

batch_file_template = "batch_{id}.h5"
hijack_cache = {}


def generate_tmp_path():
    tmp_file = tempfile.NamedTemporaryFile(prefix="dl4j")
    tmp_file.close()

    return tmp_file.name


def dump_h5(dataset, batch_size, directory_name):
    """
    Dumps the data from dataset to a series of HDF5 files. Each of them will contain at most batch_size samples.

    :param dataset: Dataset to store
    :param batch_size: Size of the batch
    :param directory_name: Directory where the batch files are going to be saved
    """

    if path.exists(directory_name):
        raise IOError("Path exists: " + directory_name)
        return

    mkdir(directory_name)

    batch_id = 0
    samples_count = dataset.shape[0]

    begin = 0
    end = batch_size

    while begin < samples_count:
        batch_file_name = batch_file_template.format(id=batch_id)
        f = h5py.File(path.join(directory_name, batch_file_name), 'w')
        f.create_dataset("data", data=dataset[begin:end])
        f.flush()
        f.close()

        begin = end
        end += batch_size
        batch_id += 1


def fit_with_dl4j(
        model,
        nb_epoch,
        features_directory,
        labels_directory
):
    """
    Executes fitting of the model by using DL4J as backend
    :param model: Model to use
    :param nb_epoch: Number of learning epochs
    :param features_directory: Directory with feature batch files
    :param labels_directory: Directory with label batch files
    :return:
    """

    model_file_path = generate_tmp_path()
    model.save(model_file_path)

    gateway = JavaGateway()

    sequential = gateway.jvm.org.deeplearning4j.keras.KerasModelType.SEQUENTIAL
    params_builder = gateway.jvm.org.deeplearning4j.keras.EntryPointFitParameters.builder()
    params_builder.type(sequential)
    params_builder.modelFilePath(model_file_path)
    params_builder.nbEpoch(nb_epoch)
    params_builder.trainFeaturesDirectory(features_directory)
    params_builder.trainLabelsDirectory(labels_directory)
    params_builder.dimOrdering(K.image_dim_ordering())
    gateway.fit(params_builder.build())


def hash_ndarray(array):
    """
    Calculates a hash of contents of ndarray
    :param array: Array to calculate hash
    :return: hex digest of the hash (as string)
    """

    hsh = xxhash.xxh64()
    hsh.update(array.view(numpy.uint8))
    return hsh.hexdigest()


def _new_fit(
        self,
        x,
        y,
        batch_size=32,
        nb_epoch=10,
        verbose=1,
        callbacks=[],
        validation_split=0.,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        **kwargs):
    def dump_ndarray(batch_size, dataset):
        dataset_hash = hash_ndarray(dataset)
        if not dataset_hash in hijack_cache:
            directory_name = generate_tmp_path()
            dump_h5(dataset, batch_size, directory_name)
            hijack_cache[dataset_hash] = directory_name
        else:
            print("Dataset already dumped")

        return hijack_cache[dataset_hash]

    x_directory = dump_ndarray(batch_size, x)
    y_directory = dump_ndarray(batch_size, y)

    fit_with_dl4j(
        self,
        nb_epoch,
        x_directory,
        y_directory
    )


def install_dl4j_backend(model):
    """
    Hijacks the `fit` method call in the model object
    :param model: Model in which fit will be hijacked
    """

    model.__dl4j_old_fit = model.fit
    model.fit = new.instancemethod(_new_fit, model, None)
