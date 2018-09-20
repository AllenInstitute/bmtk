def listify(files):
    # TODO: change this to include any iterable datastructures (sets, panda sequences, etc)
    if not isinstance(files, (list, tuple)):
        return [files]
    else:
        return files
