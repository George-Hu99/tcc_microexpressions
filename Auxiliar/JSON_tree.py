import os
import json
import abc

class JSON_tree (object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def path_to_dict(path):
        d = {'name': os.path.basename(path)}
        if os.path.isdir(path):
            d['type'] = "directory"
            d['children'] = [JSON_tree.path_to_dict(os.path.join(path,x)) for x in os.listdir(path)]

        else:
            d['type'] = "file"
        return d

    @abc.abstractmethod
    def json_to_file(json_tree, name):
        with open(name, 'w') as out:
            json.dump(json_tree, out)