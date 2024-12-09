from util.path_abstract import PathAbstract


class Path(PathAbstract):
    @staticmethod
    def db_root_dir():
        return '/home/chenjian/dataset/seqsdatasets/毕业设计试验程序及结果/dataset/Youtube-VOS/Youtube-VOS'

    @staticmethod
    def save_root_dir():
        return './models'

    @staticmethod
    def models_dir():
        return "./models"

