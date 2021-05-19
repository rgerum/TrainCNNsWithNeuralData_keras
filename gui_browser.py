import sys

# Setting the Qt bindings for QtPy
import os
import pandas as pd
import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
from qtpy import API_NAME as QT_API_NAME
if QT_API_NAME.startswith("PyQt4"):
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as Canvas
    from matplotlib.backends.backend_qt4agg import FigureManager
    from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as NavigationToolbar
else:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
    from matplotlib.backends.backend_qt5agg import FigureManager
    from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import _pylab_helpers
from pathlib import Path
import numpy as np
import imageio
import matplotlib.pyplot as plt
import yaml

from net_helpers import read_data, getMeta

""" some magic to prevent PyQt5 from swallowing exceptions """
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook
# Set the exception hook to our wrapping function
sys.excepthook = lambda *args: sys._excepthook(*args)


class MatplotlibWidget(Canvas):

    def __init__(self, parent=None, width=4, height=3, dpi=100):
        plt.ioff()
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.figure.patch.set_facecolor([0, 1, 0, 0])
        self.axes = self.figure.add_subplot(111)

        Canvas.__init__(self, self.figure)
        self.setParent(parent)

        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

        self.manager = FigureManager(self, 1)
        self.manager._cidgcf = self.figure

        """
        _pylab_helpers.Gcf.figs[num] = canvas.manager
        # get the canvas of the figure
        manager = _pylab_helpers.Gcf.figs[num]
        # set the size if it is defined
        if figsize is not None:
            _pylab_helpers.Gcf.figs[num].window.setGeometry(100, 100, figsize[0] * 80, figsize[1] * 80)
        # set the figure as the active figure
        _pylab_helpers.Gcf.set_active(manager)
        """
        _pylab_helpers.Gcf.set_active(self.manager)

def pathParts(path):
    if path.parent == path:
        return [path]
    return pathParts(path.parent) + [path]


class MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowIcon(qta.icon("mdi.folder-pound-outline"))

        # QSettings
        self.settings = QtCore.QSettings("TrainingResults", "TrainingResults")

        self.setMinimumWidth(1200)
        self.setMinimumHeight(400)
        self.setWindowTitle("Training Results Viewer")

        hlayout = QtWidgets.QHBoxLayout(self)

        self.browser = Browser()
        # hlayout.addWidget(self.browser)

        self.plot = MeasurementPlot()
        # hlayout.addWidget(self.plot)

        self.text = MetaDataEditor()
        # hlayout.addWidget(self.text)

        self.splitter_filebrowser = QtWidgets.QSplitter()
        self.splitter_filebrowser.addWidget(self.browser)
        self.splitter_filebrowser.addWidget(self.plot)
        self.splitter_filebrowser.addWidget(self.text)
        hlayout.addWidget(self.splitter_filebrowser)

        self.browser.signal_selection_changed.connect(self.selected)
        self.browser.signal_multi_selection_changed.connect(self.plot.selected)

    def selected(self, name):
        self.text.selected(name)
        #self.plot.selected(name)


class MeasurementPlot(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.hlayout = QtWidgets.QVBoxLayout(self)
        self.hlayout.setContentsMargins(0, 0, 0, 0)
        self.canvas = MatplotlibWidget(self)
        plt.clf()
        self.hlayout.addWidget(self.canvas)
        self.tools = NavigationToolbar(self.canvas, self)
        self.hlayout.addWidget(self.tools)

    def selected(self, names):
        plt.clf()
        datas = []
        for name in names:
            if name.endswith(".csv"):
                data = read_data(name, do_exclude=False)
            else:
                try:
                    data = read_data(name + "/data.csv", do_exclude=False)
                except FileNotFoundError:
                    continue
            datas.append(data)
        if len(datas) == 0:
            self.canvas.draw()
            return
        multiple = len(datas) != 1
        data = pd.concat(datas)
        if multiple and "exclude" in data:
            data = data[data.exclude != True]
        for filename, d in data.groupby("filename"):
            p, = plt.plot(d.epoch, d.val_accuracy, "-", alpha=0.5)
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.autoscale()
        plt.tight_layout()
        #plt.plot(data.rp, data.vel)
        self.canvas.draw()


class MetaDataEditor(QtWidgets.QWidget):
    yaml_file = None

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        hlayout = QtWidgets.QVBoxLayout(self)
        hlayout.setContentsMargins(0, 0, 0, 0)

        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setToolTip("Meta data from parent folders")
        hlayout.addWidget(self.text)

        self.text2 = QtWidgets.QPlainTextEdit()
        self.text2.textChanged.connect(self.save)
        self.text2.setToolTip("Meta data from current folder/file. Can be editied and will be automatically saved")
        hlayout.addWidget(self.text2)

        self.name = QtWidgets.QLineEdit()
        self.name.setReadOnly(True)
        self.name.setToolTip("The current folder/file.")
        hlayout.addWidget(self.name)

    def save(self):
        if self.yaml_file is not None:
            with open(self.yaml_file, "w") as fp:
                fp.write(self.text2.toPlainText())

    def selected(self, name):
        meta = getMeta(name)
        self.name.setText(name)

        self.text.setPlainText(yaml.dump(meta))

        self.yaml_file = None
        if name.endswith(".tif"):
            yaml_file = Path(name.replace(".tif", "_meta.yaml"))
        else:
            yaml_file = Path(name) / "meta.yaml"

        if yaml_file.exists():
            with yaml_file.open() as fp:
                self.text2.setPlainText(fp.read())
        else:
            self.text2.setPlainText("")
        self.yaml_file = yaml_file

class Browser(QtWidgets.QTreeView):
    signal_selection_changed = QtCore.Signal(str)
    signal_multi_selection_changed = QtCore.Signal(list)

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.settings = QtCore.QSettings("rgerum", "net browser")

        # self.setCentralWidget(self.frame)
        #hlayout = QtWidgets.QVBoxLayout(self)

        """ browser"""
        self.dirmodel = QtWidgets.QFileSystemModel()
        # Don't show files, just folders
        # self.dirmodel.setFilter(QtCore.QDir.NoDotAndDotDot | QtCore.QDir.AllDirs)
        self.dirmodel.setNameFilters(["*.csv"])
        self.dirmodel.setNameFilterDisables(False)
        self.folder_view = self#QtWidgets.QTreeView(parent=self)
        self.folder_view.setModel(self.dirmodel)
        self.folder_view.activated[QtCore.QModelIndex].connect(self.clicked)
        self.folder_view.selectionChanged = self.clicked
        self.folder_view.setSelectionMode(self.ExtendedSelection)
        # self.folder_view.selected[QtCore.QModelIndex].connect(self.clicked)

        # Don't show columns for size, file type, and last modified
        self.folder_view.setHeaderHidden(True)
        self.folder_view.hideColumn(1)
        self.folder_view.hideColumn(2)
        self.folder_view.hideColumn(3)

        self.selectionModel = self.folder_view.selectionModel()

        #hlayout.addWidget(self.folder_view)

        if self.settings.value("browser/path"):
            self.set_path(self.settings.value("browser/path"))
        self.set_path(
            r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_21_alginate2%_NIH_xposition_1")

    def set_path(self, path):
        path = Path(path)
        self.dirmodel.setRootPath(str(path.parent))
        for p in pathParts(path):
            self.folder_view.expand(self.dirmodel.index(str(p)))
        self.folder_view.setCurrentIndex(self.dirmodel.index(str(path)))
        print("scroll to ", str(path), self.dirmodel.index(str(path)))
        self.folder_view.scrollTo(self.dirmodel.index(str(path)))

    def clicked(self, index):
        paths = [self.dirmodel.filePath(index) for index in self.selectedIndexes()]
        print(paths)
        # get selected path of folder_view
        index = self.selectionModel.currentIndex()
        dir_path = self.dirmodel.filePath(index)
        print(dir_path)
        self.settings.setValue("browser/path", dir_path)
        self.signal_selection_changed.emit(dir_path)
        self.signal_multi_selection_changed.emit(paths)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        # accept url lists (files by drag and drop)
        for url in event.mimeData().urls():
            if str(url.toString()).strip().endswith(".npz"):
                event.accept()
                return
        event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QtCore.QEvent):
        for url in event.mimeData().urls():
            print(url)
            url = str(url.toString()).strip()
            if url.startswith("file:///"):
                url = url[len("file:///"):]
            if url.startswith("file:"):
                url = url[len("file:"):]
            self.loadFile(url)

    def openLoadDialog(self):
        # opening last directory von sttings
        self._open_dir = self.settings.value("_open_dir")
        if self._open_dir is None:
            self._open_dir = os.getcwd()

        dialog = QtWidgets.QFileDialog()
        dialog.setDirectory(self._open_dir)
        filename = dialog.getOpenFileName(self, "Open Positions", "", "Position Files (*.tif)")
        if isinstance(filename, tuple):
            filename = str(filename[0])
        else:
            filename = str(filename)
        if os.path.exists(filename):
            # noting directory to q settings
            self._open_dir = os.path.split(filename)[0]
            self.settings.setValue("_open_dir", self._open_dir)
            self.settings.sync()
            self.loadFile(filename)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
