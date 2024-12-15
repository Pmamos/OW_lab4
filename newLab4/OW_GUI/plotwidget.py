from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # Import 3D wykresów
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib import collections as mpl_collections

class plotwidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.canvas = FigureCanvas(Figure())
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)

        # Tworzenie osi 3D
        self.canvas.axes = self.canvas.figure.add_subplot(111, projection='3d')
        self.setLayout(vertical_layout)



    def get_data(self):
        """Collect raw data (lines, scatter, etc.) from the plot."""
        data = []

        # Obsługuje dane z scatter plot
        if hasattr(self.canvas.axes, 'collections'):
            for collection in self.canvas.axes.collections:
                if isinstance(collection, mpl_collections.PathCollection):
                    paths = collection.get_paths()
                    for path in paths:
                        x, y = path.vertices[:, 0], path.vertices[:, 1]
                        # W przypadku wykresu 3D dodajemy z (wartości trzeciego wymiaru)
                        z = collection.get_offsets()[:, 1]  # Jeśli z jest przechowywane w drugim wymiarze
                        data.append(("scatter", x, y, z))

        # Obsługuje dane z linii, jeśli takie istnieją
        for line in self.canvas.axes.lines:
            x, y, z = line.get_xdata(), line.get_ydata(), line._verts3d[2] if hasattr(line, '_verts3d') else []
            data.append(("line", x, y, z, line.get_label()))

        return data

    def set_data(self, data):
        """Recreate the plot from raw data."""
        self.canvas.axes.clear()  # Czyścimy obecny wykres

        for item in data:
            if item[0] == "line":
                _, x, y, z, label = item
                # Jeśli z jest dostępne, tworzymy wykres 3D
                if len(z) > 0:
                    self.canvas.axes.plot(x, y, z, label=label)
                else:
                    self.canvas.axes.plot(x, y, label=label)
            elif item[0] == "scatter":
                _, x, y, z = item
                # Sprawdzamy, czy x, y, z mają takie same długości
                if len(x) == len(y):
                    # Przycinamy lub dopasowujemy długość z do x i y
                    if len(z) != len(x):
                        z = z[:len(
                            x)]  # Przycinamy lub uzupełniamy (np. poprzez np. z = np.zeros_like(x) dla uzupełnienia)
                        print("Dopasowano długość z do długości x i y.")
                    # Teraz x, y, z mają te same wymiary
                    self.canvas.axes.scatter(x, y, z, label="Scatter Plot", edgecolor='k', s=100)
                else:
                    print("Wymiary x i y nie pasują do siebie!")

        self.canvas.draw()  # Rysujemy ponownie wykres




