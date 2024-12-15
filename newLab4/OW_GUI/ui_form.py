from PySide6.QtCore import QCoreApplication, QMetaObject, Qt
from PySide6.QtWidgets import (
    QComboBox, QGridLayout, QGroupBox, QLabel, QMenuBar,
    QPushButton, QSizePolicy, QSpacerItem, QStatusBar,
    QTableWidget, QWidget, QDialog, QVBoxLayout
)
from plotwidget import plotwidget


class Ui_MainWindow:
    def __init__(self):
        self.fullscreen_graph = None
        self.centralwidget = None
        self.gridLayout = None
        self.menubar = None
        self.statusbar = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)

        self.centralwidget = QWidget(MainWindow)
        self.gridLayout = QGridLayout(self.centralwidget)

        self._create_settings_section()
        self._create_results_section()

        MainWindow.setCentralWidget(self.centralwidget)

        # Menubar and Statusbar
        self.menubar = QMenuBar(MainWindow)
        self.statusbar = QStatusBar(MainWindow)
        MainWindow.setMenuBar(self.menubar)
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    def _create_settings_section(self):
        """Create the settings section."""
        self.settings_group = QGroupBox("Ustawienia", self.centralwidget)
        self.gridLayout_2 = QGridLayout(self.settings_group)

        self._add_settings_controls()
        self._add_alternatives_table()
        self._add_algorithm_params_group()

        self.gridLayout.addWidget(self.settings_group, 0, 0, 1, 1)

    def _add_settings_controls(self):
        self.load_btn = QPushButton("Wczytaj dane z pliku", self.settings_group)
        self.load_btn.setMinimumSize(150, 40)
        self.gridLayout_2.addWidget(self.load_btn, 0, 0, 1, 1)

        self.gridLayout_2.addItem(QSpacerItem(100, 20, QSizePolicy.Fixed, QSizePolicy.Minimum), 0, 1)

        self.criterium_select = QComboBox(self.settings_group)
        self.criterium_select.setMinimumSize(300, 0)
        self.criterium_select.addItems(["FUZZY TOPSIS", "UTA DIS", "RSM"])
        self.gridLayout_2.addWidget(self.criterium_select, 0, 2, 1, 2)

        self.gridLayout_2.addItem(QSpacerItem(100, 20, QSizePolicy.Fixed, QSizePolicy.Minimum), 0, 4)

        self.start_btn = QPushButton("Stwórz ranking", self.settings_group)
        self.start_btn.setMinimumSize(150, 40)
        self.gridLayout_2.addWidget(self.start_btn, 0, 5, 1, 1)

    def _add_alternatives_table(self):
        self.label = QLabel("Alternatywy z kryteriami", self.settings_group)
        self.label.setAlignment(Qt.AlignCenter)
        self.gridLayout_2.addWidget(self.label, 1, 0, 1, 3)

        self.alternatives_table = QTableWidget(self.settings_group)
        self.alternatives_table.setMinimumSize(0, 200)
        self.gridLayout_2.addWidget(self.alternatives_table, 2, 0, 1, 3)

        self.label_2 = QLabel("Klasy", self.settings_group)
        self.label_2.setAlignment(Qt.AlignCenter)
        self.gridLayout_2.addWidget(self.label_2, 1, 3, 1, 3)

        self.class_table = QTableWidget(self.settings_group)
        self.class_table.setMinimumSize(0, 200)
        self.gridLayout_2.addWidget(self.class_table, 2, 3, 1, 3)

    def _add_algorithm_params_group(self):
        self.groupBox = QGroupBox("Parametry specyficzne dla algorytmów:", self.settings_group)
        self.gridLayout_5 = QGridLayout(self.groupBox)

        self.label_8 = QLabel("Wariant:", self.groupBox)
        self.gridLayout_5.addWidget(self.label_8, 0, 0)

        self.variant_select = QComboBox(self.groupBox)
        self.variant_select.addItems(["Ciągły", "Dyskretny"])
        self.gridLayout_5.addWidget(self.variant_select, 0, 1)

        self.label_10 = QLabel("Miara:", self.groupBox)
        self.gridLayout_5.addWidget(self.label_10, 0, 3)

        self.metric_select = QComboBox(self.groupBox)
        self.metric_select.addItems(["Euklidesowa", "Czebyszewa"])
        self.gridLayout_5.addWidget(self.metric_select, 0, 4)

        self.label_4 = QLabel("Typ:", self.groupBox)
        self.gridLayout_5.addWidget(self.label_4, 2, 0)

        self.opti_type = QComboBox(self.groupBox)
        self.opti_type.addItems(["Minimalizacja", "Maksymalizacja"])
        self.gridLayout_5.addWidget(self.opti_type, 2, 1)

        self.gridLayout_2.addWidget(self.groupBox, 3, 0, 2, 6)

    def _create_results_section(self):
        """Create the results section."""
        self.results_group = QGroupBox("Wyniki", self.centralwidget)
        self.gridLayout_4 = QGridLayout(self.results_group)

        self.label_7 = QLabel("Wykres", self.results_group)
        self.label_7.setAlignment(Qt.AlignCenter)
        self.gridLayout_4.addWidget(self.label_7, 0, 0)

        self.graph = plotwidget(self.results_group)
        self.graph.setMinimumSize(600, 0)
        self.gridLayout_4.addWidget(self.graph, 1, 0, 2, 1)

        # Dodanie przycisku pełnoekranowego
        self.fullscreen_btn = QPushButton("Pełny ekran", self.graph)
        self.fullscreen_btn.setMinimumSize(150, 40)
        self.fullscreen_btn.clicked.connect(self._open_fullscreen_graph)
        self.gridLayout_4.addWidget(self.fullscreen_btn, 2, 0, 1, 1)

        self.label_3 = QLabel("Stworzony ranking", self.results_group)
        self.label_3.setAlignment(Qt.AlignCenter)
        self.gridLayout_4.addWidget(self.label_3, 0, 1)

        self.ranking_table = QTableWidget(self.results_group)
        self.ranking_table.setMinimumSize(0, 200)
        self.gridLayout_4.addWidget(self.ranking_table, 1, 1)

        self.gridLayout.addWidget(self.results_group, 1, 0, 1, 1)

    def _open_fullscreen_graph(self):
        """Open the graph in fullscreen mode."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout

        # Tworzenie okna dialogowego
        self.fullscreen_dialog = QDialog(self.centralwidget)
        self.fullscreen_dialog.setWindowTitle("Wykres 3D - Pełny ekran")
        self.fullscreen_dialog.resize(1000, 800)

        # Layout do wykresu
        layout = QVBoxLayout(self.fullscreen_dialog)
        self.fullscreen_graph = plotwidget(self.fullscreen_dialog)
        layout.addWidget(self.fullscreen_graph)

        # Skopiowanie danych z głównego wykresu
        data = self.graph.get_data()  # Pobranie danych z głównego wykresu
        print(data)
        self.fullscreen_graph.set_data(data)  # Ustawienie tych danych na wykresie pełnoekranowym

        # Rysowanie wykresu pełnoekranowego
        self.fullscreen_graph.canvas.draw()

        # Wyświetlenie dialogu
        self.fullscreen_dialog.exec()

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", "MainWindow"))
        self.settings_group.setTitle(QCoreApplication.translate("MainWindow", "Ustawienia"))
        self.results_group.setTitle(QCoreApplication.translate("MainWindow", "Wyniki"))
