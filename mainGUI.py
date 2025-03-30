import sys
import time
import os

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, 
    QLabel, QLineEdit, QPushButton, QCheckBox, QFileDialog, 
    QTextEdit, QMessageBox, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem

# --------------------------------------------------------------------------------
# Import your existing classes
# (Adjust the file/module names to match your project)
from ranking import DataAnalysisRanker
from feature_selection import FeatureSelector
from model_testing import ModelEvaluator
# --------------------------------------------------------------------------------

def create_multi_check_combobox(items_list):
    """
    Creates a QComboBox where each item is checkable.
    Returns the QComboBox plus a function get_checked_items() 
    that retrieves all checked item texts.
    """
    combo = QComboBox()
    combo.setEditable(True)  # so the user can see what's selected in the line edit (optional)

    model = QStandardItemModel()
    combo.setModel(model)

    for text in items_list:
        item = QStandardItem(text)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setData(Qt.Unchecked, Qt.CheckStateRole)
        model.appendRow(item)

    def get_checked_items():
        """Return a list of all items currently checked in this combo."""
        checked = []
        for i in range(model.rowCount()):
            itm = model.item(i)
            if itm.checkState() == Qt.Checked:
                checked.append(itm.text())
        return checked

    return combo, get_checked_items

class CSVRowWidget(QWidget):

    def __init__(self, remove_callback):
        super().__init__()
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.csv_path_edit = QLineEdit()
        self.csv_path_edit.setPlaceholderText("CSV file path")
        self.nickname_edit = QLineEdit()
        self.nickname_edit.setPlaceholderText("Nickname")

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_csv)

        # We store a reference to the parent's remove function
        self.remove_callback = remove_callback
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self.delete_self)

        layout.addWidget(QLabel("CSV File:"))
        layout.addWidget(self.csv_path_edit)
        layout.addWidget(QLabel("Name:"))
        layout.addWidget(self.nickname_edit)
        layout.addWidget(browse_btn)
        layout.addWidget(delete_btn)

    def browse_csv(self):
        """Open file dialog to pick the CSV file."""
        fname, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if fname:
            self.csv_path_edit.setText(fname)

    def get_csv_info(self):
        """Return (csv_path, nickname)."""
        return self.csv_path_edit.text().strip(), self.nickname_edit.text().strip()

    def delete_self(self):
        """
        Called when the user clicks 'Delete'.
        We notify the parent to remove this row widget from the layout/list.
        """
        self.remove_callback(self)  # calls the parent's method to remove me


class MainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Analysis GUI (Multiple CSV Rows + Multi-Check Combos)")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # ------------------------------------------------------------
        # 1) Metrics Log File
        # ------------------------------------------------------------
        logfile_layout = QFormLayout()
        self.metrics_log_file_input = QLineEdit("results/metrics/Evaluation_log_3.csv")
        logfile_layout.addRow("Metrics Log File:", self.metrics_log_file_input)

        browse_log_btn = QPushButton("Browse Log File")
        browse_log_btn.clicked.connect(self.browse_metrics_logfile)
        logfile_layout.addRow(browse_log_btn)
        main_layout.addLayout(logfile_layout)

        # ------------------------------------------------------------
        # 2) CSV Rows
        # ------------------------------------------------------------
        self.csv_rows_layout = QVBoxLayout()
        # This layout will hold all CSVRowWidgets
        main_layout.addLayout(self.csv_rows_layout)

        # Button to add new CSV row
        add_csv_btn = QPushButton("Add CSV File")
        add_csv_btn.clicked.connect(self.add_csv_row)
        main_layout.addWidget(add_csv_btn)

        # We'll store references to each CSVRowWidget in this list
        self.csv_rows = []

        # Add one row by default
        self.add_csv_row()

        # ------------------------------------------------------------
        # 3) Multi-Check Combos for Ranking, Model, Feature, Average
        # ------------------------------------------------------------
        ranking_method_items = ["WFI-RF", "SP", "WFI-XGB"]
        self.ranking_combo, self.get_ranking_checked = create_multi_check_combobox(ranking_method_items)

        model_type_items = ["XGB", "RF"]
        self.model_types_combo, self.get_model_types_checked = create_multi_check_combobox(model_type_items)

        fs_items = ["RFE", "SPFS", "None"]
        self.feature_selection_combo, self.get_fs_checked = create_multi_check_combobox(fs_items)

        avg_items = ["micro", "weighted"]
        self.average_type_combo, self.get_avg_checked = create_multi_check_combobox(avg_items)

        # Place them in a form layout (or any layout you prefer)
        combo_form = QFormLayout()
        combo_form.addRow("Ranking Methods:", self.ranking_combo)
        combo_form.addRow("Model Types:", self.model_types_combo)
        combo_form.addRow("Feature Selection:", self.feature_selection_combo)
        combo_form.addRow("Average Types:", self.average_type_combo)

        main_layout.addLayout(combo_form)

        # ------------------------------------------------------------
        # 4) Metrics Checkboxes
        # ------------------------------------------------------------
        metrics_box = QGroupBox("Select Metrics to Include for Ranking")
        metrics_layout = QVBoxLayout()
        metrics_box.setLayout(metrics_layout)

        self.use_std_dev_cb = QCheckBox("Use Standard Deviation")
        self.use_abs_diff_cb = QCheckBox("Use Absolute Difference")
        self.use_skew_cb = QCheckBox("Use Skewness")
        self.use_kurt_cb = QCheckBox("Use Kurtosis")

        metrics_layout.addWidget(self.use_std_dev_cb)
        metrics_layout.addWidget(self.use_abs_diff_cb)
        metrics_layout.addWidget(self.use_skew_cb)
        metrics_layout.addWidget(self.use_kurt_cb)

        main_layout.addWidget(metrics_box)

        # ------------------------------------------------------------
        # 5) Run Pipeline Button
        # ------------------------------------------------------------
        run_button = QPushButton("Run Analysis Pipeline")
        run_button.clicked.connect(self.run_pipeline)
        main_layout.addWidget(run_button)

        # ------------------------------------------------------------
        # 6) Log Output
        # ------------------------------------------------------------
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        main_layout.addWidget(self.log_output)

        # Default model directory
        self.model_dir = "results/models/"

    # ---------------------------
    # CSV Row Management
    # ---------------------------
    def add_csv_row(self):
        """Create a new CSVRowWidget and add it to csv_rows_layout."""
        row_widget = CSVRowWidget(remove_callback=self.remove_csv_row)
        self.csv_rows_layout.addWidget(row_widget)
        self.csv_rows.append(row_widget)

    def remove_csv_row(self, row_widget):
        self.csv_rows_layout.removeWidget(row_widget)
        self.csv_rows.remove(row_widget)
        row_widget.deleteLater()

    # ---------------------------
    # File Browsing
    # ---------------------------
    def browse_metrics_logfile(self):
        """Open file dialog to pick (or create) a metrics log file path."""
        fname, _ = QFileDialog.getSaveFileName(self, "Select Metrics Log File", "", "CSV Files (*.csv)")
        if fname:
            self.metrics_log_file_input.setText(fname)

    # ---------------------------
    # Run Pipeline
    # ---------------------------
    def run_pipeline(self):
        """
        For each (csv_path, nickname) from all CSVRowWidgets,
        and for each checked ranking/methods combos, run the pipeline.
        """
        # Gather (csv_path, nickname) pairs
        csv_pairs = []
        for row_widget in self.csv_rows:
            csv_path, nickname = row_widget.get_csv_info()
            if csv_path:
                # If no nickname, fallback to something
                nickname = nickname if nickname else "default_nickname"
                csv_pairs.append((csv_path, nickname))

        if not csv_pairs:
            QMessageBox.warning(self, "No CSV Files", "Please add at least one CSV file.")
            return

        # Get metrics log file
        metrics_log_file = self.metrics_log_file_input.text().strip()
        if not metrics_log_file:
            QMessageBox.warning(self, "Missing Log File", "Please specify a metrics log file.")
            return

        # Collect selected items from combos
        ranking_methods_selected = self.get_ranking_checked()
        model_types_selected = self.get_model_types_checked()
        feature_selection_selected = self.get_fs_checked()
        average_types_selected = self.get_avg_checked()

        # Collect metric checkboxes
        use_std_dev = self.use_std_dev_cb.isChecked()
        use_abs_diff = self.use_abs_diff_cb.isChecked()
        use_skewness = self.use_skew_cb.isChecked()
        use_kurtosis = self.use_kurt_cb.isChecked()

        self.log_output.append("=== Starting Analysis Pipeline ===\n")
        self.log_output.append(f"Metrics Log File: {metrics_log_file}\n")
        self.log_output.append(f"Ranking Methods: {ranking_methods_selected}")
        self.log_output.append(f"Model Types: {model_types_selected}")
        self.log_output.append(f"Feature Selection: {feature_selection_selected}")
        self.log_output.append(f"Average Types: {average_types_selected}\n")

        dataset_start_time = time.time()

        # Loop over each CSV file
        for (csv_path, dataset_name) in csv_pairs:
            self.log_output.append(f"\n>>> Processing CSV: {csv_path}")
            self.log_output.append(f"Nickname: {dataset_name}")

            start_time_dataset = time.time()

            # Now replicate the nested loops from your original approach
            for ranking_method in ranking_methods_selected:
                start_time_ranking = time.time()

                ranking_file = f"results/Rankings/Ranking_{ranking_method}_{dataset_name}.csv"
                os.makedirs("results/Rankings", exist_ok=True)

                # 1) RANK FEATURES
                ranker = DataAnalysisRanker(
                    ranking_file=ranking_file,
                    input_file=csv_path,
                    output_file=ranking_file
                )
                ranker.analyze(
                    rank_method=ranking_method,
                    use_std_dev=use_std_dev,
                    use_abs_diff=use_abs_diff,
                    use_skewness=use_skewness,
                    use_kurtosis=use_kurtosis
                )
                ranker.save_results()

                ranking_time_taken = time.time() - start_time_ranking
                self.log_output.append(
                    f"[{dataset_name} | {ranking_method}] Ranking -> {ranking_file} ({ranking_time_taken:.2f}s)"
                )

                for model_type in model_types_selected:
                    for fs_method in feature_selection_selected:
                        # 2) FEATURE SELECTION
                        start_time_selection = time.time()

                        selector = FeatureSelector(
                            dataset_name=dataset_name,
                            ranking_file=ranking_file,
                            input_file=csv_path,
                            model_type=model_type
                        )
                        selected_features = selector.perform_feature_selection(fs_method)
                        selection_time_taken = time.time() - start_time_selection

                        self.log_output.append(
                            f"[{dataset_name} | {ranking_method} | {model_type} | {fs_method}] "
                            f"Selected {len(selected_features)} features in {selection_time_taken:.2f}s"
                        )

                        for avg_type in average_types_selected:
                            # 3) MODEL EVALUATION
                            model_filename = f"{dataset_name}_{model_type}_{fs_method}.pkl"
                            model_path = os.path.join(self.model_dir, model_filename)

                            evaluator = ModelEvaluator(
                                original_file=csv_path,
                                input_file=csv_path,
                                average_type=avg_type,
                                selected_features=selected_features,
                                model_type=model_type,
                                feature_selection_method=fs_method,
                                model_path=model_path,
                                rank_method=ranking_method,
                                metrics_log_file=metrics_log_file
                            )

                            eval_time_taken = evaluator.train_and_evaluate(ranking_time_taken, selection_time_taken)
                            total_time_for_this_combo = ranking_time_taken + selection_time_taken + eval_time_taken

                            self.log_output.append(
                                f"--> [{dataset_name} | {ranking_method} | {model_type} | {fs_method} | {avg_type}] "
                                f"Done in {total_time_for_this_combo:.2f}s"
                            )

            dataset_elapsed = time.time() - start_time_dataset
            self.log_output.append(f"=== Done with {dataset_name} in {dataset_elapsed:.2f} seconds. ===")

        total_time = time.time() - dataset_start_time
        self.log_output.append(f"\n=== All CSVs processed in {total_time:.2f} seconds. ===\n")


def main():
    app = QApplication(sys.argv)
    gui = MainGUI()
    gui.resize(1050, 750)
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
