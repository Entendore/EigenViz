import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QSlider, QLabel, QGroupBox, QPushButton,
                              QFileDialog, QProgressDialog, QMessageBox, QComboBox,
                              QGridLayout)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import imageio

# Set a dark theme for matplotlib to look good on YouTube
plt.style.use('dark_background')

class EigenVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eigenvalues & Eigenvectors Visualizer")
        self.resize(1280, 820)
        self.setStyleSheet("background-color: #121212; color: #e0e0e0; font-size: 14px; font-family: Segoe UI;")

        # Main Widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left side: Matplotlib Canvas
        self.canvas = EigenCanvas(self, width=800, height=800)
        main_layout.addWidget(self.canvas, stretch=3)

        # Right side: Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setAlignment(Qt.AlignTop)
        main_layout.addWidget(controls_widget, stretch=1)

        # Title Label
        title_label = QLabel("Linear Transformation Visualizer")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #00ffcc; padding-bottom: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        controls_layout.addWidget(title_label)

        # --- PRESETS ---
        preset_group = QGroupBox("Matrix Presets")
        preset_layout = QVBoxLayout(preset_group)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom", "Identity", "Scaling (2x, 0.5x)", "Shear", "Rotation (Complex λ)", "Reflection"])
        self.preset_combo.currentIndexChanged.connect(self.apply_preset)
        preset_layout.addWidget(self.preset_combo)
        controls_layout.addWidget(preset_group)

        # --- MATRIX SLIDERS (2x2 GRID) ---
        matrix_group = QGroupBox("Matrix Elements (A)")
        matrix_grid = QGridLayout(matrix_group)
        
        # Brackets for visual matrix look
        lbl_left1 = QLabel("[")
        lbl_left1.setStyleSheet("font-size: 40px; font-weight: light;")
        lbl_left2 = QLabel("[")
        lbl_left2.setStyleSheet("font-size: 40px; font-weight: light;")
        lbl_right1 = QLabel("]")
        lbl_right1.setStyleSheet("font-size: 40px; font-weight: light;")
        lbl_right2 = QLabel("]")
        lbl_right2.setStyleSheet("font-size: 40px; font-weight: light;")

        self.sliders = {}
        self.value_labels = {}
        elements = [('a', 0, 0), ('b', 0, 1), ('c', 1, 0), ('d', 1, 1)]
        
        for elem, row, col in elements:
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-100, 100)
            slider.setValue(0 if elem in ['b', 'c'] else 50)
            slider.valueChanged.connect(self.update_plot)
            self.sliders[elem] = slider
            
            val_label = QLabel("0.00")
            val_label.setFixedWidth(40)
            val_label.setStyleSheet("font-weight: bold; color: #00ffcc;")
            self.value_labels[elem] = val_label

            # Add to grid: Slider then Value Label
            cell_widget = QWidget()
            cell_layout = QHBoxLayout(cell_widget)
            cell_layout.setContentsMargins(0,0,0,0)
            cell_layout.addWidget(slider)
            cell_layout.addWidget(val_label)
            
            matrix_grid.addWidget(cell_widget, row, col)
            
        # Add brackets to outer columns
        matrix_grid.addWidget(lbl_left1, 0, 0)
        matrix_grid.addWidget(lbl_left2, 1, 0)
        matrix_grid.addWidget(lbl_right1, 0, 3)
        matrix_grid.addWidget(lbl_right2, 1, 3)
        
        controls_layout.addWidget(matrix_group)

        # --- VECTOR CONTROLS ---
        vector_group = QGroupBox("Input Vector (v)")
        vector_layout = QVBoxLayout(vector_group)
        
        self.angle_slider = QSlider(Qt.Horizontal)
        self.angle_slider.setRange(0, 360)
        self.angle_slider.setValue(45)
        self.angle_slider.valueChanged.connect(self.update_plot)
        vector_layout.addWidget(self.angle_slider)
        
        # Play button for animation
        btn_row = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play Sweep")
        self.play_btn.setCheckable(True)
        self.play_btn.setStyleSheet("QPushButton { background-color: #333333; padding: 5px; border-radius: 3px; } QPushButton:checked { background-color: #00ffcc; color: black; }")
        self.play_btn.clicked.connect(self.toggle_play)
        btn_row.addWidget(self.play_btn)
        
        self.angle_label = QLabel("45°")
        self.angle_label.setAlignment(Qt.AlignCenter)
        self.angle_label.setStyleSheet("font-weight: bold; color: #ffcc00; font-size: 16px;")
        btn_row.addWidget(self.angle_label)
        vector_layout.addLayout(btn_row)
        
        controls_layout.addWidget(vector_group)

        # --- MATH OUTPUT ---
        math_group = QGroupBox("Eigen-Math")
        math_layout = QVBoxLayout(math_group)
        
        self.eq_label = QLabel("Av = λv")
        self.eq_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        self.eq_label.setAlignment(Qt.AlignCenter)
        math_layout.addWidget(self.eq_label)
        
        self.lambda_label = QLabel("λ₁ = N/A\nλ₂ = N/A")
        self.lambda_label.setStyleSheet("font-size: 16px; color: #00ffcc;")
        math_layout.addWidget(self.lambda_label)
        
        self.vector_label = QLabel("v₁ = N/A\nv₂ = N/A")
        self.vector_label.setStyleSheet("font-size: 16px; color: #ff0055;")
        math_layout.addWidget(self.vector_label)

        self.dynamic_label = QLabel("v = [0.00, 0.00]\nAv = [0.00, 0.00]")
        self.dynamic_label.setStyleSheet("font-size: 14px; color: #ffcc00;")
        math_layout.addWidget(self.dynamic_label)
        
        controls_layout.addWidget(math_group)
        
        # --- RENDER BUTTON ---
        self.render_btn = QPushButton("🎥 Render Sweep to MP4")
        self.render_btn.setStyleSheet("""
            QPushButton {
                background-color: #cc0000; color: white; 
                font-weight: bold; padding: 12px; 
                border-radius: 5px; font-size: 15px;
            }
            QPushButton:hover { background-color: #ff0000; }
        """)
        self.render_btn.clicked.connect(self.start_render)
        controls_layout.addWidget(self.render_btn)

        # Animation Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate_angle)
        self.anim_direction = 1

        # Initial plot
        self.update_plot()

    def apply_preset(self, index):
        text = self.preset_combo.currentText()
        
        # Block signals to prevent spamming update_plot
        for s in self.sliders.values(): s.blockSignals(True)
            
        if text == "Identity":
            self.sliders['a'].setValue(50); self.sliders['b'].setValue(0)
            self.sliders['c'].setValue(0); self.sliders['d'].setValue(50)
        elif text == "Scaling (2x, 0.5x)":
            self.sliders['a'].setValue(100); self.sliders['b'].setValue(0)
            self.sliders['c'].setValue(0); self.sliders['d'].setValue(25)
        elif text == "Shear":
            self.sliders['a'].setValue(50); self.sliders['b'].setValue(50)
            self.sliders['c'].setValue(0); self.sliders['d'].setValue(50)
        elif text == "Rotation (Complex λ)":
            self.sliders['a'].setValue(0); self.sliders['b'].setValue(-50)
            self.sliders['c'].setValue(50); self.sliders['d'].setValue(0)
        elif text == "Reflection":
            self.sliders['a'].setValue(50); self.sliders['b'].setValue(0)
            self.sliders['c'].setValue(0); self.sliders['d'].setValue(-50)
            
        for s in self.sliders.values(): s.blockSignals(False)
        self.update_plot()

    def toggle_play(self):
        if self.play_btn.isChecked():
            self.timer.start(30)  # ~30fps
            self.play_btn.setText("⏸ Pause Sweep")
        else:
            self.timer.stop()
            self.play_btn.setText("▶ Play Sweep")

    def animate_angle(self):
        val = self.angle_slider.value()
        if val >= 360:
            self.anim_direction = -1
        elif val <= 0:
            self.anim_direction = 1
        self.angle_slider.setValue(val + self.anim_direction)

    def update_plot(self):
        # Read matrix values
        a = self.sliders['a'].value() / 50.0
        b = self.sliders['b'].value() / 50.0
        c = self.sliders['c'].value() / 50.0
        d = self.sliders['d'].value() / 50.0
        
        A = np.array([[a, b], [c, d]])
        
        # Update UI labels
        self.value_labels['a'].setText(f"{a:.2f}")
        self.value_labels['b'].setText(f"{b:.2f}")
        self.value_labels['c'].setText(f"{c:.2f}")
        self.value_labels['d'].setText(f"{d:.2f}")
        
        angle = self.angle_slider.value()
        self.angle_label.setText(f"{angle}°")
        
        # Calculate Eigenvalues and Eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eig(A)
            
            angle_rad = np.deg2rad(angle)
            v = np.array([np.cos(angle_rad), np.sin(angle_rad)]) * 3
            Av = A @ v
            
            self.dynamic_label.setText(f"v  = [{v[0]:.2f}, {v[1]:.2f}]\nAv = [{Av[0]:.2f}, {Av[1]:.2f}]")
            
            if np.isreal(eigenvalues).all():
                lambda1 = eigenvalues[0]
                lambda2 = eigenvalues[1]
                self.lambda_label.setText(f"λ₁ = {lambda1:.3f}\nλ₂ = {lambda2:.3f}")
                
                v1 = eigenvectors[:, 0]
                v2 = eigenvectors[:, 1]
                self.vector_label.setText(f"v₁ = [{v1[0]:.2f}, {v1[1]:.2f}]\nv₂ = [{v2[0]:.2f}, {v2[1]:.2f}]")
                
                self.canvas.update_visualization(A, angle, eigenvalues, eigenvectors)
            else:
                self.lambda_label.setText("λ₁, λ₂ = Complex\n(No real eigenvectors)")
                self.vector_label.setText("v₁, v₂ = N/A")
                self.canvas.update_visualization(A, angle, None, None)
                
        except np.linalg.LinAlgError:
            pass

    def start_render(self):
        # Stop live animation if running
        if self.play_btn.isChecked():
            self.toggle_play()

        file_path, _ = QFileDialog.getSaveFileName(self, "Save MP4 Video", "eigen_animation.mp4", "MP4 Files (*.mp4)")
        if not file_path:
            return

        self.progress = QProgressDialog("Rendering frames...", "Cancel", 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setWindowTitle("Rendering to MP4")
        self.progress.show()

        # We do rendering synchronously but process events to keep UI alive
        frames = []
        total_frames = 180
        for i, angle in enumerate(range(0, 361, 2)):
            if self.progress.wasCanceled():
                break
                
            self.angle_slider.setValue(angle)
            QApplication.processEvents() # Keep UI responsive and force canvas draw
            
            # Grab Canvas
            qimg = self.canvas.grab().toImage().convertToFormat(QImage.Format_RGB888)
            ptr = qimg.bits()
            ptr.setsize(qimg.sizeInBytes())
            arr = np.array(ptr, dtype=np.uint8).reshape((qimg.height(), qimg.width(), 3))
            frames.append(arr)
            
            self.progress.setValue(int((i / total_frames) * 100))

        if not self.progress.wasCanceled():
            try:
                self.progress.setLabelText("Encoding video...")
                QApplication.processEvents()
                imageio.mimsave(file_path, frames, fps=30, quality=9, macro_block_size=1)
                self.progress.close()
                QMessageBox.information(self, "Success", f"Video saved successfully to:\n{file_path}")
            except Exception as e:
                self.progress.close()
                QMessageBox.critical(self, "Error", f"Failed to save video:\n{str(e)}")
        else:
            self.progress.close()


class EigenCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setup_axes()

    def setup_axes(self):
        self.ax.set_xlim(-6, 6)
        self.ax.set_ylim(-6, 6)
        self.ax.set_aspect('equal')
        self.ax.grid(True, color='#333333', linestyle='--', alpha=0.5)
        self.ax.axhline(0, color='#666666', lw=1.5)
        self.ax.axvline(0, color='#666666', lw=1.5)
        self.ax.set_facecolor('#1e1e1e')
        self.fig.patch.set_facecolor('#1e1e1e')

    def update_visualization(self, A, angle_deg, eigenvalues, eigenvectors):
        self.ax.clear()
        self.setup_axes()
        
        # 1. Draw Transformed Grid
        x = np.linspace(-5, 5, 11)
        y = np.linspace(-5, 5, 11)
        for xi in x:
            points = np.array([[xi, yi] for yi in y])
            transformed = points @ A.T
            self.ax.plot(transformed[:, 0], transformed[:, 1], color='#3a3a8a', alpha=0.4, lw=1)
        for yi in y:
            points = np.array([[xi, yi] for xi in x])
            transformed = points @ A.T
            self.ax.plot(transformed[:, 0], transformed[:, 1], color='#3a3a8a', alpha=0.4, lw=1)

        # 2. Draw Original Unit Circle (Radius 3 for visibility)
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta) * 3
        circle_y = np.sin(theta) * 3
        self.ax.plot(circle_x, circle_y, color='#ffffff', linestyle='-', lw=2, alpha=0.5, label='Original Space')

        # 3. Draw Transformed Unit Circle (The Ellipse)
        circle_points = np.vstack((circle_x, circle_y))
        transformed_circle = A @ circle_points
        self.ax.plot(transformed_circle[0, :], transformed_circle[1, :], color='#ffcc00', linestyle='-', lw=3, alpha=0.8, label='Transformed Space')

        # 4. Draw Eigenlines (if real)
        if eigenvalues is not None:
            v1 = eigenvectors[:, 0]
            v2 = eigenvectors[:, 1]
            
            # Normalize and scale for drawing the infinite line
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            
            self.ax.plot([-6*v1_norm[0], 6*v1_norm[0]], [-6*v1_norm[1], 6*v1_norm[1]], 
                         color='#00ffcc', linestyle='--', lw=2.5, alpha=0.9, label='Eigenvector 1')
            self.ax.plot([-6*v2_norm[0], 6*v2_norm[0]], [-6*v2_norm[1], 6*v2_norm[1]], 
                         color='#ff0055', linestyle='--', lw=2.5, alpha=0.9, label='Eigenvector 2')

        # 5. Draw Input Vector v
        angle_rad = np.deg2rad(angle_deg)
        v = np.array([np.cos(angle_rad), np.sin(angle_rad)]) * 3
        self.ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, 
                       color='#ffffff', width=0.012, headwidth=5)
        
        # 6. Draw Transformed Vector Av
        Av = A @ v
        self.ax.quiver(0, 0, Av[0], Av[1], angles='xy', scale_units='xy', scale=1, 
                       color='#ff9900', width=0.015, headwidth=5)
        
        self.ax.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#666666', fontsize=9)
        self.draw_idle()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EigenVisualizer()
    window.show()
    sys.exit(app.exec())