import cv2
import numpy as np
import os

def background_substraction(img, background, multiplier=3, clip=False):
    """Substracts the background from an image."""
    diff_img = cv2.absdiff(img, background) * multiplier
    if clip:
        diff_img = np.clip(diff_img, -127, 128) + np.ones_like(diff_img) * 127
    display_img = cv2.cvtColor(diff_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
    return display_img, mask

def convert_to_grayscale_cv2(image):
    """
    Converts the given image to grayscale.

    :param image: Image to be converted
    :return: Grayscale image
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def save_image_dataset(image, calib_dir, ball_diameter, data_count):
    """
    Organiza y guarda una imagen en la estructura de directorios esperada.

    Args:
        image (np.ndarray): La imagen a guardar.
        calib_dir (str): Directorio principal de calibración.
        ball_diameter (float): Diámetro del balín en milímetros.
        data_count (int): Número de experimento para este diámetro.
    """
    # Crear directorio principal si no existe
    os.makedirs(calib_dir, exist_ok=True)

    # Crear subdirectorio del diámetro
    indenter_subdir = ball_diameter
    indenter_dir = os.path.join(calib_dir, indenter_subdir)
    os.makedirs(indenter_dir, exist_ok=True)

    # Crear subdirectorio del experimento
    experiment_dir = os.path.join(indenter_dir, str(data_count))
    os.makedirs(experiment_dir, exist_ok=True)

    # Guardar la imagen como digit.png
    save_path = os.path.join(experiment_dir, "gelsight.png")
    cv2.imwrite(save_path, image)
    print(f"Imagen guardada en: {save_path}")

    # Actualizar el archivo catalog.csv
    catalog_path = os.path.join(calib_dir, "catalog.csv")
    if not os.path.isfile(catalog_path):
        with open(catalog_path, "w") as f:
            f.write("experiment_reldir,diameter(mm)\n")

    experiment_reldir = os.path.join(indenter_subdir, str(data_count))
    with open(catalog_path, "a") as f:
        f.write(f"{experiment_reldir},{ball_diameter}\n")