import numpy as np


def read_fonts_h(file_path: str) -> np.ndarray:
    font_data = read_from_file(file_path)
    return convert_dataset(font_data)


def convert_dataset(font_data):
    dataset = []
    for pattern in font_data:
        flattened_pattern = [int(bit) for row in pattern for bit in row]
        dataset.append(flattened_pattern)
    return np.array(dataset)


def read_from_file(file_path: str) -> np.ndarray:
    font_data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Strip inline comments and leading/trailing whitespace
            line = line.split('//')[0].strip()

            # Check if the remaining part of the line contains font data
            if line.startswith('{') and line.endswith('},'):
                # Extract the hexadecimal numbers
                hex_values = line.strip('{').strip('},').split(',')
                # Convert hex values to binary and store them
                binary_values = [format(int(hx.strip(), 16), '05b') for hx in hex_values]
                font_data.append(binary_values)
    return np.array(font_data)
