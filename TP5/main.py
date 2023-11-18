import matplotlib.pyplot as plt

def read_font_data(file_path):
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
    return font_data

# Replace 'path/to/font.h' with the actual path to your 'font.h' file
font_patterns = read_font_data('TP5/font.h')

# Print the binary patterns
for pattern in font_patterns:
    for row in pattern:
        print(' '.join(row))
    print('----------')

